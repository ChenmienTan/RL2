from typing import Dict, Tuple, List
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from RL2.datasets import pack_tensor_dicts
from RL2.utils.logging import time_logger

def compute_approx_kl(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    estimator: str
) -> torch.Tensor:
    # logps of non-action tokens are zeros (see `Actor.forward`)
    # so their corresponding approx_kl will also be zero.

    log_ratio = logps - ref_logps
    if estimator == "k1":
        return log_ratio
    elif estimator == "k2":
        return log_ratio.pow(2) / 2
    elif estimator == "k3":
        return log_ratio + torch.exp(- log_ratio) - 1
    else:
        raise NotImplementedError

def _compute_gae(
    tensor_dict: Dict[str, torch.Tensor], gamma: float, lamda: float
) -> Dict[str, torch.Tensor]:
    
    # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    next_values = F.pad(tensor_dict["old_values"][:, 1:], (0, 1), value=0)
    deltas = tensor_dict["rewards"] + gamma * next_values - tensor_dict["old_values"]

    # A_t = \delta_t + \gamma * \lambda * A_{t+1}
    gae, reversed_gaes = 0, []
    for t in reversed(range(deltas.shape[-1])):
        gae = deltas[:, t] + gamma * lamda * gae
        reversed_gaes.append(gae)
    gaes = torch.stack(reversed_gaes[::-1], -1)
    returns = gaes + tensor_dict["old_values"]

    action_gaes = gaes[torch.where(tensor_dict["action_mask"])]
    advantages = (gaes - action_gaes.mean()) * tensor_dict["action_mask"] / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    return {"advantages": advantages, "returns": returns}

def _compute_reinforce_adv(
    tensor_dict: Dict[str, torch.Tensor],
    responses_per_prompt: int,
    global_norm: bool,
    norm_var: bool
) -> Dict[str, torch.Tensor]:
    
    rewards = tensor_dict["rewards"].sum(-1).view(-1, responses_per_prompt)

    if global_norm:
        baseline = rewards.mean()
        std = rewards.std()
    else:
        baseline = rewards.mean(-1, keepdim=True)
        std = rewards.std(-1, keepdim=True)

    advantages = rewards - baseline
    if norm_var:
        advantages /= (
            std + torch.finfo(advantages.dtype).eps
        )

    advantages = advantages.view(-1, 1) * tensor_dict["action_mask"]
    return {"advantages": advantages}

@time_logger("compute_advantages")
def compute_advantages(
    config: DictConfig,
    tensor_dict: Dict[str, torch.Tensor],
    cu_seqs: torch.Tensor,
    step: int
):

    def extract_actions(
        tensor_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        indices = torch.where(tensor_dict["action_mask"])
        return {
            k: v[indices]
            for k, v in tensor_dict.items()
        }
    
    processed_tensor_dict = pack_tensor_dicts([
        extract_actions(
            {
                k: v[start:end]
                for k, v in tensor_dict.items()
            }
        )
        for start, end in zip(cu_seqs[:-1], cu_seqs[1:])
    ])

    if config.estimator == "gae":
        tensor_dict_delta = _compute_gae(
            processed_tensor_dict, config.gamma, config.lamda
        )
    elif config.estimator == "reinforce":
        tensor_dict_delta = _compute_reinforce_adv(
            processed_tensor_dict,
            config.responses_per_prompt,
            config.global_norm,
            config.norm_var
        )
    else:
        raise NotImplementedError

    for k, v in tensor_dict_delta.items():
        tensor_dict[k] = torch.zeros(tensor_dict["states"].shape)
        for idx, (start, end) in enumerate(
            zip(cu_seqs[:-1], cu_seqs[1:])
        ):
            indices = torch.where(tensor_dict["action_mask"][start:end])
            tensor_dict[k][start:end][indices] = v[idx][:len(indices[0])]

def rm_loss(
    minibatch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:

    chosen_rewards, rejected_rewards = minibatch["values"].sum(-1).view(-1, 2).T
    reward_margins = chosen_rewards - rejected_rewards
    losses = - F.logsigmoid(reward_margins)
    return losses, {"accuracy": (reward_margins > 0).tolist()}

def dpo_loss(
    config: DictConfig, minibatch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:

    chosen_rewards, rejected_rewards = config.beta * (
        minibatch["logps"] - minibatch["ref_logps"]
    ).sum(-1).view(-1, 2).T
    reward_margins = chosen_rewards - rejected_rewards
    losses = - F.logsigmoid(reward_margins)
    metric = {
        "rewards/chosen": chosen_rewards.tolist(),
        "rewards/rejected": rejected_rewards.tolist(),
        "rewards/margin": reward_margins.tolist(),
        "accuracy": (reward_margins > 0).tolist()
    }
    return losses, metric

def actor_ppo_loss(
    config: DictConfig, minibatch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:

    ratio = torch.exp(
        minibatch["logps"] - minibatch.get(
            "old_logps", minibatch["logps"].detach()
        )
        # We do not compute old_logps
        # if kl.coef == 0 and update_per_rollout == 1
    )
    clipped_ratio = torch.clamp(
        ratio, 1 - config.clip, 1 + config.clip
    )
    objective = minibatch["advantages"] * ratio
    clipped_objective = minibatch["advantages"] * clipped_ratio
    losses = - torch.min(objective, clipped_objective)
    clip_ratios = objective > clipped_objective

    if config.kl.coef > 0 and config.kl.type == "loss":
        kl_losses = compute_approx_kl(
            minibatch["logps"],
            minibatch["ref_logps"],
            config.kl.loss_estimator
        )
        losses = losses + config.kl.coef * kl_losses

    if config.tis_coef > 0:
        # https://fengyao.notion.site/off-policy-rl
        tis = torch.exp(
            minibatch["logps"].detach() - minibatch["llm_logps"]
        ).clamp(max=config.tis_coef)
        losses *= tis

    losses = losses - config.entropy.coef * minibatch["entropy"]
    return losses, clip_ratios

def critic_ppo_loss(
    config: DictConfig, minibatch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:

    clipped_values = torch.clamp(
        minibatch["values"],
        minibatch["old_values"] - config.clip,
        minibatch["old_values"] + config.clip
    )
    mse = (minibatch["values"] - minibatch["returns"]).pow(2)
    clipped_mse = (clipped_values - minibatch["returns"]).pow(2)
    losses = torch.max(mse, clipped_mse)
    clip_ratios = mse < clipped_mse
    return losses, clip_ratios