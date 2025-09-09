import functools
import torch
from RL2.datasets import pack_tensor_dicts

def compute_approx_kl(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    estimator: str
) -> torch.Tensor:
    # The (ref_)logps of non-action tokens are zero (see `Actor.
    # forward`), so their corresponding kl_term will also be zero.

    log_ratio = logps - ref_logps
    if estimator == "k1":
        return log_ratio
    elif estimator == "k2":
        return log_ratio.pow(2) / 2
    elif estimator == "k3":
        return log_ratio + torch.exp(- log_ratio) - 1
    else:
        raise NotImplementedError

def action_extractor(func):

    @functools.wraps(func)
    def compute_adv_with_action_extraction(
        raw_tensor_dict, cu_seqs, *args, **kwargs
    ):
        
        def _extract_actions(tensor_dict):

            indices = torch.where(tensor_dict["action_mask"])
            return {
                k: v[indices]
                for k, v in tensor_dict.items()
            }
        
        tensor_dict = pack_tensor_dicts([
            _extract_actions(
                {
                    k: v[start_idx:end_idx]
                    for k, v in raw_tensor_dict.items()
                }
            )
            for start_idx, end_idx in zip(cu_seqs[:-1], cu_seqs[1:])
        ])

        tensor_dict_delta = func(tensor_dict, *args, **kwargs)
        
        for k, v in tensor_dict_delta.items():
            raw_tensor_dict[k] = torch.zeros(raw_tensor_dict["states"].shape)
            for idx, (start_idx, end_idx) in enumerate(
                zip(cu_seqs[:-1], cu_seqs[1:])
            ):
                indices = torch.where(raw_tensor_dict["action_mask"][start_idx:end_idx])
                raw_tensor_dict[k][start_idx:end_idx][indices] = v[idx][:len(indices[0])]
    
    return compute_adv_with_action_extraction

@action_extractor
def compute_gae(tensor_dict, cu_seqs, gamma, lamda):
    
    # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    next_values = torch.cat((
        tensor_dict["values"][:, 1:],
        torch.zeros((tensor_dict["values"].shape[0], 1))
    ), -1)
    deltas = tensor_dict["rewards"] + gamma * next_values - tensor_dict["values"]

    # A_t = \delta_t + \gamma * \lambda * A_{t+1}
    gae, reversed_gaes = 0, []
    for t in reversed(range(deltas.shape[-1])):
        gae = deltas[:, t] + gamma * lamda * gae
        reversed_gaes.append(gae)
    gaes = torch.stack(reversed_gaes[::-1], -1)
    returns = gaes + tensor_dict["values"]

    action_gaes = gaes[torch.where(tensor_dict["action_mask"])]
    advantages = (gaes - action_gaes.mean()) * tensor_dict["action_mask"] / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    return {"advantages": advantages, "returns": returns}

@action_extractor
def compute_reinforce_adv(
    tensor_dict,
    cu_seqs,
    responses_per_prompt,
    global_norm: bool,
    norm_var: bool
):
    gamma = 1.0
    
    for start_idx, end_idx in zip(cu_seqs[:-1], cu_seqs[1:]):
        trajectory_rewards = tensor_dict["rewards"][start_idx:end_idx]
        trajectory_mask = tensor_dict["action_mask"][start_idx:end_idx]
        
        action_indices = torch.where(trajectory_mask)[0]
        if len(action_indices) > 0:
            action_diff = torch.diff(trajectory_mask.float(), append=torch.tensor([0.0]))
            last_reward_indices = torch.where(action_diff == -1)[0]
            
            if trajectory_mask[-1] == 1:
                last_reward_indices = torch.cat([last_reward_indices, torch.tensor([len(trajectory_mask) - 1])])
            
            if len(last_reward_indices) > 0:
                discounted_reward = trajectory_rewards[last_reward_indices[-1]].item()
                
                for last_idx in reversed(last_reward_indices):
                    tensor_dict["rewards"][start_idx + last_idx] = discounted_reward
                    discounted_reward *= gamma
    
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