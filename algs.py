from typing import List, Dict, Optional
import torch

def accumulate_to_eos(
    value: torch.Tensor,
    eos_mask: torch.Tensor
) -> torch.Tensor:
    # Example:
    #   - value: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    #   - eos_mask: [0, 0, 1, 0, 1, 0, 1]
    #   - output: [0.0, 0.0, 0.6, 0.0, 0.9, 0.0, 1.3]
    
    end_indices = torch.where(eos_mask)[1]
    start_indices = torch.cat((
        torch.LongTensor([0]).to(end_indices.device),
        end_indices[:-1] + 1
    ))

    result = torch.zeros_like(value)
    for start_idx, end_idx in zip(start_indices, end_indices):
        result[0, end_idx] = value[0, start_idx:end_idx + 1].sum()
    return result

def compute_kl_term(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    kl_estimator: str,
    eos_mask: Optional[torch.Tensor]=None
) -> torch.Tensor:
    # The (ref_)logps of non-action tokens are zero (see `Actor.
    # forward`), so their corresponding kl_term will also be zero.
    
    if eos_mask is not None:
        # If eos_mask is provided, we firstly compute the logp of 
        # the sequence. This corresponds to `kl_level=sequence`.
        logps = accumulate_to_eos(logps, eos_mask)
        ref_logps = accumulate_to_eos(logps, eos_mask)

    logp_diffs = logps - ref_logps
    if kl_estimator == "k1":
        return logp_diffs
    elif kl_estimator == "k2":
        return logp_diffs.pow(2) / 2
    elif kl_estimator == "k3":
        return logp_diffs + torch.exp(- logp_diffs) - 1
    else:
        raise NotImplementedError

def compute_gae(
    data_list: List[Dict[str, torch.Tensor]],
    gamma: float,
    lamda: float
):

    for ex in data_list:

        # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
        # if s_{t+1} is a terminal state, V(s_{t+1}) = 0
        next_values = torch.cat(
            (ex["values"][:, 1:], torch.FloatTensor([[0]])),
        dim=-1)
        delta = ex["rewards"] + gamma * next_values - ex["values"]

        # A_t = \delta_t + \gamma * \lambda * A_{t+1}
        # if s_{t+1} is a terminal state, A_{t+1} = 0
        gae, reversed_gaes = 0, []
        for t in reversed(range(delta.shape[-1])):
            gae = delta[0, t] + gamma * lamda * gae
            reversed_gaes.append(gae)
        gaes = reversed_gaes[::-1]

        ex["advantages"] = torch.FloatTensor([gaes]) * ex["action_mask"]
        ex["returns"] = ex["advantages"] + ex["values"]

def compute_reinforce_adv(
    data_list: List[Dict[str, torch.Tensor]],
    rollout_per_prompt: int,
    norm_var: bool
):

    rewards = torch.FloatTensor(
        [ex["rewards"].sum() for ex in data_list]
    ).view(-1, rollout_per_prompt)
    baselines = rewards.mean(-1, keepdim=True)
    advantages = rewards - baselines
    if norm_var:
        stds = rewards.std(-1, keepdim=True)
        advantages /= (stds + torch.finfo(stds.dtype).eps)
    advantages = advantages.flatten()

    for ex, advantage in zip(data_list, advantages):
        ex["advantages"] = advantage * ex["action_mask"]