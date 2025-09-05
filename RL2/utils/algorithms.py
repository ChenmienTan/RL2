import torch
from torch.nn.utils.rnn import pad_sequence

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

def compute_gae(tensor_dict, gamma, lamda):

    # extract rewards and values of action tokens
    rewards, values, action_masks = [], [], []
    for reward, value, action_mask in zip(
        tensor_dict["rewards"],
        tensor_dict["values"],
        tensor_dict["action_mask"]
    ):
        indices = torch.where(action_mask)[0]
        rewards.append(reward[indices])
        values.append(value[indices])
        action_masks.append(action_mask[indices])

    # pad to identical length for efficient computation
    rewards = pad_sequence(rewards, True)
    values = pad_sequence(values, True)
    action_masks = pad_sequence(action_masks, True)
    
    # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    next_values = torch.cat((
        values[:, 1:],
        torch.zeros((values.shape[0], 1))
    ), -1)
    deltas = rewards + gamma * next_values - values

    # A_t = \delta_t + \gamma * \lambda * A_{t+1}
    gae, reversed_gaes = 0, []
    for t in reversed(range(deltas.shape[-1])):
        gae = deltas[:, t] + gamma * lamda * gae
        reversed_gaes.append(gae)
    gaes = torch.stack(reversed_gaes[::-1], -1)
    returns = gaes + values

    action_gaes = gaes[torch.where(action_masks)]
    gaes = (gaes - action_gaes.mean()) * action_masks / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    tensor_dict["advantages"] = torch.zeros_like(tensor_dict["rewards"])
    tensor_dict["returns"] = torch.zeros_like(tensor_dict["rewards"])
    for advantage, ret, action_mask, gae, action_ret in zip(
        tensor_dict["advantages"],
        tensor_dict["returns"],
        tensor_dict["action_mask"],
        gaes,
        returns
    ):
        indices = torch.where(action_mask)[0]
        advantage[indices] = gae[:len(indices)]
        ret[indices] = action_ret[:len(indices)]

def compute_reinforce_adv(
    tensor_dict,
    responses_per_prompt,
    global_norm: bool,
    norm_var: bool
):
    
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

    tensor_dict["advantages"] = advantages.view(-1, 1) * tensor_dict["action_mask"]