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

def compute_gae(tensor_dicts, gamma, lamda):

    # extract rewards and values of action tokens
    rewards, values, action_mask = [], [], []
    for td in tensor_dicts:
        indices = torch.where(td["action_mask"])[0]
        rewards.append(td["rewards"][indices])
        values.append(td["values"][indices])
        action_mask.append(td["action_mask"][indices])
    # pad to identical length for efficient computation
    rewards = pad_sequence(rewards, True)
    values = pad_sequence(values, True)
    action_mask = pad_sequence(action_mask, True)
    
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

    action_gaes = gaes[torch.where(action_mask)]
    gaes = (gaes - action_gaes.mean()) * action_mask / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    for td, gae, ret in zip(tensor_dicts, gaes, returns):
        td["advantages"] = torch.zeros_like(td["rewards"])
        td["returns"] = torch.zeros_like(td["rewards"])
        indices = torch.where(td["action_mask"])[0]
        td["advantages"][indices] = gae[:len(indices)]
        td["returns"][indices] = ret[:len(indices)]

def compute_reinforce_adv(
    tensor_dicts,
    responses_per_prompt,
    global_norm: bool,
    norm_var: bool
):
    rewards = torch.FloatTensor(
        [td["rewards"].sum() for td in tensor_dicts]
    ).view(-1, responses_per_prompt)

    if global_norm:
        baseline = rewards.mean()
        std = rewards.std()
    else:
        baseline = rewards.mean(-1).unsqueeze(-1)
        std = rewards.std(-1).unsqueeze(-1)

    advantages = rewards - baseline
    if norm_var:
        advantages /= (
            std + torch.finfo(advantages.dtype).eps
        )

    for td, advantage in zip(tensor_dicts, advantages.flatten()):
        td["advantages"] = advantage * td["action_mask"]