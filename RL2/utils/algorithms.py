import torch
from torch.nn.utils.rnn import pad_sequence
from RL2.utils.functions import sequence_all_reduce

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

def compute_gae(data_list, gamma, lamda):

    # extract rewards and values of action tokens
    rewards, values, action_mask = [], [], []
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        rewards.append(ex["rewards"][indices])
        values.append(ex["values"][indices])
        action_mask.append(ex["action_mask"][indices])
    # pad to identical length for efficient computation
    rewards = pad_sequence(rewards, True)
    values = pad_sequence(values, True)
    action_mask = pad_sequence(action_mask, True)
    
    # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    next_values = torch.cat((values[:, 1:], torch.zeros((values.shape[0], 1))), -1)
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

    for ex, gae, ret in zip(data_list, gaes, returns):
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["returns"] = torch.zeros_like(ex["rewards"])
        indices = torch.where(ex["action_mask"])[0]
        ex["advantages"][indices] = gae[:len(indices)]
        ex["returns"][indices] = ret[:len(indices)]

def compute_reinforce_adv(
    data_list, responses_per_prompt, norm_var: bool
):
    
    rewards = torch.FloatTensor(
        [ex["rewards"].sum() for ex in data_list]
    ).view(-1, responses_per_prompt)
    baseline = rewards.mean(-1)
    advantages = rewards - baseline.unsqueeze(-1)

    if norm_var:
        stds = rewards.std(-1)
        advantages /= (
            stds.unsqueeze(-1) + torch.finfo(advantages.dtype).eps
        )

    for ex, advantage in zip(data_list, advantages.flatten()):
        ex["advantages"] = advantage * ex["action_mask"]

def compute_ppo_loss(worker, logps, minibatch, total_actions):
    
    ratio = torch.exp(
        logps - minibatch.get("old_logps", logps.detach())
    )
    clipped_ratio = torch.clamp(
        ratio, 1 - worker.config.clip, 1 + worker.config.clip
    )
    objective = minibatch["advantages"] * ratio
    clipped_objective = minibatch["advantages"] * clipped_ratio
    loss = - torch.min(objective, clipped_objective).sum() / total_actions
    clip_ratio = (objective > clipped_objective).sum() / total_actions
    
    return loss, clip_ratio

def compute_kimi_loss(worker, logps, minibatch, total_sequences):
    
    # https://arxiv.org/pdf/2501.12599
    kwargs = {
        "cu_seqlens": minibatch["cu_seqlens"],
        "device_mesh": worker.device_mesh["sp"]
    }
    adv = sequence_all_reduce(
        minibatch["eos_mask"] * minibatch["advantages"], **kwargs
    )
    log_ratio = sequence_all_reduce(
        logps - minibatch.get("old_logps", logps.detach()), **kwargs
    )
    loss = (
        adv - worker.config.tau * log_ratio
    ).pow(2).sum() / total_sequences
    return loss, 0.0

def compute_gspo_loss(worker, logps, minibatch, total_sequences):

    # https://arxiv.org/pdf/2507.18071
    kwargs = {
        "cu_seqlens": minibatch["cu_seqlens"],
        "device_mesh": worker.device_mesh["sp"]
    }
    adv = sequence_all_reduce(
        minibatch["eos_mask"] * minibatch["advantages"], **kwargs
    )
    log_ratio = sequence_all_reduce(
        logps - minibatch.get("old_logps", logps.detach()), **kwargs
    ) / sequence_all_reduce(
        minibatch["action_mask"], **kwargs
    )

    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio, 1 - worker.config.clip, 1 + worker.config.clip
    )
    objective = adv * ratio
    clipped_objective = adv * clipped_ratio
    loss = - torch.min(objective, clipped_objective).sum() / total_sequences
    clip_ratio = (objective > clipped_objective).sum() / total_sequences

    return loss, clip_ratio

def compute_surrogate_loss(
    worker, logps, minibatch, total_actions, total_sequences
):
    if worker.config.loss == "ppo":
        return compute_ppo_loss(worker, logps, minibatch, total_actions)
    elif worker.config.loss == "kimi":
        return compute_kimi_loss(worker, logps, minibatch, total_sequences)
    elif worker.config.loss == "gspo":
        return compute_gspo_loss(worker, logps, minibatch, total_sequences)
    else:
        raise NotImplementedError