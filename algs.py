from typing import List, Dict
import torch
from utils import dispatch_list, all_gather_list

def compute_gae(self, data_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:

    data_list = dispatch_list(data_list, self.device_mesh)

    for ex in data_list:

        # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
        # if s_{t+1} is a terminal state, V(s_{t+1}) = 0
        next_values = torch.cat(
            (ex["values"][:, 1:], torch.FloatTensor([[0]])),
        dim=-1)
        delta = ex["rewards"] + self.config.adv.gamma * next_values - ex["values"]

        # A_t = \delta_t + \gamma * \lambda * A_{t+1}
        # if s_{t+1} is a terminal state, A_{t+1} = 0
        gae, reversed_gaes = 0, []
        for t in reversed(range(delta.shape[-1])):
            gae = delta[0, t] + self.config.adv.gamma * self.config.adv.lamda * gae
            reversed_gaes.append(gae)
        gaes = reversed_gaes[::-1]

        ex["advantages"] = torch.FloatTensor([gaes]) * ex["action_mask"]
        ex["returns"] = ex["advantages"] + ex["values"]

    return all_gather_list(data_list, self.device_mesh)

def compute_reinforce_adv(self, data_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:

    data_list = dispatch_list(data_list, self.device_mesh)

    rewards = torch.FloatTensor(
        [ex["rewards"].sum() for ex in data_list]
    ).view(-1, self.config.actor.rollout.rollout_per_prompt)
    baselines = rewards.mean(-1, keepdim=True)
    advantages = rewards - baselines
    if self.config.adv.group_norm:
        stds = rewards.std(-1, keepdim=True)
        advantages /= (stds + torch.finfo(stds.dtype).eps)
    advantages = advantages.flatten()

    for ex, advantage in zip(data_list, advantages):
        ex["advantages"] = advantage * ex["action_mask"]
    
    return all_gather_list(data_list, self.device_mesh)