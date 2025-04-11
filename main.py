from typing import Dict, List
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler, DataLoader
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import wandb
from data import RLDataset
from actor import Actor
from critic import Critic
from utils import (
    dispatch,
    all_gather,
    accumulate_to_eos,
    compute_kl_term,
    initialize_global_process_group
)


class Trainer:

    def __init__(self, config, world_size):

        self.config = config
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(world_size,)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        self.sampler, self.train_dataloader = self.prepare_sampler_dataloader(
            config.data.train_data_path, True
        )
        _, self.test_dataloader = self.prepare_sampler_dataloader(
            config.data.test_data_path, False
        )

        if config.actor.kl.coef > 0:
            self.ref_actor = Actor(config.actor, self.device_mesh, False)
        if config.adv.estimator == "gae":
            self.critic = Critic(config.critic, self.device_mesh)
        self.actor = Actor(config.actor, self.device_mesh, True)

        if self.device_mesh.get_rank() == 0:
            wandb.init(
                project=self.config.trainer.project,
                name=self.config.trainer.experiment_name,
                config=OmegaConf.to_container(self.config)
            )

    def prepare_sampler_dataloader(self, data_path: str, train: bool):

        dataset = RLDataset(
            data_path,
            self.tokenizer,
            self.config.data.max_prompt_length,
            self.config.data.rollout_per_prompt if train else 1
            # if train, each prompt will be repeated for rollout_per_prompt times
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.device_mesh.size(),
            rank=self.device_mesh.get_rank(),
            shuffle=train,
            drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            (self.config.data.batch_size if train else len(dataset)) // self.device_mesh.size(),
            # if test, pack all data in a single batch
            sampler=sampler,
            collate_fn=dataset.collate_fn
        )

        return sampler, dataloader
    
    def prepare_data_list_for_update(self, old_data_list) -> List[Dict[str, torch.Tensor]]:

        data_list = []
        for ex in old_data_list:

            prompt_id = ex["prompt_id"]
            response_id = ex["response_id"]
            reward = ex["reward"]

            states = prompt_id + response_id[:-1]
            actions = (len(prompt_id) - 1) * [0] + response_id
            rewards = (len(prompt_id) + len(response_id) - 2) * [0] + [reward]
            position_ids = list(range(len(prompt_id) + len(response_id) - 1))
            action_mask = (len(prompt_id) - 1) * [0] + len(response_id) * [1]
            eos_mask = (len(prompt_id) + len(response_id) - 2) * [0] + [1]

            data_list.append({
                "states": torch.LongTensor([states]),
                "actions": torch.LongTensor([actions]),
                "rewards": torch.FloatTensor([rewards]),
                "position_ids": torch.LongTensor([position_ids]),
                "action_mask": torch.LongTensor([action_mask]),
                "eos_mask": torch.LongTensor([eos_mask])
            })

        return all_gather(data_list, self.device_mesh)
    
    def add_kl_term_to_reward(self, data_list):

        data_list = dispatch(data_list, self.device_mesh)

        for ex in data_list:

            kl_term = compute_kl_term(
                ex["old_logps"], ex["ref_logps"],
                self.config.actor.kl.estimator,
                ex["eos_mask"] if self.config.actor.kl.level == "sequence" else None
            )
            ex["rewards"] -= self.config.actor.kl.coef * kl_term

        return all_gather(data_list, self.device_mesh)
    
    def compute_gae(self, data_list):

        data_list = dispatch(data_list, self.device_mesh)

        for ex in data_list:

            # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
            # if s_{t+1} is a terminal state, V(s_{t+1}) = 0
            next_values = torch.cat(
                (ex["values"][: 1:], torch.FloatTensor([[0]])),
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

        return all_gather(data_list, self.device_mesh)
    
    def compute_reinforce_adv(self, data_list):

        data_list = dispatch(data_list, self.device_mesh)

        rewards = torch.FloatTensor([ex["rewards"].sum() for ex in data_list]).view(-1, self.config.actor.rollout.rollout_per_prompt)
        baselines = rewards.mean(-1, keepdim=True)
        advantages = rewards - baselines
        if self.config.adv.group_norm:
            stds = rewards.std(-1, keepdim=True)
            advantages /= (stds + torch.finfo(stds.dtype).eps)
        advantages = advantages.flatten()

        for ex, advantage in zip(data_list, advantages):
            ex["advantages"] = advantage * ex["action_mask"]
        
        return all_gather(data_list, self.device_mesh)

    def train(self):

        step = 0
        for data_dict in self.test_dataloader:
            self.actor.rollout(data_dict, False, step)
    
        for epoch in range(self.config.trainer.n_epochs):
            self.sampler.set_epoch(epoch)
            for data_list in self.train_dataloader:

                data_list = self.actor.rollout(data_list, True, step)
                data_list = self.prepare_data_list_for_update(data_list)

                data_list = self.actor.compute_logps(data_list, step)
                if self.config.kl.coef > 0:
                    data_list = self.ref_actor.compute_logps(data_list, step)

                if self.config.actor.kl.type == "reward":
                    data_list = self.add_kl_term_to_reward(data_list)
                if self.config.adv.estimator == "gae":
                    data_list = self.critic.compute_values(data_list, step)
                    data_list = self.compute_gae(data_list)
                    self.critic.update(data_list)
                else:
                    data_list = self.compute_reinforce_adv(data_list)

                self.actor.update(data_list)

                step += 1
                if step % self.config.trainer.test_freq == 0:
                    for data_dict in self.test_dataloader:
                        self.actor.rollout(data_dict, False, step)
                    
                if step % self.config.trainer.save_freq == 0:
                    self.actor.save(step)


@hydra.main(config_path="", config_name="config", version_base=None)
def main(config):

    _, _, world_size = initialize_global_process_group()
    
    trainer = Trainer(config, world_size)
    trainer.train()

if __name__ == "__main__":
    main()