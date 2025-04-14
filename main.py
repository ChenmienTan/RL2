from typing import Dict, List
import hydra
from omegaconf import OmegaConf
import os
from torch.utils.data import DistributedSampler, DataLoader
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import wandb
from data import RLDataset
from workers.actor import Actor
from workers.critic import Critic
from algs import compute_gae, compute_reinforce_adv
from utils.comm import initialize_global_process_group


class Trainer:

    def __init__(self, config):

        self.config = config
        world_size = int(os.environ["WORLD_SIZE"])
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(world_size,)
        )

        if config.actor.kl.coef > 0:
            self.ref_actor = Actor(config.actor, self.device_mesh, False)
        if config.adv.estimator == "gae":
            self.critic = Critic(config.critic, self.device_mesh)
        self.actor = Actor(config.actor, self.device_mesh, True)

        self.tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        self.sampler, self.train_dataloader = self.prepare_sampler_dataloader(
            config.data.train_data_path, True
        )
        _, self.test_dataloader = self.prepare_sampler_dataloader(
            config.data.test_data_path, False
        )

        if self.device_mesh.get_rank() == 0:
            wandb.init(
                project=self.config.trainer.project,
                name=self.config.trainer.experiment_name,
                config=OmegaConf.to_container(self.config)
            )

    def prepare_sampler_dataloader(self, data_path, train: bool):

        dataset = RLDataset(
            data_path,
            self.tokenizer,
            self.config.data.max_prompt_length
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.actor.sp_device_mesh["dp"].size(),
            rank=self.actor.sp_device_mesh["dp"].get_rank(),
            shuffle=train,
            drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            (self.config.data.batch_size if train else len(dataset)) // self.actor.sp_device_mesh["dp"].size(),
            # if test, pack all data in a single batch
            sampler=sampler,
            collate_fn=dataset.collate_fn
        )

        return sampler, dataloader
    
    def compute_advantages(self, data_list: List[Dict[str, torch.Tensor]]):

        if self.config.adv.estimator == "gae":
            compute_gae(
                data_list,
                self.config.adv.gamma,
                self.config.adv.lamda
            )
        elif self.config.adv.estimator == "reinforce":
            compute_reinforce_adv(
                data_list,
                self.config.actor.rollout.rollout_per_prompt,
                self.config.adv.norm_var
            )
        else:
            raise NotImplementedError

        return data_list
            
    def train(self):

        step = 0
        for data_dict in self.test_dataloader:
            self.actor.rollout(data_dict, False, step)
    
        for epoch in range(self.config.trainer.n_epochs):
            self.sampler.set_epoch(epoch)
            for data_list in self.train_dataloader:

                data_list = self.actor.rollout(data_list, True, step)

                data_list = self.actor.compute_logps(data_list, step)
                if self.config.actor.kl.coef > 0:
                    data_list = self.ref_actor.compute_logps(data_list, step)

                if self.config.adv.estimator == "gae":
                    data_list = self.critic.compute_values(data_list, step)

                if self.device_mesh.get_rank() == 0:
                    data_list = self.compute_advantages(data_list)

                if self.config.adv.estimator == "gae":
                    self.critic.update(data_list, step)

                self.actor.update(data_list, step)

                step += 1
                if step % self.config.trainer.test_freq == 0:
                    for data_dict in self.test_dataloader:
                        self.actor.rollout(data_dict, False, step)
                    
                if step % self.config.trainer.save_freq == 0:
                    path = f"{self.config.trainer.save_dir}/{self.config.experiment_name}/{step}"
                    self.tokenizer.save_pretrained(path)
                    self.actor.save(path)


@hydra.main(config_path="", config_name="config", version_base=None)
def main(config):

    initialize_global_process_group()
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()