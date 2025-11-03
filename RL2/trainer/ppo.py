from typing import Dict
import hydra
from omegaconf import DictConfig
import asyncio
import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import trange
import wandb
from RL2.trainer import Trainer
from RL2.datasets import RLDataset, get_dataloader
from RL2.workers import (
    initialize_actor,
    initialize_critic,
    initialize_rollout
)
from RL2.utils.communication import initialize_global_process_group
from RL2.utils.functions import aggregate_values
from RL2.utils.algorithms import (
    compute_approx_kl, compute_advantages
)
from RL2.utils.logging import time_logger


class PPOTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)
        self.actor.prepare_scheduler(
            self.config.trainer.total_steps
        )
        if config.actor.kl.coef > 0:
            self.ref_actor = initialize_actor(config.ref_actor, False)
        if config.adv.estimator == "gae":
            self.critic = initialize_critic(config.critic)
            self.critic.prepare_scheduler(
                self.config.trainer.total_steps
            )
        self.rollout = initialize_rollout(config.rollout)
        self.train_dataloader = self.get_dataloader(True)
        self.test_dataloader = self.get_dataloader(False)    

    def get_dataloader(self, train: bool) -> StatefulDataLoader:

        dataset = RLDataset(
            self.config.train_data
            if train else self.config.test_data,
            self.actor.tokenizer
        )

        return get_dataloader(dataset)
    
    @time_logger("compute_approx_kl")
    def compute_approx_kl(
        self, tensor_dict: Dict[str, torch.Tensor], step: int
    ):

        old_ref_approx_kl = compute_approx_kl(
            tensor_dict["old_logps"],
            tensor_dict["ref_logps"],
            self.config.actor.kl.reward_estimator
        )
        if self.config.actor.kl.type == "reward":
            tensor_dict["rewards"] -= self.config.actor.kl.coef * old_ref_approx_kl
        llm_ref_approx_kl = compute_approx_kl(
            tensor_dict["llm_logps"],
            tensor_dict["old_logps"],
            self.config.actor.kl.reward_estimator
        )
        old_ref_approx_kl, llm_ref_approx_kl = aggregate_values(
            (old_ref_approx_kl, llm_ref_approx_kl),
            tensor_dict["action_mask"],
            self.config.actor.avg_level,
            tensor_dict["action_mask"].sum(),
            tensor_dict["states"].shape[0]
        )
        wandb.log({
            "actor/old_ref_approx_kl": old_ref_approx_kl.item(),
            "actor/llm_ref_approx_kl": llm_ref_approx_kl.item()
        }, step=step)
            
    async def train(self):

        initial = self.load_ckpt(
            (self.actor, self.critic)
            if self.config.adv.estimator == "gae"
            else (self.actor,)
        )
        for step in trange(
            1,
            self.config.trainer.total_steps + 1,
            disable=(dist.get_rank() != 0),
            initial=initial + 1
        ):

            tensor_dict, cu_seqs = await self.rollout(self.train_dataloader, True, step)

            if self.config.actor.kl.coef > 0:
                tensor_dict = self.ref_actor.compute_logps(tensor_dict, step)
            if self.config.adv.estimator == "gae":
                tensor_dict = self.critic.compute_values(tensor_dict, step)
            if self.config.actor.kl.coef > 0 or self.config.actor.update_per_rollout > 1:
                tensor_dict = self.actor.compute_logps(tensor_dict, step)

            if dist.get_rank() == 0:
                if self.config.actor.kl.coef > 0:
                    self.compute_approx_kl(tensor_dict, step)
                compute_advantages(self.config.adv, tensor_dict, cu_seqs, step)

            self.actor.ppo_update(tensor_dict, step)
            if self.config.adv.estimator == "gae":
                self.critic.ppo_update(tensor_dict, step)
            self.save_ckpt(
                (self.actor, self.critic)
                if self.config.adv.estimator == "gae"
                else (self.actor,),
                step
            )

            self.actor.update_rollout(self.rollout, step)
            if self.config.trainer.test_freq is not None and step % self.config.trainer.test_freq == 0:
                await self.rollout(self.test_dataloader, False, step)

        self.save_model(
            (self.actor, self.critic)
            if self.config.adv.estimator == "gae"
            else (self.actor,)
        )


@hydra.main(config_path="config", config_name="ppo", version_base=None)
def main(config: DictConfig):

    initialize_global_process_group()
    
    trainer = PPOTrainer(config)
    asyncio.run(trainer.train())

    dist.destroy_process_group()

if __name__ == "__main__":
    main()