import hydra
from omegaconf import DictConfig
import asyncio
import torch.distributed as dist
from tqdm import trange
from RL2.trainer import Trainer
from RL2.workers import (
    initialize_actor,
    initialize_critic,
    initialize_rollout
)
from RL2.utils.communication import (
    initialize_global_process_group,
    open_session,
    close_session
)
from RL2.utils.algorithms import compute_advantages


class PPOTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        if not config.trainer.eval_only:

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
            
    async def train(self):

        await open_session()

        self.rollout = await initialize_rollout(self.config.rollout)

        if self.config.trainer.eval_only:
            await self.rollout(False, 0)
            await close_session()
            return

        initial = self.load_ckpt(
            (self.actor, self.critic)
            if self.config.adv.estimator == "gae"
            else (self.actor,)
        )
        for step in trange(
            1,
            self.config.trainer.total_steps + 1,
            disable=(dist.get_rank() != 0),
            initial=initial
        ):

            tensor_dict, cu_seqs = await self.rollout(True, step)

            if self.config.actor.kl.coef > 0:
                tensor_dict = self.ref_actor.compute_logps(tensor_dict, step)
            if self.config.adv.estimator == "gae":
                tensor_dict = self.critic.compute_values(tensor_dict, step)
            if self.config.actor.kl.coef > 0 or self.config.actor.update_per_rollout > 1:
                tensor_dict = self.actor.compute_logps(tensor_dict, step)

            if dist.get_rank() == 0:
                compute_advantages(self.config, tensor_dict, cu_seqs, step)

            self.actor.ppo_update(tensor_dict, step)
            if self.config.adv.estimator == "gae":
                self.critic.ppo_update(tensor_dict, step)
            self.save_ckpt(
                (self.actor, self.critic)
                if self.config.adv.estimator == "gae"
                else (self.actor,),
                step
            )

            await self.actor.update_rollout(self.rollout, step)
            if self.config.trainer.test_freq is not None and step % self.config.trainer.test_freq == 0:
                await self.rollout(False, step)

        self.save_model(
            (self.actor, self.critic)
            if self.config.adv.estimator == "gae"
            else (self.actor,)
        )

        await close_session()

    @property
    def train_dataloader(self):
        return self.rollout.train_dataloader


@hydra.main(config_path="config", config_name="ppo", version_base=None)
def main(config: DictConfig):

    initialize_global_process_group(True)
    
    trainer = PPOTrainer(config)
    asyncio.run(trainer.train())

    dist.destroy_process_group()

if __name__ == "__main__":
    main()