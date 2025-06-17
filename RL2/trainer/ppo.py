import hydra
import torch.distributed as dist
from tqdm import tqdm
import wandb
from RL2.trainer import Trainer
from RL2.dataset import RLDataset
from RL2.workers import Actor, Critic
from RL2.algs import (
    compute_kl_term,
    compute_gae,
    compute_reinforce_adv,
    fill_zero_adv
)
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.timing import time_logger


class PPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        if config.actor.kl.coef > 0:
            self.ref_actor = Actor(
                config.actor, self.device_mesh, False
            )
        if config.adv.estimator == "gae":
            self.critic = Critic(config.critic, self.device_mesh)
        self.actor = Actor(config.actor, self.device_mesh, True)

        self.sampler, self.train_dataloader = self.prepare_sampler_dataloader(True)
        _, self.test_dataloader = self.prepare_sampler_dataloader(False)

    def prepare_sampler_dataloader(self, train: bool):

        dataset = RLDataset(
            self.config.data.train_data_path if train else self.config.data.test_data_path,
            self.config.data.responses_per_prompt if train else 1
        )

        return super().prepare_sampler_dataloader(
            dataset,
            (
                self.config.data.prompts_per_rollout
                if train else len(dataset)
            ) // self.actor.rollout_device_mesh["dp"].size(),
            train,
            self.actor.rollout_device_mesh["dp"]
        )
    
    @time_logger("compute_kl_term")
    def compute_kl_term(self, data_list, step):
        
        kl = 0
        for ex in data_list:
            kl_term = compute_kl_term(
                ex["old_logps"],
                ex["ref_logps"],
                self.config.actor.kl.reward_estimator
            )
            if self.config.actor.kl.type == "reward":
                ex["rewards"] -= self.config.actor.kl.coef * kl_term
            kl += kl_term.sum().item()
        kl /= sum([ex["action_mask"].sum().item() for ex in data_list])
        wandb.log({"actor/kl": kl}, step=step)
    
    @time_logger("compute_advantages")
    def compute_advantages(self, data_list, step):

        if self.config.adv.estimator == "gae":
            compute_gae(
                data_list,
                self.config.adv.gamma,
                self.config.adv.lamda
            )
        elif self.config.adv.estimator == "reinforce":
            compute_reinforce_adv(
                data_list,
                self.config.adv.norm_var
            )
        elif self.config.adv.estimator == "zeros":
            fill_zero_adv(data_list)
        else:
            raise NotImplementedError

        return data_list
            
    def train(self):

        step = 0
        for data_list in self.test_dataloader:
            self.actor.rollout(data_list, False, step)
    
        for epoch in range(self.config.trainer.n_epochs):
            self.sampler.set_epoch(epoch)
            for data_list in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(self.device_mesh.get_rank() != 0)
            ):

                data_list = self.actor.rollout(data_list, True, step)

                data_list = self.actor.compute_logps(data_list, step)
                if self.config.actor.kl.coef > 0:
                    data_list = self.ref_actor.compute_logps(data_list, step)
                    if self.device_mesh.get_rank() == 0:
                        self.compute_kl_term(data_list, step)

                if self.config.adv.estimator == "gae":
                    data_list = self.critic.compute_values(data_list, step)

                if self.device_mesh.get_rank() == 0:
                    data_list = self.compute_advantages(data_list, step)

                if self.config.adv.estimator == "gae":
                    self.critic.update(data_list, step)

                self.actor.update(data_list, step)

                step += 1
                if step % self.config.trainer.test_freq == 0:
                    for data_list in self.test_dataloader:
                        self.actor.rollout(data_list, False, step)


@hydra.main(config_path="config", config_name="ppo", version_base=None)
def main(config):

    initialize_global_process_group()
    
    trainer = PPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()