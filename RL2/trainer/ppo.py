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
from RL2.workers.rollout import shutdown_processes_when_exit
from RL2.utils.communication import (
    initialize_global_process_group,
    with_session
)
from RL2.utils.algorithms import compute_advantages


class PPOTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        # Check if independent policy mode
        self.is_independent = (
            config.multi_agent.enabled and
            not config.multi_agent.shared_policy
        ) if hasattr(config, 'multi_agent') else False

        if not config.trainer.eval_only:

            self.actor = initialize_actor(config.actor, True)

            # For independent policy, actor is a dict
            if not self.is_independent:
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

        self.rollout = initialize_rollout(self.config.rollout)

    def _get_or_create_actor(self, agent_id=None):
        """Get actor for agent_id (creates if needed for independent policy)."""
        if not self.is_independent:
            return self.actor

        # Independent policy: get or create actor for this agent
        if agent_id not in self.actor:
            from hydra.core.hydra_config import HydraConfig
            hydra_config = HydraConfig.get()
            backend = hydra_config.runtime.choices.get("actor")

            if backend == "fsdp":
                from RL2.workers.fsdp.actor import FSDPActor
                self.actor[agent_id] = FSDPActor(self.config.actor, True)
            elif backend == "megatron":
                from RL2.workers.megatron.actor import MegatronActor
                self.actor[agent_id] = MegatronActor(self.config.actor, True)

            self.actor[agent_id].prepare_scheduler(self.config.trainer.total_steps)
            print(f"[PPOTrainer] Created independent actor for {agent_id}")

        return self.actor[agent_id]

    def _apply_to_actors(self, func_name, *args, **kwargs):
        """Apply a function to all actors (shared or independent)."""
        if not self.is_independent:
            return getattr(self.actor, func_name)(*args, **kwargs)

        # Independent policy: apply to each actor
        results = {}
        for agent_id, actor in self.actor.items():
            results[agent_id] = getattr(actor, func_name)(*args, **kwargs)
        return results

    @shutdown_processes_when_exit
    @with_session
    async def train(self):

        if self.config.trainer.eval_only:
            await self.rollout(False, 0)
            return

        # Load checkpoint
        actors_to_load = list(self.actor.values()) if self.is_independent else [self.actor]
        initial = self.load_ckpt(
            tuple(actors_to_load) + ((self.critic,) if self.config.adv.estimator == "gae" else ())
        )
        for step in trange(
            1,
            self.config.trainer.total_steps + 1,
            disable=(dist.get_rank() != 0),
            initial=initial
        ):

            tensor_dict, cu_seqs = await self.rollout(True, step)

            if self.config.actor.kl.coef > 0:
                tensor_dict = self._apply_to_actors('compute_logps', tensor_dict, step)
            if self.config.adv.estimator == "gae":
                tensor_dict = self.critic.compute_values(tensor_dict, step)
            if self.config.actor.kl.coef > 0 or self.config.actor.update_per_rollout > 1:
                tensor_dict = self._apply_to_actors('compute_logps', tensor_dict, step)

            if dist.get_rank() == 0:
                compute_advantages(self.config, tensor_dict, cu_seqs, step)

            self._apply_to_actors('ppo_update', tensor_dict, step)
            if self.config.adv.estimator == "gae":
                self.critic.ppo_update(tensor_dict, step)

            # Save checkpoint
            actors_to_save = list(self.actor.values()) if self.is_independent else [self.actor]
            self.save_ckpt(
                tuple(actors_to_save) + ((self.critic,) if self.config.adv.estimator == "gae" else ()),
                step
            )

            self._apply_to_actors('update_rollout', self.rollout, step)
            if self.config.trainer.test_freq is not None and step % self.config.trainer.test_freq == 0:
                await self.rollout(False, step)

        # Save final model
        actors_to_save = list(self.actor.values()) if self.is_independent else [self.actor]
        self.save_model(
            tuple(actors_to_save) + ((self.critic,) if self.config.adv.estimator == "gae" else ())
        )

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