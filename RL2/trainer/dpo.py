import hydra
from collections import defaultdict
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import DPODataset, get_dataloader
from RL2.workers import Actor
from RL2.utils.functions import sequence_all_reduce
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log
)


class DPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, True)
        self.ref_actor = Actor(config.ref_actor, False)
        dataset = DPODataset(
            config.data.path, self.actor.tokenizer, config.data.max_length
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.actor.scheduler = self.prepare_scheduler(self.actor)

    @time_logger("update_actor")
    def update_actor(self, data_list, step):

        minibatches = self.actor.scatter_and_pack_data_list(data_list, pair=True)
        metrics = defaultdict(list)
        for minibatch in progress_bar(
            minibatches, desc="Update actor"
        ):
            logps = self.actor.forward(minibatch)
            chosen_rewards, rejected_rewards = sequence_all_reduce(
                self.config.actor.beta * (logps - minibatch["ref_logps"]),
                minibatch["cu_seqlens"],
                self.actor.device_mesh["sp"]
            ).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / self.config.data.batch_size
            self.actor.backward(loss)

            metrics["rewards/chosen"].extend(chosen_rewards.tolist())
            metrics["rewards/rejected"].extend(rejected_rewards.tolist())
            metrics["rewards/margin"].extend(reward_margins.tolist())
            metrics["loss"].append(loss.item())
            metrics["accuray"].extend((reward_margins > 0).tolist())

        grad_norm = self.actor.optimizer_step()
        metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, self.actor.device_mesh["dp"], step)

    def train(self):

        step = self.load_ckpt((self.actor,))
        for epoch in range(
            step // len(self.train_dataloader), self.config.trainer.n_epochs
        ):
            for data_list in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):
                step += 1
                data_list = self.ref_actor.compute_logps(data_list, step)
                self.update_actor(data_list, step)
                self.save_ckpt((self.actor,), step)
        self.save_model(self.actor)


@hydra.main(config_path="config", config_name="dpo", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = DPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()