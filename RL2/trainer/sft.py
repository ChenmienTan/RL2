import hydra
from collections import defaultdict
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import SFTDataset
from RL2.workers import Actor
from RL2.utils.comm import initialize_global_process_group
from RL2.algs import sequence_all_reduce
from RL2.utils.logging import gather_and_log
from RL2.utils.timing import time_logger


class SFTTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, True)
        dataset = SFTDataset(
            config.data.path, self.actor.tokenizer, config.data.max_length
        )
        self.dataloader = self.prepare_dataloader(
            dataset, config.data.batch_size, True
        )
        self.scheduler = self.prepare_scheduler(self.actor)

    @time_logger("update_actor")
    def update_actor(self, data_list, step):

        minibatches = self.actor.scatter_and_pack_data_list(data_list)
        metrics = defaultdict(list)
        for minibatch in self.actor.tqdm(
            minibatches, desc="Update actor"
        ):
            logps = self.actor.forward(minibatch)
            logps = sequence_all_reduce(
                logps, minibatch["cu_seqlens"], self.actor.device_mesh["sp"]
            ) / sequence_all_reduce(
                minibatch["action_mask"],
                minibatch["cu_seqlens"],
                self.actor.device_mesh["sp"]
            )
            loss = - logps.sum() / self.config.data.batch_size
            self.actor.backward(loss)
            metrics["loss"].append(loss.item())

        grad_norm = self.actor.optimizer_step()
        self.scheduler.step()
        metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, self.actor.device_mesh["dp"], step)

    def train(self):

        step = 0
        for epoch in range(self.config.trainer.n_epochs):
            for data_list in tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0)
            ):
                self.update_actor(data_list, step)
                step += 1

                if self.actor.config.save_freq is not None and step % self.actor.config.save_freq == 0:
                    self.actor.save(step)

        self.actor.save()


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()