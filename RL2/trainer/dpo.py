import hydra
from omegaconf import DictConfig
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import DPODataset, get_dataloader
from RL2.workers import initialize_actor
from RL2.utils.communication import initialize_global_process_group


class DPOTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)
        self.ref_actor = initialize_actor(config.ref_actor, False)
        dataset = DPODataset(
            config.data, self.actor.tokenizer
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.actor.prepare_scheduler(
            self.config.trainer.n_epochs * len(self.train_dataloader)
        )

    def train(self):

        step = self.load_ckpt((self.actor,))
        for epoch in range(
            step // len(self.train_dataloader),
            self.config.trainer.n_epochs
        ):
            for tensor_dict in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):
                step += 1
                tensor_dict = self.ref_actor.compute_logps(tensor_dict, step, True)
                self.actor.dpo_update(tensor_dict, step)
                self.save_ckpt((self.actor,), step)
        self.save_model((self.actor,))


@hydra.main(config_path="config", config_name="dpo", version_base=None)
def main(config: DictConfig):

    initialize_global_process_group()

    trainer = DPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()