import hydra
from omegaconf import DictConfig
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import SFTDataset, get_dataloader
from RL2.workers import initialize_actor
from RL2.utils.communication import initialize_global_process_group


class SFTTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)
        dataset = SFTDataset(
            config.data.train, self.actor.tokenizer
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.train.batch_size
        )
        if config.data.test.path:
            dataset = SFTDataset(
                config.data.test, self.actor.tokenizer
            )
            self.test_dataloader = get_dataloader(
                dataset, len(dataset)
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
                self.actor.sft_step(tensor_dict, True, step)
                self.save_ckpt((self.actor,), step)

                if self.config.trainer.test_freq is not None and step % self.config.trainer.test_freq == 0:
                    for tensor_dict in self.test_dataloader:
                        self.actor.sft_step(tensor_dict, False, step)

        self.save_model((self.actor,))


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config: DictConfig):

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()