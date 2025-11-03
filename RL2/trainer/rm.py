import hydra
from omegaconf import DictConfig
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import RMDataset, get_dataloader
from RL2.workers import initialize_critic
from RL2.utils.communication import initialize_global_process_group


class RMTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.critic = initialize_critic(config.critic)
        dataset = RMDataset(
            config.data, self.critic.tokenizer
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.critic.prepare_scheduler(
            self.config.trainer.n_epochs * len(self.train_dataloader)
        )

    def train(self):

        step = self.load_ckpt((self.critic,))
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
                self.critic.rm_update(tensor_dict, step)
                self.save_ckpt((self.critic,), step)
        self.save_model((self.critic,))


@hydra.main(config_path="config", config_name="rm", version_base=None)
def main(config: DictConfig):

    initialize_global_process_group()

    trainer = RMTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()