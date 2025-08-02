import hydra
from collections import defaultdict
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import RMDataset, get_dataloader
from RL2.workers import Critic
from RL2.utils.functions import sequence_all_reduce
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log
)


class RMTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.critic = Critic(config.critic)
        dataset = RMDataset(
            config.data.path, self.critic.tokenizer, config.data.max_length
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.critic.scheduler = self.prepare_scheduler(self.critic)

    @time_logger("update_critic")
    def update_critic(self, data_list, step):

        minibatches = self.critic.scatter_and_pack_data_list(data_list, pair=True)
        metrics = defaultdict(list)
        for minibatch in progress_bar(
            minibatches, desc="Update critic"
        ):
            rewards = self.critic.forward(minibatch)
            chosen_rewards, rejected_rewards = sequence_all_reduce(
                rewards,
                minibatch["cu_seqlens"],
                self.critic.device_mesh["sp"]
            ).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / self.config.data.batch_size
            self.critic.backward(loss)
            metrics["loss"].append(loss.item())
            metrics["accuray"].extend((reward_margins > 0).tolist())

        grad_norm = self.critic.optimizer_step()
        metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, self.critic.device_mesh["dp"], step)

    def train(self):

        step = self.load_ckpt((self.critic,))
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
                self.update_critic(data_list, step)
                self.save_ckpt((self.critic,), step)
        self.save_model(self.critic, rm=True)


@hydra.main(config_path="config", config_name="rm", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = RMTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()