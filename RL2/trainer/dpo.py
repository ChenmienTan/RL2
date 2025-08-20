import hydra
from collections import defaultdict
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import DPODataset, get_dataloader
from RL2.workers import Actor
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.functions import aggregate_values
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.checkpointing import load_ckpt, save_ckpt, save_model
from RL2.utils.logging import progress_bar, time_logger, gather_and_log

@time_logger("update_actor")
@data_manager(pair=True)
def update(worker, minibatches, step):

    total_sequences = count_total(
        minibatches, "eos_mask", worker.device_mesh["dp"]
    ) // 2
    metrics = defaultdict(list)
    for minibatch in progress_bar(
        minibatches, desc="Update actor"
    ):
        logps = worker.forward(minibatch)
        chosen_rewards, rejected_rewards = aggregate_values(
            worker.config.beta * (logps - minibatch["ref_logps"]),
            minibatch,
            "seq_token_sum"
        ).view(-1, 2).T
        reward_margins = chosen_rewards - rejected_rewards
        loss = - F.logsigmoid(reward_margins).sum() / total_sequences
        worker.backward(loss)

        metrics["rewards/chosen"].extend(chosen_rewards.tolist())
        metrics["rewards/rejected"].extend(rejected_rewards.tolist())
        metrics["rewards/margin"].extend(reward_margins.tolist())
        metrics["loss"].append(loss.item())
        metrics["accuray"].extend((reward_margins > 0).tolist())

    grad_norm = worker.optimizer_step()
    metrics["grad_norm"].append(grad_norm)
    gather_and_log(metrics, worker.device_mesh["dp"], step)


class DPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, True)
        self.ref_actor = Actor(config.ref_actor, False)
        dataset = DPODataset(
            config.data, self.actor.tokenizer
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.actor.scheduler = self.prepare_scheduler(self.actor)

    def train(self):

        step = load_ckpt(self, (self.actor,))
        for epoch in range(
            step // len(self.train_dataloader), self.config.trainer.n_epochs
        ):
            for tensor_dicts in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):
                step += 1
                tensor_dicts = self.ref_actor.compute_logps(tensor_dicts, step)
                update(self.actor, tensor_dicts, step)
                save_ckpt(self, (self.actor,), step)
        save_model(self, self.actor)


@hydra.main(config_path="config", config_name="dpo", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = DPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()