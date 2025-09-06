import hydra
from collections import defaultdict
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import DPODataset, get_dataloader
from RL2.workers import Actor
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.checkpointing import load_ckpt, save_ckpt, save_model
from RL2.utils.logging import progress_bar, time_logger, gather_and_log


@time_logger("update_actor")
@data_manager(pair=True)
def update(worker, minibatches, step):

    total_pairs = count_total(minibatches, "eos_mask", worker.device_mesh["dp"]) // 2
    metrics = defaultdict(list)
    for minibatch in progress_bar(minibatches, desc="Update actor"):
        logps = worker.forward(minibatch)
        response_lens = minibatch["action_mask"].sum(-1)
        chosen_rewards, rejected_rewards = worker.config.beta * (
            ((logps).sum(-1) / response_lens.clamp(min=1)).view(-1, 2).T
        )
        reward_margins = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(reward_margins - worker.config.gamma).sum() / total_pairs
        worker.backward(loss)

        metrics["rewards/chosen"].extend(chosen_rewards.tolist())
        metrics["rewards/rejected"].extend(rejected_rewards.tolist())
        metrics["rewards/margin"].extend(reward_margins.tolist())
        metrics["loss"].append(loss.item())
        metrics["accuray"].extend((reward_margins > 0).tolist())

    grad_norm = worker.optimizer_step()
    metrics["grad_norm"].append(grad_norm)
    gather_and_log(metrics, worker.device_mesh["dp"], step)


class SimPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, True)
        dataset = DPODataset(config.data, self.actor.tokenizer)
        self.train_dataloader = get_dataloader(dataset, config.data.batch_size)
        self.actor.scheduler = self.prepare_scheduler(self.actor)

    def train(self):

        step = load_ckpt(self, (self.actor,))
        for epoch in range(
            step // len(self.train_dataloader), self.config.trainer.n_epochs
        ):
            for tensor_dict in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader),
            ):
                step += 1
                update(self.actor, tensor_dict, step)
                save_ckpt(self, (self.actor,), step)
        save_model(self, self.actor)


@hydra.main(config_path="config", config_name="simpo", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = SimPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
