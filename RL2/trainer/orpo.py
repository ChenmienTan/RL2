import hydra
from collections import defaultdict
import torch.distributed as dist
import torch.nn.functional as F
import torch
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import DPODataset, get_dataloader
from RL2.workers import Actor
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.checkpointing import load_ckpt, save_ckpt, save_model
from RL2.utils.logging import progress_bar, time_logger, gather_and_log


@time_logger("update_actor")
@data_manager()
def update(worker, minibatches, step):

    total_pairs = count_total(minibatches, "eos_mask", worker.device_mesh["dp"]) // 2
    metrics = defaultdict(list)
    for minibatch in progress_bar(minibatches, desc="Update actor"):
        logps = worker.forward(minibatch)
        response_lens = minibatch["action_mask"].sum(-1)
        chosen_ll, rejected_ll = (
            ((logps).sum(-1) / response_lens.clamp(min=1)).view(-1, 2).T
        )
        assert chosen_ll.size(0) == rejected_ll.size(0)
        chosen_logit = chosen_ll - torch.log1p(
            -chosen_ll.exp().clamp(max=1 - worker.config.eps)
        )
        rejected_logit = rejected_ll - torch.log1p(
            -rejected_ll.exp().clamp(max=1 - worker.config.eps)
        )
        sft_loss = -chosen_ll.mean()
        odds_loss = -F.logsigmoid(chosen_logit - rejected_logit).sum() / total_pairs
        loss = sft_loss + worker.config.lambda_orpo * odds_loss
        worker.backward(loss)
        metrics["sft_loss"].append(sft_loss.item())
        metrics["odds_loss"].append(odds_loss.item())
        metrics["loss"].append(loss.item())

    grad_norm = worker.optimizer_step()
    metrics["grad_norm"].append(grad_norm)
    gather_and_log(metrics, worker.device_mesh["dp"], step)


class ORPOTrainer(Trainer):

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


@hydra.main(config_path="config", config_name="orpo", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = ORPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
