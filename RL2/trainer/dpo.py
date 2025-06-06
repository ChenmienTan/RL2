import hydra
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import DPODataset
from RL2.workers import Actor
from RL2.algs import sequence_all_reduce
from RL2.utils.comm import initialize_global_process_group


class DPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        dataset = DPODataset(
            config.data.path, tokenizer, config.data.max_length
        )
        _, self.dataloader = self.prepare_sampler_dataloader(
            dataset, self.config.data.batch_size, True
        )

        self.actor = Actor(config.actor, self.device_mesh, True)
        self.ref_actor = Actor(config.ref_actor, self.device_mesh, False)

    def train(self):

        step = 0
        for _ in range(self.config.trainer.n_epochs):
            for data_list in (
                tqdm(self.dataloader) if self.device_mesh.get_rank() == 0 else self.dataloader
            ):
                data_list = self.ref_actor.compute_logps(data_list)
                minibatches = self.actor.scatter_and_pack_data_list(data_list, pair=True)

                metrics = defaultdict(list)
                for minibatch in minibatches:
                    logps = self.actor.forward(minibatch)
                    chosen_rewards, rejected_rewards = sequence_all_reduce(
                        minibatch,
                        self.config.trainer.beta * (logps - minibatch["ref_logps"]),
                        self.actor.sp_device_mesh["sp"]
                    ).view(-1, 2).T
                    reward_margins = chosen_rewards - rejected_rewards
                    loss = - F.logsigmoid(reward_margins).sum() / self.config.data.batch_size
                    (loss * self.actor.device_mesh.size()).backward()

                    metrics["rewards/chosen"].append(chosen_rewards.tolist())
                    metrics["rewards/rejected"].append(rejected_rewards.tolist())
                    metrics["rewards/margin"].append(reward_margins.tolist())
                    metrics["loss"].append(
                        self.actor.sp_device_mesh["dp"].size() * len(minibatches) * loss.item()
                    )
                    metrics["accuray"].append((reward_margins > 0).tolist())

                grad_norm = clip_grad_norm_(
                    self.actor.model.parameters(),
                    max_norm=self.actor.config.max_grad_norm
                )
                self.actor.optimizer_step()
                metrics["grad_norm"].append(grad_norm.full_tensor().item())
                self.actor.log(
                    metrics, step, self.actor.sp_device_mesh["dp"]
                )
                step += 1

                if self.actor.config.save_freq is not None and step % self.actor.config.save_freq == 0:
                    self.actor.save(step)

        self.actor.save(step)


@hydra.main(config_path="config", config_name="dpo", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = DPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()