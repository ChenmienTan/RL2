from collections import defaultdict
import torch
from transformers import AutoModelForTokenClassification
from RL2.workers import Worker
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.ring_attn import ring_attn_manager
from RL2.utils.functions import aggregate_values
from RL2.utils.offloading import model_offloading_manager
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_reduce,
    rank0_log
)


class Critic(Worker):

    def __init__(self, config):
        super().__init__(config, True)

        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=1,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        self.prepare_model_optimizer()

    @ring_attn_manager
    def forward(self, minibatch) -> torch.Tensor:

        return self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.squeeze(-1) * minibatch["action_mask"]

    @time_logger("compute_values")
    @model_offloading_manager
    @torch.no_grad()
    @data_manager(gather=True)
    def compute_values(self, minibatches, step):

        self.model.eval()
        for minibatch in progress_bar(minibatches, desc="Compute values"):
            minibatch["values"] = self.forward(minibatch)
        
        return minibatches

    @time_logger("update_critic")
    @model_offloading_manager
    @data_manager(pack_minibatches=True)
    def update(self, batches, step: int):

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update critic"
        )
        metrics = defaultdict(list)
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "sos_mask"),
                self.device_mesh["dp"]
            )
            metric = defaultdict(list)
            for minibatch in batch:

                values = self.forward(minibatch)
                clipped_values = torch.clamp(
                    values,
                    minibatch["values"] - self.config.clip,
                    minibatch["values"] + self.config.clip
                )
                mse = (values - minibatch["returns"]).pow(2)
                clipped_mse = (clipped_values - minibatch["returns"]).pow(2)
                losses = torch.max(mse, clipped_mse)
                clip_ratios = mse < clipped_mse

                loss, clip_ratio = aggregate_values(
                    (losses, clip_ratios),
                    minibatch,
                    self.config.agg_mode,
                    total_actions,
                    total_sequences
                )

                self.backward(loss)

                tbar.update()
                metric["critic/loss"].append(loss.item())
                metric["critic/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()
            
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"])
                )
            metrics["critic/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)