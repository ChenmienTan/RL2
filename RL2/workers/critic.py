from collections import defaultdict
import torch
from transformers import AutoModelForTokenClassification
from RL2.workers import Worker
from RL2.utils.models import prepare_lora_model
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.offloading import (
    offload_model_to_cpu, load_model_to_gpu
)
from RL2.utils.checkpointing import save_model_and_optimizer
from RL2.utils.logging import (
    progress_bar,
    gather_and_reduce,
    rank0_log
)
from RL2.utils.timing import time_logger


class Critic(Worker):

    def __init__(self, config):
        super().__init__(config, True)

        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=1,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        if hasattr(self.config, "lora") and self.config.lora.rank > 0:
            assert config.tp_size == 1, \
                "LoRA is not compatible with tensor parallelism."
            self.model = prepare_lora_model(
                self.model, "TOKEN_CLS", config.lora
            )

        self.prepare_model_optimizer()

    def forward(self, minibatch) -> torch.Tensor:
        update_params_of_ring_attn(
            minibatch["cu_seqlens"], self.device_mesh["sp"]
        )

        return self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.squeeze(-1) * minibatch["action_mask"]

    @time_logger("compute_values")
    @torch.no_grad()
    def compute_values(self, data_list, step):

        if getattr(self.config, "offload_model", False):
            load_model_to_gpu(self.model)
        minibatches = self.scatter_and_pack_data_list(data_list)

        self.model.eval()
        for minibatch in progress_bar(minibatches, desc="Compute values"):
            minibatch["values"] = self.forward(minibatch)
        
        if getattr(self.config, "offload_model", False):
            offload_model_to_cpu(self.model)
        return self.unpack_and_gather_data_list(minibatches)

    @time_logger("update_critic")
    def update(self, data_list, step: int):

        if getattr(self.config, "offload_model", False):
            load_model_to_gpu(self.model)
        batches = self.scatter_and_pack_data_list(data_list, True)

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update critic"
        )
        metrics = defaultdict(list)
        for batch in batches:

            total_actions = self.count_total_actions(batch)
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
                loss = torch.max(mse, clipped_mse).sum() / total_actions
                clip_ratio = (mse < clipped_mse).sum() / total_actions
                
                self.backward(loss)

                tbar.update()
                metric["critic/loss"].append(loss.item())
                metric["critic/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()
            
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh)
                )
            metrics["critic/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
        if self.config.save_freq is not None and (step + 1) % self.config.save_freq == 0:
            save_model_and_optimizer(self, step)

        if getattr(self.config, "offload_model", False):
            offload_model_to_cpu(self.model)