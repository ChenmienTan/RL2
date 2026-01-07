from typing import Dict, Optional
from omegaconf import DictConfig
from collections import defaultdict
import torch
from transformers import  AutoModelForTokenClassification
from RL2.workers.fsdp import FSDPWorker
from RL2.utils.sequences import count_total, slide_along_cp, gather_along_cp
from RL2.utils.fsdp.context_parallelism import update_ring_attn_params
from RL2.utils.functions import aggregate_values
from RL2.utils.algorithms import rm_loss, critic_ppo_loss
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log
)


class FSDPCritic(FSDPWorker):

    def __init__(self, config: DictConfig):
        super().__init__(config, True)

        with self._init_weight_context():
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.model_name,
                num_labels=1,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )

        self._prepare_model_optimizer()

    def _forward(
        self,
        minibatch: Dict[str, torch.Tensor],
        prefix: str = ""
    ) -> Dict[str, torch.Tensor]:

        minibatch, cu_seqlens = slide_along_cp(
            minibatch,
            self.device_mesh["cp"].get_group(),
            self.device_mesh["tp"].size()
        )
        update_ring_attn_params(
            self.device_mesh["cp"].get_group(),
            cu_seqlens
        )
        minibatch[f"{prefix}values"] = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.squeeze(-1) * minibatch["action_mask"]
        return gather_along_cp(
            minibatch,
            self.device_mesh["cp"].get_group(),
            cu_seqlens
        )

    @time_logger("compute_values")
    @torch.no_grad()
    def compute_values(
        self,
        tensor_dict: Optional[Dict[str, torch.Tensor]],
        step: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        minibatches = self._scatter_data(tensor_dict)
        self._load_model_to_device(torch.cuda.current_device())

        self.model.eval()
        processed_minibatches = []
        for minibatch in progress_bar(minibatches, desc="Compute values"):
            processed_minibatch = self._forward(minibatch, "old_")
            processed_minibatches.append(processed_minibatch)

        self._load_model_to_device("cpu")
        return self._gather_data(processed_minibatches)

    @time_logger("update_critic")
    def rm_step(
        self,
        tensor_dict: Optional[Dict[str, torch.Tensor]],
        train: bool,
        step: int
    ):
        minibatches = self._scatter_data(tensor_dict, pair=True)
        self.model.train(train)

        total_pairs = count_total(
            minibatches, "eos_mask", self.device_mesh["dp"].get_group()
        ) // 2
        metrics = defaultdict(list)
        for minibatch in progress_bar(
            minibatches, desc="RM step"
        ):
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            losses, metric = rm_loss(minibatch)
            loss = losses.sum() / total_pairs
            if train:
                self._scale_loss(loss).backward()
            prefix = "train" if train else "test"
            metric[f"{prefix}_loss"] = [loss.item()]
            for k, v in metric.items():
                metrics[k].extend(v)

        if train:
            grad_norm = self._optimizer_step()
            metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_critic")
    def ppo_update(
        self,
        tensor_dict: Optional[Dict[str, torch.Tensor]],
        step: int
    ):
        batches = self._scatter_data(tensor_dict, pack_minibatches=True)
        self._load_model_to_device(torch.cuda.current_device())

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update Critic"
        )
        metrics = defaultdict(list)
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                self.device_mesh["dp"].get_group()
            )
            metric = defaultdict(list)
            for minibatch in batch:

                minibatch = self._forward(minibatch)
                losses, clip_ratios = critic_ppo_loss(self.config, minibatch)

                loss, clip_ratio = aggregate_values(
                    (losses, clip_ratios),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )

                self._scale_loss(loss).backward()

                tbar.update()
                metric["critic/loss"].append(loss.item())
                metric["critic/clip_ratio"].append(clip_ratio.item())

            grad_norm = self._optimizer_step()
            
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"].get_group())
                )
            metrics["critic/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
        self._load_model_to_device("cpu")