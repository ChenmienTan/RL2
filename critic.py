from typing import Dict, List
import torch
from transformers import AutoModelForTokenClassification
from tqdm import tqdm
from worker import Worker
from utils import (
    all_gather_list,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer
)


class Critic(Worker):

    def __init__(self, config, device_mesh):
        super().__init__(config, device_mesh, True)

        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=1,
            attn_implementation="flash_attention_2"
        )

        self.prepare_model_optimizer()

    def forward(self, minibatch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.squeeze(-1) * minibatch["action_mask"]

    @torch.no_grad()
    def compute_values(self, data_list: List[Dict[str, torch.Tensor]], step: int) -> List[Dict[str, torch.Tensor]]:
        self.load_model_to_gpu()
        minibatches = self.pack_data_list_to_minibatches(data_list, False)

        self.model.eval()
        for minibatch in (
            tqdm(minibatches, desc=f"Step {step + 1}, compute values")
            if self.device_mesh.get_rank() == 0 else minibatches
        ):
            minibatch["values"] = self.forward(minibatch)
        
        # No need to offload model because it will be updated soon. See `Trainer.train`.
        return self.resume_minibatches_to_data_list(minibatches)

    def update(self, data_list: List[Dict[str, torch.Tensor]], step: int):
        # Model has been loaded in `compute_values`.
        self.load_optimizer_to_gpu()
        minibatches = self.pack_data_list_to_minibatches(data_list, True)
        batches = self.group_minibatches_into_batches(minibatches)

        self.model.train()
        losses, clip_ratios, grad_norms = [], [], []
        tbar = tqdm(total=len(minibatches), desc=f"Step {step + 1}, update critic")
        for batch in batches:

            total_actions = sum(all_gather_list(
                [sum([
                    minibatch["action_mask"].sum()
                    for minibatch in batch
                ])],
            self.device_mesh))

            for minibatch in batch:

                values = self.forward(minibatch)
                cliped_values = torch.clamp(
                    values,
                    minibatch["values"] - self.config.value_clip,
                    minibatch["values"] + self.config.value_clip
                )
                mse = (values - minibatch["returns"]).pow(2)
                clipped_mse = (cliped_values - minibatch["returns"]).pow(2)
                loss = torch.max(mse, clipped_mse).sum() / total_actions
                clip_ratio = (mse < clipped_mse).sum() / total_actions

                loss.backward()
                losses.append(self.device_mesh.size() * len(batch) * loss.item())
                clip_ratios.append(self.device_mesh.size() * len(batch) * clip_ratio.item())
                tbar.update()

            grad_norm = self.model.clip_grad_norm_(self.config.max_grad_norm)
            grad_norms.append(grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.log({
            "critic/loss": losses,
            "critic/clip_ratio": clip_ratios,
            "critic/grad_norm": grad_norms
        }, step)

        self.offload_model_to_cpu()
        self.offload_optimizer_to_cpu()