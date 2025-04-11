from typing import Dict, List
import torch
from transformers import AutoModelForTokenClassification
from tqdm import tqdm
from worker import Worker
from utils import (
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
    def compute_values(self, data_list, step: int):
        load_fsdp_model_to_gpu(self.model)
        minibatches = self.pack_data_to_minibatches(data_list, False)

        self.model.eval()
        for minibatch in (
            tqdm(minibatches, desc=f"Step {step + 1}, compute values")
            if self.device_mesh.get_rank() == 0 else minibatches
        ):
            minibatch["values"] = self.forward(minibatch)
        
        # no need to offload model because it will be updated soon
        return self.resume_minibatches_to_data_list(minibatches)

    def update(self, data_list, step):
        # critic has been loaded on GPU in `compute_values`
        load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())
        minibatches = self.pack_data_list_to_minibatches(data_list, True)
        batches = self.group_minibatches_into_batches(minibatches)

        self.model.train()
        losses, grad_norms = [], []
        for batch in batches:

            total_actions = sum([minibatch["action_mask"].sum() for minibatch in batch])
            for minibatch in batch:

                values = self.forward(minibatch)
                cliped_values = torch.clamp(
                    values,
                    minibatch["values"] - self.config.value_clip,
                    minibatch["values"] + self.config.value_clip
                )
                loss = (torch.max(
                    (values - minibatch["returns"]).pow(2),
                    (cliped_values - minibatch["returns"]).pow(2)
                )).sum() / total_actions
                loss.backward()
                losses.append(loss.item())

            grad_norm = self.model.clip_grad_norm_(self.config.max_grad_norm)
            grad_norms.append(grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.log({
            "critic/loss": losses,
            "critic/grad_norm": grad_norms
        }, step)

        offload_fsdp_model_to_cpu(self.model)
        offload_fsdp_optimizer(self.optimizer)