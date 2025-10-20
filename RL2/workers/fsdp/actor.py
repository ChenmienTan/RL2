from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from RL2.workers.fsdp import FSDPWorker
from RL2.utils.sequences import count_total, slide_along_cp, gather_along_cp
from RL2.utils.fsdp.context_parallelism import update_ring_attn_params
from RL2.utils.functions import (
    compute_logps_and_entropy, aggregate_values
)
from RL2.utils.algorithms import compute_approx_kl
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log
)


class FSDPActor(FSDPWorker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)

        if config.use_liger_kernel:
            assert config.tp_size == 1, \
                "Liger kernel is not compatible with tensor parallelism."
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model_cls = AutoLigerKernelForCausalLM
        else:
            model_cls = AutoModelForCausalLM

        with self.init_weight_context():
            self.model = model_cls.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )

        self.prepare_model_optimizer()

    def forward(self, minibatch, prefix=None, return_entropy=False):

        minibatch, cu_seqlens = slide_along_cp(
            minibatch,
            self.device_mesh["cp"].get_group(),
            self.device_mesh["tp"].size()
        )
        update_ring_attn_params(
            self.device_mesh["cp"].get_group(),
            cu_seqlens
        )
        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.to(torch.float32)
        # bfloat16 is unstable for the subsequent `logsumexp` operation.
        # See https://github.com/OpenRLHF/OpenRLHF/pull/634.
        compute_logps_and_entropy(
            logits / getattr(self.config, "temperature", 1.0),
            minibatch,
            self.device_mesh["tp"].get_group(),
            prefix,
            return_entropy
        )
        return gather_along_cp(
            minibatch,
            self.device_mesh["cp"].get_group(),
            cu_seqlens
        )

    @time_logger("compute_logps")
    @torch.no_grad()
    def compute_logps(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)
        self.load_model_to_device(torch.cuda.current_device())

        prefix = "old" if self.train else "ref"
        self.model.eval()
        processed_minibatches = []
        for minibatch in progress_bar(
            minibatches, desc=f"Compute {prefix} logps"
        ):
            processed_minibatch = self.forward(minibatch, prefix)
            processed_minibatches.append(processed_minibatch)

        if not self.train:
            self.load_model_to_device("cpu")
        return self.gather_data(processed_minibatches)

    @time_logger("update_actor")
    def sft_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)

        total_actions, total_sequences = count_total(
            minibatches,
            ("action_mask", "eos_mask"),
            self.device_mesh["dp"].get_group()
        )
        metrics = defaultdict(list)
        for minibatch in progress_bar(
            minibatches, desc="Update actor"
        ):
            minibatch = self.forward(minibatch)
            loss = aggregate_values(
                - minibatch["logps"],
                minibatch["action_mask"],
                self.config.avg_level,
                total_actions,
                total_sequences
            )
            self.scale_loss(loss).backward()
            metrics["loss"].append(loss.item())

        grad_norm = self.optimizer_step()
        metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def dpo_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict, pair=True)

        total_pairs = count_total(
            minibatches, "eos_mask", self.device_mesh["dp"].get_group()
        ) // 2
        metrics = defaultdict(list)
        for minibatch in progress_bar(
            minibatches, desc="Update actor"
        ):
            minibatch = self.forward(minibatch)
            chosen_rewards, rejected_rewards = self.config.beta * (
                minibatch["logps"] - minibatch["ref_logps"]
            ).sum(-1).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / total_pairs
            self.scale_loss(loss).backward()
            metrics["rewards/chosen"].extend(chosen_rewards.tolist())
            metrics["rewards/rejected"].extend(rejected_rewards.tolist())
            metrics["rewards/margin"].extend(reward_margins.tolist())
            metrics["loss"].append(loss.item())
            metrics["accuracy"].extend((reward_margins > 0).tolist())

        grad_norm = self.optimizer_step()
        metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())
    
    @time_logger("update_actor")
    def ppo_update(self, tensor_dict, step: int):
        if step < self.config.freeze_steps:
            return
        batches = self.scatter_data(tensor_dict, pack_minibatches=True)
        self.load_model_to_device(torch.cuda.current_device())

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
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

                minibatch = self.forward(
                    minibatch, return_entropy=True
                )
                ratio = torch.exp(
                    minibatch["logps"] - minibatch.get(
                        "old_logps", minibatch["logps"].detach()
                    )
                )
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip, 1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                losses = - torch.min(objective, clipped_objective)
                clip_ratios = objective > clipped_objective

                if self.config.tis_coef > 0:
                    # https://fengyao.notion.site/off-policy-rl
                    tis = torch.exp(
                        minibatch["logps"].detach() - minibatch["llm_logps"]
                    ).clamp(max=self.config.tis_coef)
                    losses *= tis
                    
                loss, clip_ratio, entropy = aggregate_values(
                    (losses, clip_ratios, minibatch["entropy"]),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )
                loss = loss - self.config.entropy.coef * entropy
                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl_loss = compute_approx_kl(
                        minibatch["logps"],
                        minibatch["ref_logps"],
                        self.config.kl.loss_estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl_loss

                self.scale_loss(loss).backward()

                tbar.update()
                metric["actor/entropy"].append(entropy.item())
                metric["actor/loss"].append(loss.item())
                metric["actor/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()

            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"].get_group())
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
        self.load_model_to_device("cpu")

    @time_logger("update_rollout")
    def update_rollout(self, rollout, step):

        if self.config.use_lora:
            lora_dir = f"lora_{self.config.lora.r}"
            self.save_lora(lora_dir)
            rollout.update_lora(lora_dir)
        else:
            state_dict = self.get_model_state_dict(cpu_offload=False)
            rollout.update(state_dict.items())