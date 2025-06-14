from omegaconf import OmegaConf
import os
import asyncio
import importlib
from collections import defaultdict
import torch
import torch.distributed as dist
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
from RL2.workers import Worker
from RL2.dataset import tokenize_messages
from RL2.algs import compute_kl_term, compute_baseline
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.comm import gather_and_concat_list, sum_across_processes
from RL2.utils.timing import time_logger


class Actor(Worker):

    def __init__(self, config, device_mesh, train: bool):
        super().__init__(config, device_mesh, train)
        
        self.model = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_name if train else config.ref_model_name,
            torch_dtype=torch.float32 if train else torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        self.prepare_model_optimizer()
        if hasattr(config, "rollout") and train:
            self.prepare_environment()
            self.prepare_inference_engine()

    def prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.rollout.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    def prepare_inference_engine(self):

        self.rollout_device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(
                self.device_mesh.size() // self.config.rollout.tp_size,
                self.config.rollout.tp_size
            )
        )

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()
        cuda_visible_devices = self.rollout_device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            os.environ["LOCAL_RANK"],
            self.rollout_device_mesh["tp"].get_group()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

        if self.rollout_device_mesh["tp"].get_local_rank() == 0:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            self.llm = Engine(
                model_path=self.config.model_name,
                dtype="bfloat16",
                tp_size=self.rollout_device_mesh["tp"].size(),
                mem_fraction_static=self.config.rollout.gpu_memory_utilization,
                enable_memory_saver=True
            )
        
            self.train_sampling_params = OmegaConf.to_container(
                self.config.rollout.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                self.config.rollout.test_sampling_params
            )

        dist.barrier()

    async def single_rollout(self, ex, train):

        uid, messages, answer = ex["uid"], ex["messages"], ex["answer"]
        metric = defaultdict(list)
        for turn in range(self.config.rollout.n_turns):

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tool=getattr(self.env, "TOOL", None),
                add_generation_prompt=True,
                tokenize=False
            )
            response = await self.llm.async_generate(
                prompt, sampling_params=self.train_sampling_params if train else self.test_sampling_params
            )
            messages.append(
                {"role": "assistant", "content": response["text"]}
            )

            meta_info = response["meta_info"]
            metric["response_length"].append(meta_info["completion_tokens"])
            metric["length_clip_ratio"].append(
                meta_info["finish_reason"]["type"] == "length"
            )

            # Do not invoke tools in the last turn.
            if turn + 1 == self.config.rollout.n_turns:
                break

            env_messages = self.env.interact(messages)
            # Terminate if no tool is invoked.
            if len(env_messages) == 0:
                break

            messages.extend(env_messages)

        reward = self.env.reward_fn(messages, answer)

        metric["n_turns"].append(turn + 1)
        metric["rewards"].append(reward)

        ex = tokenize_messages(self.tokenizer, messages)
        ex.update({
            "uid": uid,
            "rewards": torch.FloatTensor((ex["states"].shape[-1] - 1) * [0] + [reward])
        })  

        return ex, metric

    @time_logger("rollout")
    def rollout(self, data_list, train: bool, step: int):

        if self.rollout_device_mesh["tp"].get_local_rank() == 0:

            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                tqdm.gather(
                    *(self.single_rollout(ex, train) for ex in data_list),
                    desc="Rollout", position=1, leave=False,
                    disable=(self.device_mesh.get_rank() != 0)
                )
            )
            if train:
                # If test, llm will soon be called again. See `Trainer.train`.
                self.llm.release_memory_occupation()

            data_list, metrics = map(list, zip(*outputs))
            metrics = {
                k: sum([metric[k] for metric in metrics], [])
                for k in metrics[0].keys()
            }

            # Filter out over-length trajectories and groups with too low 
            # or too high average rewards, e.g., all trajectories within 
            # the group succeed or fail. We firstly perform length 
            # filtering because it may exclude some trajectories within a 
            # group and affect the average reward.
            is_length_filtered = [
                len(ex["states"]) > self.config.sp_size * self.config.max_length_per_device
                for ex in data_list
            ]
            data_list = [
                ex for ex, filtered in zip(data_list, is_length_filtered)
                if not filtered
            ]
            metrics["length_filtering_ratio"] = is_length_filtered

            _, _, uid2baseline = compute_baseline(data_list)
            valid_uids = [
                uid for uid, baseline in uid2baseline.items()
                if self.config.rollout.group_filtering.lower < baseline < self.config.rollout.group_filtering.upper
            ]
            is_group_filtered = [
                ex["uid"] not in valid_uids for ex in data_list
            ]
            data_list = [
                ex for ex, filtered in zip(data_list, is_group_filtered)
                if not filtered
            ]
            metrics["group_filtering_ratio"] = is_group_filtered
            
            suffix = "train" if train else "test"
            self.log(
                {f"{k}/{suffix}": v for k, v in metrics.items()},
                step=step,
                device_mesh=self.rollout_device_mesh["dp"]
            )

        dist.barrier()

        # After each worker operation, the data is aggregated to rank 0 and 
        # re-distributed before the next operation, which facilitates to do 
        # model-agnostic operations, e.g., computing advantages, globally 
        # and guarantees the load balancing across all model computations.
        if self.rollout_device_mesh["tp"].get_local_rank() == 0:
            return gather_and_concat_list(
                data_list,
                self.rollout_device_mesh["dp"]
            )

    def forward(self, minibatch, compute_entropy=False) -> torch.Tensor:
        update_params_of_ring_attn(
            minibatch["cu_seqlens"], self.sp_device_mesh["sp"]
        )

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits / (
            self.config.rollout.train_sampling_params.temperature
            if hasattr(self.config, "rollout") else 1.0
        )

        # Non-action logps are masked to zero so that they are excluded 
        # from computations. For example, in `algs.compute_kl_term`, 
        # because the `logps` and `ref_logps` of non-action tokens are both 
        # zero, their KL terms are also zero, regardless of the estimator.
        logps = torch.gather(
            logits.log_softmax(-1),
            dim=-1,
            index=minibatch["actions"].unsqueeze(-1)
        ).squeeze(-1) * minibatch["action_mask"]

        if compute_entropy:
            probs = logits.softmax(-1)
            entropy = logits.logsumexp(-1) - (probs * logits).sum(-1)
            return logps, entropy * minibatch["action_mask"]
        else:
            return logps

    @time_logger("compute_logps")
    @torch.no_grad()
    def compute_logps(self, data_list, step):
        self.load_model_to_gpu()
        minibatches = self.scatter_and_pack_data_list(data_list)

        prefix = "old" if self.train else "ref"

        self.model.eval()
        for minibatch in self.tqdm(
            minibatches, desc=f"Compute {prefix} logps"
        ):
            minibatch[f"{prefix}_logps"] = self.forward(minibatch)

        self.offload_model_to_cpu()
        return self.resume_and_gather_data_list(minibatches) 
    
    @time_logger("update_actor")
    def update(self, data_list, step: int):
        self.load_model_to_gpu()
        if step < self.config.freeze_steps:
            self.sync_llm_weight()
            return
        batches = self.scatter_and_pack_data_list(data_list, True)

        self.model.train()
        tbar = self.tqdm(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
        )
        metrics = defaultdict(list)
        for batch in batches:
            
            total_actions = sum_across_processes(
                sum([minibatch["action_mask"].sum() for minibatch in batch])
            )
            
            for minibatch in batch:

                logps, entropy = self.forward(minibatch, True)
                ratio = (logps - minibatch["old_logps"]).exp()
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip,
                    1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                entropy = entropy.sum() / total_actions
                policy_loss = - torch.min(objective, clipped_objective).sum() / total_actions
                loss = policy_loss - self.config.entropy.coef * entropy
                clip_ratio = (objective > clipped_objective).sum() / total_actions

                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl_loss = compute_kl_term(
                        logps,
                        minibatch["ref_logps"],
                        self.config.kl.loss_estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl_loss

                (loss * self.device_mesh.size()).backward() 

                tbar.update()
                # The losses on each device (resp. of minibatches within a 
                # batch) are accumulated but the value will be averaged in 
                # `Worker.log`. Therefore we multiply the world size (resp. 
                # bsz) here to get the correct value.
                metrics["actor/entropy"].append(self.device_mesh.size() * len(batch) * entropy.item())
                metrics["actor/loss"].append(self.device_mesh.size() * len(batch) * loss.item())
                metrics["actor/clip_ratio"].append(self.device_mesh.size() * len(batch) * clip_ratio.item())

            grad_norm = self.optimizer_step()
            metrics["actor/grad_norm"].append(grad_norm)

        self.log(metrics, step)
        if self.config.save_freq is not None and (step + 1) % self.config.save_freq == 0:
            self.save(step)

        self.sync_llm_weight()

    def sync_llm_weight(self):

        torch.cuda.empty_cache()
        # or llm.resume_memory_occupation() may OOM
        if self.rollout_device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation()

        named_tensors = [(k, v) for k, v in self.model.state_dict().items()]
        for idx, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor.full_tensor()
            )
            serialized_tensors = [
                None for _ in range(self.rollout_device_mesh["tp"].size())
            ] if self.rollout_device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.rollout_device_mesh["tp"].get_group(),
            )
            if self.rollout_device_mesh["tp"].get_local_rank() == 0:
                self.llm.update_weights_from_tensor(
                    named_tensors=[(
                        name, LocalSerializedTensor(values=serialized_tensors)
                    )],
                    flush_cache=(idx == len(named_tensors) - 1)
                )
        dist.barrier()
        self.offload_model_to_cpu()
        # Offload params here, or the params cannot be loaded.