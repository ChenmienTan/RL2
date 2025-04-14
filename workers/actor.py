from typing import Dict, List, Optional
import importlib.util
from collections import defaultdict
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm
from workers.base import Worker
from algs import compute_kl_term
from utils.ring_attn import RingAttnManager
from utils.comm import gather_and_concat_list, sum_across_processes


class RolloutRngStateManager:

    def __init__(self, device_mesh):

        self.original_rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(device_mesh.get_local_rank())
        self.rollout_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.original_rng_state)

    def __enter__(self):
        torch.cuda.set_rng_state(self.rollout_rng_state)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.set_rng_state(self.original_rng_state)


class Actor(Worker):

    def __init__(self, config, device_mesh, train: bool):
        super().__init__(config, device_mesh, train)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32 if train else torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        self.prepare_model_optimizer()
        if train:
            self.prepare_reward_fn()
            self.prepare_inference_engine()

    def prepare_reward_fn(self):

        spec = importlib.util.spec_from_file_location("custom_module", self.config.rollout.reward_fn_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reward_fn = module.reward_fn

    def prepare_inference_engine(self):

        self.rollout_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(self.device_mesh.size() // self.config.rollout.tp_size, self.config.rollout.tp_size)
        )

        self.llm = LLM(
            self.config.model_name,
            tensor_parallel_size=self.config.rollout.tp_size,
            distributed_executor_backend="external_launcher",
            # SPMD, see https://github.com/vllm-project/vllm/issues/11400
            gpu_memory_utilization=self.config.rollout.gpu_memory_utilization,
            enable_sleep_mode=True,
            seed=self.rollout_device_mesh["dp"].get_local_rank()
        )

        self.rollout_rng_state_manager = RolloutRngStateManager(self.rollout_device_mesh["dp"])

        self.train_sampling_params = SamplingParams(
            n=self.config.rollout.rollout_per_prompt,
            temperature=self.config.rollout.train_temperature,
            max_tokens=self.config.rollout.max_response_length
        )
        self.test_sampling_params = SamplingParams(
            temperature=self.config.rollout.test_temperature,
            max_tokens=self.config.rollout.max_response_length
        )

    def rollout(
        self,
        data_list: List[Dict],
        train: bool,
        step: int
    ) -> Optional[List[Dict[str, torch.Tensor]]]:

        with self.rollout_rng_state_manager:
            responses = self.llm.generate(
                [ex["prompt"] for ex in data_list],
                sampling_params=self.train_sampling_params
                if train else self.test_sampling_params,
                use_tqdm=(self.device_mesh.get_rank() == 0)
            )

        if train:
            # If test, llm will soon be called again. See `Trainer.train`.
            self.llm.sleep() # TODO: perhaps level=2
            torch.cuda.empty_cache()

        data_list = [
            {
                "response": output.text,
                "response_id": list(output.token_ids),
                **ex
            }
            for ex, response in zip(data_list, responses)
            for output in response.outputs
        ]

        rank = self.sp_device_mesh["sp"].get_local_rank()
        batch_size_per_device = len(data_list) // self.sp_device_mesh["sp"].size()
        data_list = data_list[rank * batch_size_per_device:(rank + 1) * batch_size_per_device]
        # Each device grades its respective data to avoid duplicate computation.

        for ex in data_list:
            ex["reward"] = self.reward_fn(ex["response"], ex["answer"])
        # Only support outcome reward. RM should be served remotely if there is

        self.log({f"reward/{'train' if train else 'test'}": [ex["reward"] for ex in data_list]}, step)

        if train:

            tensor_data_list = []
            for ex in data_list:

                prompt_id = ex["prompt_id"]
                response_id = ex["response_id"]
                reward = ex["reward"]

                states = prompt_id + response_id[:-1]
                actions = (len(prompt_id) - 1) * [0] + response_id
                rewards = (len(prompt_id) + len(response_id) - 2) * [0] + [reward]
                position_ids = list(range(len(prompt_id) + len(response_id) - 1))
                action_mask = (len(prompt_id) - 1) * [0] + len(response_id) * [1]
                eos_mask = (len(prompt_id) + len(response_id) - 2) * [0] + [1]

                tensor_data_list.append({
                    "states": torch.LongTensor([states]),
                    "actions": torch.LongTensor([actions]),
                    "rewards": torch.FloatTensor([rewards]),
                    "position_ids": torch.LongTensor([position_ids]),
                    "action_mask": torch.LongTensor([action_mask]),
                    "eos_mask": torch.LongTensor([eos_mask])
                })

            return gather_and_concat_list(tensor_data_list, self.device_mesh)

    def forward(self, minibatch: Dict[str, torch.Tensor]) -> torch.Tensor:

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits / self.config.rollout.train_temperature

        return torch.gather(
            logits.log_softmax(-1),
            dim=-1,
            index=minibatch["actions"].unsqueeze(-1)
        ).squeeze(-1) * minibatch["action_mask"]

    @torch.no_grad()
    def compute_logps(
        self,
        data_list: List[Dict[str, torch.Tensor]],
        step: int
    ) -> List[Dict[str, torch.Tensor]]:
        self.load_model_to_gpu()
        minibatches = self.scatter_and_pack_data_list(data_list, False)

        prefix = "old" if self.train else "ref"
        
        total_actions = sum_across_processes(
            sum([minibatch["action_mask"].sum() for minibatch in minibatches])
        )
        total_sequences = sum_across_processes(
            sum([minibatch["eos_mask"].sum() for minibatch in minibatches])
        )
        
        self.model.eval()
        metrics = defaultdict(list)
        for minibatch in (
            tqdm(minibatches, desc=f"Step {step + 1}, compute {prefix} logps")
            if self.device_mesh.get_rank() == 0 else minibatches
        ):
            minibatch[f"{prefix}_logps"] = self.forward(minibatch)

            if not self.train and self.config.kl.type == "reward":
                kl_term = compute_kl_term(
                    minibatch["old_logps"], minibatch["ref_logps"],
                    self.config.kl.estimator,
                    minibatch["eos_mask"]
                    if self.config.kl.level == "sequence"
                    else None               
                )
                minibatch["rewards"] -= self.config.kl.coef * kl_term
                kl = kl_term.sum() / (
                    total_actions
                    if self.config.kl.level == "token"
                    else total_sequences
                )
                metrics["kl"].append(self.device_mesh.size() * len(minibatches) * kl)
        
        if not self.train and self.config.kl.type == "reward":
            self.log(metrics, step)

        self.offload_model_to_cpu()
        return self.resume_and_gather_data_list(minibatches) 

    def update(self, data_list: List[Dict[str, torch.Tensor]], step: int):
        self.load_model_to_gpu()
        self.load_optimizer_to_gpu()
        minibatches = self.scatter_and_pack_data_list(data_list, True)
        batches = self.group_minibatches_into_batches(minibatches)

        self.model.train()
        metrics = defaultdict(list)
        tbar = tqdm(total=len(minibatches), desc=f"Step {step + 1}, update actor")
        for batch in batches:
            
            total_actions = sum_across_processes(
                sum([minibatch["action_mask"].sum() for minibatch in batch])
            )
            total_sequences = sum_across_processes(
                sum([minibatch["eos_mask"].sum() for minibatch in batch])
            )
            
            for minibatch in batch:

                with (
                    RingAttnManager(self.sp_device_mesh["sp"], minibatch["seqlens"])
                    if self.sp_device_mesh["sp"].size() > 1 else nullcontext()
                ):
                    logps = self.forward(minibatch)
                ratio = (logps - minibatch["old_logps"]).exp()
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip,
                    1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                loss = - torch.min(objective, clipped_objective).sum() / total_actions
                clip_ratio = (objective > clipped_objective).sum() / total_actions

                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    # TODO: problematic when using ring attn and `kl.level=sequence`
                    kl = compute_kl_term(
                        logps, minibatch["ref_logps"],
                        self.config.kl.estimator,
                        minibatch["eos_mask"]
                        if self.config.kl.level == "sequence" else None
                    ).sum() / (
                        total_actions
                        if self.config.kl.level == "token"
                        else total_sequences
                    )
                    loss = loss + self.config.kl.coef * kl

                    metrics["kl"].append(self.device_mesh.size() * len(batch) * kl)

                loss.backward()         
                metrics["actor/loss"].append(self.device_mesh.size() * len(batch) * loss.item())
                # The losses on different data processes (resp. of 
                # minibatches within a batch) are accumulated but the 
                # value will be averaged in `Worker.log`. Therefore 
                # we multiply the world size (resp. bsz) here to get 
                # the correct value.
                metrics["actor/clip_ratio"].append(self.device_mesh.size() * len(batch) * clip_ratio.item())
                tbar.update()

            grad_norm = self.model.clip_grad_norm_(self.config.max_grad_norm)
            metrics["grad_norm"].append(grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.log(metrics, step)

        self.offload_optimizer_to_cpu()
        with FSDP.summon_full_params(self.model, offload_to_cpu=True):
            state_dict = self.model.state_dict()
        self.offload_model_to_cpu()
        # offload params here, or params cannot be summoned
        torch.cuda.empty_cache() # or llm.wake_up() will OOM
        self.llm.wake_up() # upload inference engine to GPU
        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(((name, param) for name, param in state_dict.items()))