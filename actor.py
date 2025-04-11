from typing import Dict, List
import importlib.util
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm
from worker import Worker
from utils import (
    all_gather,
    dispatch,
    compute_kl_term,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer
)


class Actor(Worker):

    def __init__(self, config, device_mesh, train):
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
            enable_sleep_mode=True
        )

        self.original_rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(self.rollout_device_mesh["dp"].get_local_rank())
        self.rollout_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.original_rng_state)

        self.train_sampling_params = SamplingParams(
            temperature=self.config.rollout.train_temperature,
            max_tokens=self.config.rollout.max_response_length
        )
        self.test_sampling_params = SamplingParams(
            temperature=self.config.rollout.test_temperature,
            max_tokens=self.config.rollout.max_response_length
        )

    def rollout(self, data_list, train, step):

        data_list = all_gather(data_list, self.rollout_device_mesh["tp"])

        # TODO: context manager
        self.set_rng_state(self.rollout_rng_state)
        responses = self.llm.generate(
            [ex["prompt"] for ex in data_list],
            sampling_params=self.train_sampling_params if train else self.test_sampling_params,
            use_tqdm=(self.device_mesh.get_rank() == 0)
        )
        self.set_rng_state(self.original_rng_state)

        if train:
            self.llm.sleep()

        for ex, response in zip(data_list, responses):
            ex.update({
                "response": response.outputs[0].text,
                "response_id": list(response.outputs[0].token_ids)
            })

        data_list = dispatch(data_list, self.rollout_device_mesh["tp"])

        for ex in data_list:
            ex["reward"] = self.reward_fn(ex["response"], ex["answer"])

        self.log({f"reward/{'train' if train else 'test'}": [ex["reward"] for ex in data_list]}, step)

        return data_list


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
    def compute_logps(self, data_list, step:int):
        load_fsdp_model_to_gpu(self.model)
        minibatches = self.pack_data_to_minibatches(data_list, False)

        prefix = "old" if self.train else "ref"
        self.model.eval()
        for minibatch in (
            tqdm(minibatches, desc=f"Step {step + 1}, compute {prefix} logps")
            if self.device_mesh.get_rank() == 0 else minibatches
        ):
            minibatch[f"{prefix}_logps"] = self.forward(minibatch)
            
        offload_fsdp_model_to_cpu(self.model)
        return self.resume_minibatches_to_data_list(minibatches) 

    def update(self, data_list, step):
        load_fsdp_model_to_gpu(self.model)
        load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())
        minibatches = self.pack_data_list_to_minibatches(data_list, True)
        batches = self.group_minibatches_into_batches(minibatches)

        self.model.train()
        losses, grad_norms = [], []
        for batch in batches:
            
            total_actions = sum([minibatch["action_mask"].sum() for minibatch in batch])
            total_sequences = sum([(minibatch["eos_mask"]).sum() for minibatch in batch])
            for minibatch in batch:

                logps = self.forward(minibatch)
                ratio = (logps - minibatch["old_logps"]).exp()
                cliped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.epsilon_clip,
                    1 + self.config.epsilon_clip
                )
                loss = - torch.min(minibatch["advantages"] * ratio, minibatch["advantages"] * cliped_ratio).sum() / total_actions

                # TODO: log cliped_ratio
                if self.config.kl.type == "loss":
                    
                    kl_term = compute_kl_term(
                        logps, minibatch["ref_logps"], self.config.kl.estimator,
                        minibatch["eos_mask"] if self.config.kl.level == "sequence" else None
                    )
                    kl_loss = - self.config.kl.coef * kl_term.sum() / (total_actions if self.config.kl.level == "token" else total_sequences)
                    loss = loss + kl_loss

                loss.backward()
                losses.append(loss.item()) # TODO: first sum then avg

            grad_norm = self.model.clip_grad_norm_(self.config.max_grad_norm)
            grad_norms.append(grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.log({
            "actor/loss": losses,
            "actor/grad_norm": grad_norms
        }, step)

        offload_fsdp_optimizer(self.optimizer)
        with FSDP.summon_full_params(self.model, offload_to_cpu=True):
            state_dict = self.model.state_dict()
        offload_fsdp_model_to_cpu(self.model)
        # offload params here, or params cannot be summoned
        torch.cuda.empty_cache() # or llm.wake_up() will OOM
        self.llm.wake_up() # upload inference engine to GPU
        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(((name, param) for name, param in state_dict.items()))