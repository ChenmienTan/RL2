from omegaconf import OmegaConf
import os
import json
import asyncio
import importlib
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
from RL2.workers import Worker
from RL2.datasets import tokenize_messages
from RL2.utils.comm import split_and_scatter_list, gather_and_concat_list
from RL2.utils.logging import time_logger, gather_and_log


class Rollout(Worker):

    def __init__(self, config):
        super().__init__(config, None)
        
        self.prepare_environment_variables()
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.prepare_environment()

            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.llm = Engine(
                model_path=config.model_name,
                dtype="bfloat16",
                tp_size=self.device_mesh["tp"].size(),
                mem_fraction_static=config.gpu_memory_utilization,
                enable_memory_saver=True,
                port=30000 + dist.get_rank()
            )
        
            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

        dist.barrier()

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % self.config.tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {self.config.tp_size}."
        self.dp_size = world_size // self.config.tp_size
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(self.dp_size, self.config.tp_size)
        )

    def prepare_environment_variables(self):

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            cuda_visible_devices = cuda_visible_devices.split(",")
            cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
        else:
            cuda_visible_device = os.environ["LOCAL_RANK"]
        cuda_visible_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            cuda_visible_device,
            self.device_mesh["tp"].get_group(),
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    def prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)
        
    async def rollout(self, ex, train):

        messages, answer = ex["messages"], ex["answer"]
        metric = defaultdict(list)
        for turn in range(self.config.max_turns):

            if self.config.apply_chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                prompt = "".join([
                    msg["content"] for msg in messages
                ])
            
            response = await self.llm.async_generate(
                prompt,
                sampling_params=self.train_sampling_params
                if train else self.test_sampling_params
            )

            meta_info = response["meta_info"]
            metric["response_length"].append(meta_info["completion_tokens"])
            metric["length_clip_ratio"].append(
                meta_info["finish_reason"]["type"] == "length"
            )

            # Current SGLang engine will generate sequence longer than 
            # `max_new_tokens`.
            # TODO (P1): Check whether all configurations are properly set 
            # and whether the bug has been fixed in the latest version.
            content = self.tokenizer.decode(
                self.tokenizer.encode(
                    response["text"], add_special_tokens=False
                )[:meta_info["completion_tokens"]]
            )
            messages.append(
                {"role": "assistant", "content": content}
            )

            # Do not invoke tools in the last turn.
            if turn + 1 == self.config.max_turns:
                break

            env_messages = self.env.interact(messages)
            # Terminate if no tool is invoked.
            if len(env_messages) == 0:
                break

            messages.extend(env_messages)

        reward = self.env.reward_fn(messages, answer)

        td = tokenize_messages(
            self.tokenizer,
            messages,
            self.config.apply_chat_template
        )
        td["rewards"] = torch.FloatTensor(
            (td["states"].shape[-1] - 1) * [0] + [reward]
        )

        metric["n_turns"].append(turn + 1)
        metric["rewards"].append(reward)
        metric["trajectory_length"].append(len(td["states"]))

        return td, messages, metric

    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):

        # The data is distributed from rank 0 before each worker operation
        # and gathered before the next operation, which facilitates to do
        # model-agnostic operations, e.g., computing advantages, globally 
        # and guarantees the load balancing across all model computations.
        if self.device_mesh["tp"].get_local_rank() == 0:

            data_list = split_and_scatter_list(
                data_list, self.device_mesh["dp"]
            )
            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                tqdm.gather(
                    *(self.rollout(ex, train) for ex in data_list),
                    desc="Rollout", position=1, leave=False,
                    disable=(dist.get_rank() != 0)
                )
            )
            if train:
                # If test, llm will soon be called again. See `Trainer.train`.
                self.llm.release_memory_occupation()

        dist.barrier()

        if self.device_mesh["tp"].get_local_rank() == 0:

            tensor_dicts, all_messages, metrics = map(list, zip(*outputs))

            if dist.get_rank() == 0:
                tqdm.write(json.dumps(all_messages[0], indent=4))

            suffix = "train" if train else "test"
            metrics = {
                f"{k}/{suffix}": sum([metric[k] for metric in metrics], [])
                for k in metrics[0].keys()
            }
            gather_and_log(metrics, self.device_mesh["dp"], step)

            if not train:
                return

            tensor_dicts = gather_and_concat_list(
                tensor_dicts, self.device_mesh["dp"]
            )

            if dist.get_rank() == 0:
                if not self.config.dynamic_filtering:
                    return tensor_dicts

                rewards = torch.FloatTensor(
                    [td["rewards"].sum() for td in tensor_dicts]
                ).view(-1, self.config.responses_per_prompt)
                are_filtered = (rewards.std(-1) == 0).tolist()
                wandb.log({
                    "dynamic_filtering_ratio": sum(are_filtered) / len(are_filtered)
                }, step=step)
                return sum([
                    tensor_dicts[idx * self.config.responses_per_prompt:(idx + 1) * self.config.responses_per_prompt]
                    for idx, is_filtered in enumerate(are_filtered)
                    if not is_filtered
                ], [])
        
    @time_logger("update_rollout")
    def update(self, state_dict, step):

        torch.cuda.empty_cache()
        # or llm.resume_memory_occupation() may OOM
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation()
        
        for idx, (name, tensor) in enumerate(state_dict.items()):
            tensor = tensor.to(torch.cuda.current_device())
            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            )
            serialized_tensors = [
                None for _ in range(self.device_mesh["tp"].size())
            ] if self.device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            if self.device_mesh["tp"].get_local_rank() == 0:
                self.llm.update_weights_from_tensor(
                    named_tensors=[(
                        name, LocalSerializedTensor(values=serialized_tensors)
                    )],
                    flush_cache=(idx == len(state_dict) - 1)
                )
        dist.barrier()