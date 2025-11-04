from typing import Optional, Union, Dict, Any, List, Tuple, Generator, Set, Sequence
from omegaconf import OmegaConf, DictConfig
import os
import time
import base64
import asyncio
import aiohttp
import requests
import importlib
import multiprocessing
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server_engine import launch_server_process
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang_router.launch_router import RouterArgs, launch_router
import wandb
from RL2.datasets import (
    pack_tensor_dicts,
    StatefulCycleDataLoader,
    ExperienceGroup
)
from RL2.utils.communication import get_host, get_available_port
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log
)

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError: # older version of SGLang
    from sglang.srt.patch_torch import monkey_patch_torch_reductions


class Rollout:

    def __init__(self, config: DictConfig):
        
        self.config = config
        self._prepare_device_mesh()
        self._prepare_environment_variables()

        if self.device_mesh["tp"].get_local_rank() == 0:

            self._launch_server_process()
            self.worker_urls = [
                None for _ in range(self.device_mesh["dp"].size())
            ] if self.device_mesh["dp"].get_local_rank() == 0 else None
            dist.gather_object(
                self.worker_url,
                self.worker_urls,
                group_dst=0,
                group=self.device_mesh["dp"].get_group(),
            )
        
        if dist.get_rank() == 0:

            self._prepare_environment()
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.server_args.model_path, trust_remote_code=True
            )
            self._launch_router_process()
            self.experience_groups = []

            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

    def _prepare_device_mesh(self):

        world_size = dist.get_world_size()
        tp_size = self.config.server_args.tp_size
        assert world_size % tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {tp_size}."
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(world_size // tp_size, tp_size)
        )

    def _prepare_environment_variables(self):

        # TODO: check whether this is required
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

    def _prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    def _launch_server_process(self):

        server_args = OmegaConf.to_container(self.config.server_args)
        server_args = ServerArgs(
            enable_memory_saver=True,
            host=get_host(),
            port=get_available_port(),
            log_level="error",
            **server_args
        )
        self.worker_url = server_args.url()
        launch_server_process(server_args)

    def _launch_router_process(self):

        router_args = RouterArgs(
            worker_urls=self.worker_urls,
            policy="cache_aware",
            host=get_host(),
            port=get_available_port(),
            log_level="error"
        )
        self.router_url = f"http://{router_args.host}:{router_args.port}"
        process = multiprocessing.Process(
            target=launch_router, args=(router_args,)
        )
        process.start()
        time.sleep(3)
        assert process.is_alive()

    def _make_request(
        self,
        endpoint: str,
        url: Optional[Union[str, List[str]]] = None,
        method: str = "POST",
        payload: Dict[str, Any] = {},
        max_trials: int = 3,
        retry_delay: int = 1
    ) -> Union[Optional[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:

        if self.device_mesh["tp"].get_local_rank() != 0:
            return
        
        if isinstance(url, list):
            return [
                self._make_request(
                    endpoint, u, method, payload, max_trials, retry_delay
                )
                for u in url
            ]
                
        for _ in range(max_trials):
            try:
                if method == "POST":
                    response = requests.post(
                        f"{url or self.worker_url}/{endpoint}",
                        json=payload
                    )
                elif method == "GET":
                    response = requests.get(
                        f"{url or self.worker_url}/{endpoint}"
                    )
                response.raise_for_status()
                return response
            except:
                time.sleep(retry_delay)

    async def _async_generate(
        self,
        states: List[int],
        train: bool,
        max_trials: int = 3,
        retry_delay: int = 1
    ) -> Optional[Dict[str, Any]]:
        
        payload = {
            "input_ids": states,
            "sampling_params": self.train_sampling_params if train else self.test_sampling_params,
            "return_logprob": True
        }

        async with aiohttp.ClientSession() as session:
            for _ in range(max_trials):
                try:
                    async with session.post(
                        f"{self.router_url}/generate",
                        json=payload
                    ) as response:
                        return await response.json(content_type=None)
                except:
                    await asyncio.sleep(retry_delay)

    def add_make_experience_tasks(
        self,
        pendings: Set[asyncio.Task],
        experience_groups: Sequence[ExperienceGroup],
        train: bool
    ):
        for experience_group in experience_groups:
            pendings.add(
                asyncio.create_task(
                    experience_group.make(
                        self._async_generate,
                        self.env.step,
                        train,
                        getattr(self.env, "reset", None)
                    )
                )
            )

    @time_logger("rollout")
    async def __call__(
        self,
        dataloader: StatefulCycleDataLoader,
        train: bool,
        step: int
    ) -> Optional[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]:

        if dist.get_rank() == 0:

            self._make_request(
                "continue_generation", self.worker_urls
            )
            prompts_per_rollout = (
                self.config.train_prompts_per_rollout if train else
                self.config.test_prompts_per_rollout or len(dataloader)
            )
            tbar = progress_bar(
                total=prompts_per_rollout, desc="Rollout"
            )
            num_tasks_to_finish = prompts_per_rollout
            first_iter = True
            pendings = set()
            all_tensor_dicts: List[List[Dict[str, torch.Tensor]]] = []
            metrics: List[Dict[str, Union[List[float], List[int], List[bool]]]] = []
            if self.config.partial_rollout and train:
                self.add_make_experience_tasks(
                    pendings, self.experience_groups, train
                )
            while num_tasks_to_finish > 0:
                if first_iter or (self.config.partial_rollout and train):
                    experience_groups = dataloader(
                        prompts_per_rollout - len(pendings)
                    )
                    self.add_make_experience_tasks(
                        pendings, experience_groups, train
                    )
                done, pendings = await asyncio.wait(
                    pendings, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if num_tasks_to_finish > 0:
                        tbar.update()
                        num_tasks_to_finish -= 1
                        first_iter = False # TODO: print when first_iter is True
                        experience_group = task.result()
                        all_tensor_dicts_delta, metrics_delta = experience_group.to_all_tensor_dicts_and_metrics()
                        all_tensor_dicts.extend(all_tensor_dicts_delta)
                        metrics.extend(metrics_delta)

            self._make_request(
                "pause_generation", self.worker_urls
            )
            if pendings:
                done, _ = await asyncio.wait(pendings)
                self.experience_groups = [
                    task.result() for task in done
                ]

            suffix = "train" if train else "test"
            metrics = {
                f"{k}/{suffix}": sum([metric[k] for metric in metrics], [])
                for k in metrics[0].keys()
            }
            gather_and_log(metrics, step)

        dist.barrier()

        if not train:
            return

        self._make_request("release_memory_occupation")

        if dist.get_rank() != 0:
            return None, None

        group_size = self.config.responses_per_prompt
        if group_size > 1 and self.config.dynamic_filtering:

            rewards = torch.FloatTensor([
                sum([td["rewards"].sum().item() for td in tensor_dicts])
                for tensor_dicts in all_tensor_dicts
            ]).view(-1, group_size)
            are_filtered = rewards.std(-1) == 0
            all_tensor_dicts = sum([
                all_tensor_dicts[idx * group_size:(idx + 1) * group_size]
                for idx, is_filtered in enumerate(are_filtered)
                if not is_filtered
            ], [])
            wandb.log({
                "dynamic_filtering_ratio": are_filtered.float().mean().item()
            }, step=step)

        tensor_dicts: List[Dict[str, torch.Tensor]] = sum(all_tensor_dicts, [])
        tensor_dict: Dict[str, torch.Tensor] = pack_tensor_dicts(tensor_dicts)
        seqs = torch.LongTensor([
            len(tensor_dicts) for tensor_dicts in all_tensor_dicts
        ])
        cu_seqs = torch.cumsum(
            torch.cat((torch.LongTensor([0]), seqs)), dim=0
        )
        
        return tensor_dict, cu_seqs
    
    @torch.no_grad()
    def update(
        self,
        named_tensor_generator: Generator[Tuple[str, torch.Tensor], None, None]
    ):

        torch.cuda.empty_cache()
        dist.barrier()
        # or resume_memory_occupation() may OOM
        self._make_request(
            "resume_memory_occupation",
            payload={"tags": ["weights"]}
        )
        
        for name, tensor in named_tensor_generator:
            # TODO: bucketize parameters to improve efficiency
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
                named_tensors = [
                    (name, LocalSerializedTensor(values=serialized_tensors))
                ]
                serialized_named_tensors = [
                    MultiprocessingSerializer.serialize(named_tensors)
                    for _ in range(self.device_mesh["tp"].size())
                ]
                serialized_named_tensors = [
                    base64.b64encode(snt).decode("utf-8")
                    for snt in serialized_named_tensors
                ]
                self._make_request(
                    "update_weights_from_tensor",
                    payload={
                        "serialized_named_tensors": serialized_named_tensors,
                        "flush_cache": False
                    }
                )
        self._make_request(
            "resume_memory_occupation",
            payload={"tags": ["kv_cache"]}
        )