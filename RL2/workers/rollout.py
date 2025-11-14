from typing import Set, Optional, Union, Dict, Any, List, Tuple, Generator, Sequence
from omegaconf import OmegaConf, DictConfig
import os
import time
import asyncio
import aiohttp
import requests
import importlib
import multiprocessing
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server_engine import launch_server_process
from sglang.srt.utils import MultiprocessingSerializer
from sglang_router.launch_router import RouterArgs, launch_router
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
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket


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

            self.pendings: Set[asyncio.Task] = set()

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

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
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
        monkey_patch_torch_reductions()

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
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:

        if self.device_mesh["tp"].get_local_rank() != 0:
            return
        
        if isinstance(url, list):
            return [
                self._make_request(
                    endpoint, u, method, payload, max_trials, retry_delay
                )
                for u in url
            ]
                
        for trial in range(max_trials):
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
            except Exception:
                if trial == max_trials - 1:
                    raise
                time.sleep(retry_delay)

    async def _async_generate(
        self,
        states: List[int],
        sampling_params: Dict[str, Any],
        max_trials: int = 3,
        retry_delay: int = 1
    ) -> Dict[str, Any]:
        
        payload = {
            "input_ids": states,
            "sampling_params": sampling_params,
            "return_logprob": True
        }

        async with aiohttp.ClientSession() as session:
            for trial in range(max_trials):
                try:
                    async with session.post(
                        f"{self.router_url}/generate",
                        json=payload
                    ) as response:
                        return await response.json(content_type=None)
                except Exception:
                    if trial == max_trials - 1:
                        raise
                    await asyncio.sleep(retry_delay)

    def _schedule_experience_tasks(
        self,
        experience_groups: Sequence[ExperienceGroup]
    ):

        for experience_group in experience_groups:
            self.pendings.add(
                asyncio.create_task(
                    experience_group.make(
                        self._async_generate,
                        self.env.step,
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

            experience_buffer = []
            if self.pendings:
                # Flush pendings to avoid tasks in previous steps calling the
                # inference engine. We wait for tasks to finish at the begining 
                # of next step rather than at the end of the last step so that 
                # the time-consuming `env.step` can overlap with the training
                done, self.pendings = await asyncio.wait(self.pendings)
                experience_buffer = [task.result() for task in done]
            self._make_request("continue_generation", self.worker_urls)

            prompts_per_rollout = (
                self.config.train_prompts_per_rollout if train else
                self.config.test_prompts_per_rollout or len(dataloader)
            )
            tbar = progress_bar(
                total=prompts_per_rollout, desc="Rollout"
            )
            experiences_to_done = prompts_per_rollout

            first_iter, filtered_prompts = True, 0
            all_tensor_dicts: List[List[Dict[str, torch.Tensor]]] = []
            metrics: Dict[str, List[Union[float, int, bool]]] = defaultdict(list)

            if train and self.config.partial_rollout:
                self._schedule_experience_tasks(experience_buffer)

            while experiences_to_done > 0:

                if first_iter or (train and self.config.partial_rollout):
                    experience_groups = dataloader(
                        prompts_per_rollout - len(self.pendings)
                    )
                    self._schedule_experience_tasks(experience_groups)
                first_iter = False

                done, self.pendings = await asyncio.wait(
                    self.pendings, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    if experiences_to_done == 0:
                        break
                    tbar.update()
                    experiences_to_done -= 1
                    experience_group = task.result()
                    all_tensor_dicts_delta, metrics_delta = (
                        experience_group.to_all_tensor_dicts_and_metrics()
                    )
                    for k, v in metrics_delta.items():
                        metrics[k].extend(v)
                    if (
                        len(all_tensor_dicts_delta) > 1 and
                        self.config.dynamic_filtering and
                        torch.tensor(metrics_delta["rewards"]).std() == 0
                    ):
                        filtered_prompts += 1
                        continue
                    all_tensor_dicts.extend(all_tensor_dicts_delta)
            # TODO: maybe save `all_tensor_dicts`
            self._make_request("pause_generation", self.worker_urls)

            metrics["dynamic_filtering_ratio"].append(
                filtered_prompts / prompts_per_rollout
            )
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": v for k, v in metrics.items()}
            gather_and_log(metrics, step)

        dist.barrier()

        if not train:
            return

        self._make_request("release_memory_occupation")

        if dist.get_rank() != 0:
            return None, None

        tensor_dicts: List[Dict[str, torch.Tensor]] = [
            td for tds in all_tensor_dicts for td in tds
        ]
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
        self, named_tensor_generator: Generator[Tuple[str, torch.Tensor], None, None]
    ):

        torch.cuda.empty_cache()
        dist.barrier()
        # or resume_memory_occupation() may OOM
        self._make_request(
            "resume_memory_occupation", payload={"tags": ["weights"]}
        )

        def _update_tensor_bucket(
            dtype_to_named_tensors: Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]
        ):

            torch.cuda.synchronize()
            serialized_tensors = []
            for _, named_tensors in dtype_to_named_tensors.items():

                flattened_tensor_bucket = FlattenedTensorBucket(named_tensors)
                flattened_tensor_data = {
                    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                    "metadata": flattened_tensor_bucket.get_metadata()
                }
                serialized_tensors.append(
                    MultiprocessingSerializer.serialize(
                        flattened_tensor_data, output_str=True
                    )
                )

            gathered_serialized_tensors = [
                None for _ in range(self.device_mesh["tp"].size())
            ] if self.device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensors,
                gathered_serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            
            if self.device_mesh["tp"].get_local_rank() == 0:

                num_dtypes = len(gathered_serialized_tensors[0])
                for i in range(num_dtypes):
                    # HTTP server only sends meta data. Actual weights will be directly 
                    # copied from GPUs
                    self._make_request(
                        "update_weights_from_tensor",
                        payload={
                            "serialized_named_tensors": [
                                tensors[i] for tensors in gathered_serialized_tensors
                            ],
                            "load_format": "flattened_bucket",
                            "flush_cache": False
                        }
                    )
        
        dtype_to_named_tensors = defaultdict(list)
        bucket_size = 0
        for name, tensor in named_tensor_generator:

            tensor = tensor.to(
                torch.cuda.current_device(), non_blocking=True
            )
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            param_size = tensor.numel() * tensor.element_size()

            if bucket_size > 0 and bucket_size + param_size > (self.config.bucket_size << 20):

                _update_tensor_bucket(dtype_to_named_tensors)
                dtype_to_named_tensors = defaultdict(list)
                bucket_size = 0
            
            dtype_to_named_tensors[tensor.dtype].append((name, tensor))
            bucket_size += param_size

        _update_tensor_bucket(dtype_to_named_tensors)

        self._make_request(
            "resume_memory_occupation", payload={"tags": ["kv_cache"]}
        )
        dist.barrier()