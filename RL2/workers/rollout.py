from typing import Optional, Union, Dict, Any, List, Tuple, Generator, Sequence
from omegaconf import OmegaConf, DictConfig
import os
import time
import asyncio
import requests
import importlib
import multiprocessing
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server_engine import launch_server_process
from sglang.srt.utils import MultiprocessingSerializer
from sglang_router.launch_router import RouterArgs, launch_router
from RL2.datasets import (
    pack_tensor_dicts,
    RLDataset,
    StatefulCycleDataLoader,
    SampleGroup
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

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.server_args.model_path, trust_remote_code=True
            )
            self._launch_router_process()
            self.train_dataloader = self._prepare_dataloader(True)
            self.test_dataloader = self._prepare_dataloader(False)

            self._prepare_environment()
            self.sample_buffer: List[SampleGroup] = []

    def _prepare_device_mesh(self):

        world_size = dist.get_world_size()
        tp_size = self.config.server_args.tp_size
        assert world_size % tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {tp_size}."

        self.process_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="gloo"
        )
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
        self.config.train.router_url = self.config.test.router_url = f"http://{router_args.host}:{router_args.port}"
        process = multiprocessing.Process(
            target=launch_router, args=(router_args,)
        )
        process.start()
        time.sleep(3)
        assert process.is_alive()

    def _prepare_dataloader(self, train: bool):
        
        dataset = RLDataset(
            self.config.train if train else self.config.test,
            self.tokenizer
        )
        return StatefulCycleDataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda batch: batch[0]
        )

    def _prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    def _make_request(
        self,
        endpoint: str,
        url: Optional[Union[str, List[str]]] = None,
        payload: Dict[str, Any] = {},
        max_trials: int = 3,
        retry_delay: int = 1
    ):

        if self.device_mesh["tp"].get_local_rank() != 0:
            return
        
        if isinstance(url, list):
            for u in url:
                self._make_request(
                    endpoint, u, payload, max_trials, retry_delay
                )
            return
                
        for trial in range(max_trials):
            try:
                response = requests.post(
                    f"{url or self.worker_url}/{endpoint}",
                    json=payload
                )
                response.raise_for_status()
                return
            except Exception:
                if trial == max_trials - 1:
                    raise
                time.sleep(retry_delay)

    @time_logger("rollout")
    async def __call__(
        self,
        train: bool,
        step: int
    ) -> Optional[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]:

        def _schedule_tasks(
            sample_groups: Sequence[SampleGroup]
        ):

            for sample_group in sample_groups:
                pendings.add(
                    asyncio.create_task(
                        sample_group.generate(
                            self.env.generate
                        )
                    )
                )

        if dist.get_rank() == 0:

            config = self.config.train if train else self.config.test
            dataloader = self.train_dataloader if train else self.test_dataloader
            groups_to_complete = config.prompts_per_rollout or len(dataloader)
            
            tbar = progress_bar(
                total=groups_to_complete, desc="Rollout"
            )

            pendings, first_iter = set(), True
            filtered_groups, completed_groups = 0, 0
            all_tensor_dicts: List[List[Dict[str, torch.Tensor]]] = []
            metrics: Dict[str, List[Union[float, int, bool]]] = defaultdict(list)

            if train and config.partial_rollout:
                _schedule_tasks(self.sample_buffer)

            while completed_groups < groups_to_complete:

                if first_iter or (train and config.partial_rollout):
                    sample_groups = dataloader(
                        groups_to_complete - len(pendings)
                    )
                    _schedule_tasks(sample_groups)
                first_iter = False

                done, pendings = await asyncio.wait(
                    pendings, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    if completed_groups < groups_to_complete:
                        tbar.update()
                    completed_groups += 1
                    sample_group = task.result()
                    all_tensor_dicts_delta, metrics_delta = (
                        sample_group.to_all_tensor_dicts_and_metrics()
                    )
                    for k, v in metrics_delta.items():
                        metrics[k].extend(v)
                    if (
                        train and config.dynamic_filtering and
                        len(metrics_delta["rewards"]) > 1 and
                        torch.tensor(metrics_delta["rewards"]).std() == 0
                    ):
                        filtered_groups += 1
                        continue
                    all_tensor_dicts.extend(all_tensor_dicts_delta)

            # TODO: maybe save `all_tensor_dicts`
            if train and config.partial_rollout:
                self._make_request("pause_generation", self.worker_urls)
                done, _ = await asyncio.wait(pendings)
                self.sample_buffer = [task.result() for task in done]
                self._make_request("continue_generation", self.worker_urls)

            metrics["dynamic_filtering_ratio"].append(
                filtered_groups / completed_groups
            )
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": v for k, v in metrics.items()}
            gather_and_log(metrics, step)

        dist.barrier(group=self.process_group)

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
        dist.barrier(group=self.process_group)
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
            param_size = tensor.numel() * tensor.element_size()

            if bucket_size > 0 and bucket_size + param_size > (self.config.bucket_size << 20):

                _update_tensor_bucket(dtype_to_named_tensors)
                dtype_to_named_tensors = defaultdict(list)
                bucket_size = 0

            tensor = tensor.to(
                torch.cuda.current_device(), non_blocking=True
            )
            if isinstance(tensor, DTensor):
                tensor = tensor.redistribute(
                    placements=[Replicate()] * tensor.device_mesh.ndim,
                    async_op=True
                ).to_local()
            
            dtype_to_named_tensors[tensor.dtype].append((name, tensor))
            bucket_size += param_size

        _update_tensor_bucket(dtype_to_named_tensors)

        self._make_request(
            "resume_memory_occupation", payload={"tags": ["kv_cache"]}
        )
        dist.barrier(group=self.process_group)