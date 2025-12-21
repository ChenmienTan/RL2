from typing import Optional, Union, Dict, List, Tuple, Generator, Sequence
from omegaconf import OmegaConf, DictConfig
import os
import asyncio
import uvicorn
import importlib
import multiprocessing
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.entrypoints.http_server import app, _GlobalState, set_global_state
from sglang.srt.utils import (
    set_uvicorn_logging_configs,
    MultiprocessingSerializer
)
from sglang_router.launch_router import RouterArgs, launch_router
from RL2.datasets import (
    pack_tensor_dicts,
    RLDataset,
    StatefulCycleDataLoader,
    SampleGroup
)
from RL2.utils.communication import (
    get_host,
    get_available_port,
    GLOO_GROUP,
    gather_and_concat_list,
    async_request
)
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


class Router:
    """
    SGLang router is wrapped as a context for two purpose:
        - ensure the router only exists during rollout, avioding health 
        checks when server is offloaded
        - terminate the router process when errors occur so that the 
        training task can exit properly
    """
    def __init__(self, worker_urls):
        self.worker_urls = worker_urls

    async def __aenter__(self):

        router_args = RouterArgs(
            host=get_host(),
            port=get_available_port(),
            log_level="error"
        )
        url = f"http://{router_args.host}:{router_args.port}"
        self.process = multiprocessing.Process(
            target=launch_router, args=(router_args,)
        )
        self.process.start()
        await async_request(url, "health", "GET", 10)
        await asyncio.gather(*[
            async_request(url, f"add_worker?url={worker_url}")
            for worker_url in self.worker_urls
        ])
        return url

    async def __aexit__(self, type, val, tb):
        
        self.process.terminate()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.kill()


class Rollout:

    def __init__(self, config: DictConfig):
        
        self.config = config
        self._prepare_device_mesh()
        self._prepare_environment_variables()
        
        if dist.get_rank() == 0:

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.server_args.model_path, trust_remote_code=True
            )
            self.train_dataloader = self._prepare_dataloader(True)
            self.test_dataloader = self._prepare_dataloader(False)

            self._prepare_environment()
            self.sample_buffer: List[SampleGroup] = []

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

    async def launch_server_process(self):
        """
        Make SGLang server a coroutine rather than a subprocess so that 
        the task can exit properly when training or inference error occurs.
        """
        if not self.device_mesh["tp"].get_local_rank() == 0:
            return

        # TODO: support cross-node server
        server_args = OmegaConf.to_container(self.config.server_args)
        server_args = ServerArgs(
            enable_memory_saver=True,
            host=get_host(),
            port=get_available_port(),
            log_level="error",
            **server_args
        )
        self.worker_url = server_args.url()
        
        (
            tokenizer_manager,
            template_manager,
            scheduler_info
        ) = _launch_subprocesses(server_args=server_args)
        set_global_state(
            _GlobalState(
                tokenizer_manager=tokenizer_manager,
                template_manager=template_manager,
                scheduler_info=scheduler_info
            )
        )

        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_args = (server_args, None, None)
        set_uvicorn_logging_configs()
        config = uvicorn.Config(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = False
        asyncio.create_task(server.serve())

        await async_request(
            self.worker_url, "health", "GET", 10 * 60
        ) # up to 10 minutes to load the model
        
        self.worker_urls = gather_and_concat_list(
            [self.worker_url],
            self.device_mesh["dp"].get_group()
        )

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
                            router_url, self.env.generate
                        )
                    )
                )

        if dist.get_rank() == 0:

            async with Router(self.worker_urls) as router_url:

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

                    done, pendings = await asyncio.wait(
                        pendings, return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        if completed_groups < groups_to_complete:
                            tbar.update()
                        completed_groups += 1
                        sample_group = task.result()
                        if first_iter:
                            sample_group.print()
                            first_iter = False
                        await asyncio.to_thread(sample_group.save, step)
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

                if train and config.partial_rollout:
                    await async_request(self.worker_urls, "pause_generation")
                    done, _ = await asyncio.wait(pendings)
                    self.sample_buffer = [task.result() for task in done]
                    await async_request(self.worker_urls, "continue_generation")

            metrics["dynamic_filtering_ratio"].append(
                filtered_groups / completed_groups
            )
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": v for k, v in metrics.items()}
            gather_and_log(metrics, step)

        # Use GLOO group to avoid affecting SGLang server
        await asyncio.to_thread(dist.barrier, group=GLOO_GROUP)

        if not train:
            return

        if self.device_mesh["tp"].get_local_rank() == 0:
            await async_request(
                self.worker_url,
                "release_memory_occupation",
                json={"tags": ["weights", "kv_cache"]}
            )

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
    async def update(
        self, named_tensor_generator: Generator[Tuple[str, torch.Tensor], None, None]
    ):

        torch.cuda.empty_cache()
        dist.barrier(group=GLOO_GROUP)
        # or resume_memory_occupation() may OOM
        if self.device_mesh["tp"].get_local_rank() == 0:
            await async_request(
                self.worker_url,
                "resume_memory_occupation",
                json={"tags": ["weights"]}
            )

        async def _update_tensor_bucket(
            dtype_to_named_tensors: Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]
        ):

            torch.cuda.synchronize()
            serialized_tensors = []
            for named_tensors in dtype_to_named_tensors.values():

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
            # [
            #     [tp0_bucket0, tp0_bucket1, ...],
            #     [tp1_bucket0, tp1_bucket1, ...],
            #     ...
            # ]
            if self.device_mesh["tp"].get_local_rank() == 0:

                for serialized_named_tensors in zip(*gathered_serialized_tensors):
                    # [
                    #     (tp0_bucket0, tp1_bucket0, ...),
                    #     (tp0_bucket1, tp1_bucket1, ...),
                    #     ...
                    # ]
                    # HTTP server only sends meta data. Actual weights will be directly 
                    # copied from GPUs
                    await async_request(
                        self.worker_url,
                        "update_weights_from_tensor",
                        json={
                            "serialized_named_tensors": serialized_named_tensors,
                            "load_format": "flattened_bucket",
                            "flush_cache": False
                        }
                    )
        
        dtype_to_named_tensors = defaultdict(list)
        bucket_size = 0
        for name, tensor in named_tensor_generator:
            param_size = tensor.numel() * tensor.element_size()

            if bucket_size > 0 and bucket_size + param_size > (self.config.bucket_size << 20):

                await _update_tensor_bucket(dtype_to_named_tensors)
                dtype_to_named_tensors = defaultdict(list)
                bucket_size = 0

            tensor = tensor.to(
                torch.cuda.current_device(), non_blocking=True
            ).detach()
            if isinstance(tensor, DTensor):
                # async version of `tensor.full_tensor()`
                tensor = tensor.redistribute(
                    placements=[Replicate()] * tensor.device_mesh.ndim,
                    async_op=True
                ).to_local()
            
            dtype_to_named_tensors[tensor.dtype].append((name, tensor))
            bucket_size += param_size

        await _update_tensor_bucket(dtype_to_named_tensors)

        if self.device_mesh["tp"].get_local_rank() == 0:
            await async_request(
                self.worker_url,
                "resume_memory_occupation",
                json={"tags": ["kv_cache"]}
            )
        dist.barrier(group=GLOO_GROUP)