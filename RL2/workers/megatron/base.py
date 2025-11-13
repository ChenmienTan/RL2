from typing import Dict, Union, List, Optional, Callable, Tuple, Iterator, Any
from omegaconf import OmegaConf, DictConfig
import os
import gc
from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist
from megatron.core import (
    parallel_state as mpu,
    dist_checkpointing
)
from megatron.core.distributed import (
    DistributedDataParallel as DDP,
    DistributedDataParallelConfig
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper
)
from megatron.bridge import AutoBridge
from RL2.workers import Worker
from RL2.utils.communication import broadcast_object
from RL2.utils.sequences import scatter_data, gather_data, slide_along_cp


class MegatronWorker(Worker):

    def __init__(self, config: DictConfig, train: bool):
        super().__init__(config, train)
        
        self.bridge = AutoBridge.from_hf_pretrained(config.model_name)

        self.provider = self.bridge.to_megatron_provider()
        
        dtype = getattr(torch, config.dtype)
        self.provider.params_dtype = dtype
        self.provider.autocast_dtype = dtype
        self.provider.pipeline_dtype = dtype
        self.provider.fp16 = config.dtype == "float16"
        self.provider.bf16 = config.dtype == "bfloat16"
        self.provider.attention_backend = "flash"
        self.provider.variable_seq_lengths = True
        self.provider.moe_token_dispatcher_type = "alltoall"
        tf_config = OmegaConf.to_container(config.tf_config)
        for k, v in tf_config.items():
            setattr(self.provider, k, v)
        self.provider.sequence_parallel = self.provider.tensor_model_parallel_size > 1
        self.provider.finalize()
        if not mpu.is_initialized():
            self.provider.initialize_model_parallel(seed=42)

        ddp_config = OmegaConf.to_container(config.ddp_config)
        self.ddp_config = DistributedDataParallelConfig(**ddp_config)

    def _prepare_model_optimizer(self):

        if dist.get_rank() == 0:
            print(self.model[0].config)

        if self.train:

            optimizer_config = OmegaConf.to_container(self.config.optimizer)
            optimizer_config = OptimizerConfig(
                fp16=self.config.dtype == "float16",
                bf16=self.config.dtype == "bfloat16",
                params_dtype=getattr(torch, self.config.dtype),
                use_distributed_optimizer=self.config.ddp_config.use_distributed_optimizer,
                **optimizer_config
            )
            self.optimizer = get_megatron_optimizer(
                optimizer_config, self.model
            )

        self._offload_model_to_cpu()

    def prepare_scheduler(self, total_steps: int):

        num_training_steps = total_steps * getattr(
            self.config, "update_per_rollout", 1
        )
        scheduler_config = OmegaConf.to_container(self.config.scheduler)
        lr_warmup_steps = int(
            scheduler_config.pop("warmup_ratio") * num_training_steps
        )
        lr_decay_steps = num_training_steps - lr_warmup_steps
        self.scheduler = OptimizerParamScheduler(
            self.optimizer,
            max_lr=self.config.optimizer.lr,
            min_lr=self.config.optimizer.min_lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            wd_incr_steps=num_training_steps,
            **scheduler_config
        )

    def _scatter_data(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        pack_minibatches: bool = False,
        pair: bool = False
    ) -> Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]:
        multiple_of = mpu.get_data_parallel_world_size()
        if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
            multiple_of *= mpu.get_pipeline_model_parallel_world_size()
        max_length_per_dp = mpu.get_context_parallel_world_size() * mpu.get_tensor_model_parallel_world_size() * (
            self.config.max_length_per_device
            if torch.is_grad_enabled()
            else self.config.max_inference_length_per_device
        )
        return scatter_data(
            tensor_dict,
            mpu.get_data_parallel_group(),
            multiple_of,
            max_length_per_dp,
            self.config.update_per_rollout if pack_minibatches else None,
            pair
        )

    def _gather_data(
        self, minibatches: List[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        return gather_data(minibatches, mpu.get_data_parallel_group_gloo())

    def _offload_model_to_cpu(self):

        if not getattr(self.config, "offload_model", False):
            return

        gc.collect()
        for model in self.model:
            if isinstance(model, DDP):
                for buffers in [model.buffers, model.expert_parallel_buffers]:
                    for buffer in buffers:
                        if buffer.param_data.storage().size() > 0:
                            buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                            buffer.param_data_size = buffer.param_data.storage().size()
                            buffer.param_data.storage().resize_(0)
            else:
                for _, param in model.named_parameters():
                    param.data = param.data.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

    def _load_model_to_gpu(self):

        if not getattr(self.config, "offload_model", False):
            return

        torch.cuda.empty_cache()
        for model in self.model:
            if isinstance(model, DDP):
                for buffers in [model.buffers, model.expert_parallel_buffers]:
                    for buffer in buffers:
                        if buffer.param_data.storage().size() == 0:
                            buffer.param_data.storage().resize_(buffer.param_data_size)
                            buffer.param_data.copy_(
                                buffer.param_data.cpu_data,
                                non_blocking=True
                            )
            else:
                for _, param in model.named_parameters():
                    param.data = param.data.to(
                        torch.cuda.current_device(),
                        non_blocking=True
                    )
        gc.collect()

    def _load_optimizer_to_device(self, device: Union[torch.device, str]):

        if not getattr(self.config, "offload_optimizer", False):
            return

        for optimizer in self.optimizer.chained_optimizers:
            for group in optimizer.shard_fp32_from_float16_groups:
                for param in group:
                    param.data = param.data.to(device, non_blocking=True)
            for value in optimizer.optimizer.state.values():
                if "exp_avg" in value:
                    value["exp_avg"] = value["exp_avg"].to(
                        device, non_blocking=True
                    )
                if "exp_avg_sq" in value:
                    value["exp_avg_sq"] = value["exp_avg_sq"].to(
                        device, non_blocking=True
                    )

            gc.collect()
            torch.cuda.empty_cache()

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return mpu.get_data_parallel_world_size(with_context_parallel=True) * loss

    def _forward_backward(
        self,
        f: Callable,
        minibatches: List[Dict[str, torch.Tensor]]
    ) -> Union[Tuple[Dict[str, List[float]], torch.Tensor], List[Dict[str, torch.Tensor]]]:

        def _forward_step(
            data_iterator: Iterator, model: List[Union[DDP, nn.Module]]
        ) -> Tuple[torch.Tensor, Callable]:

            minibatch = next(data_iterator)
            minibatch, cu_seqlens = slide_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                mpu.get_tensor_model_parallel_world_size()
            )
            global_cu_seqlens = mpu.get_context_parallel_world_size() * cu_seqlens
            max_seqlen = (global_cu_seqlens[1:] - global_cu_seqlens[:-1]).max().item()
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=global_cu_seqlens,
                cu_seqlens_kv=global_cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
                qkv_format="thd"
            )
            output_tensor = model(
                input_ids=minibatch["states"],
                attention_mask=None,
                position_ids=None,
                labels=None,
                packed_seq_params=packed_seq_params
            )

            return output_tensor, partial(f, minibatch, cu_seqlens)

        forward_backward = get_forward_backward_func()
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size:
            data_iterator = [iter(minibatches) for _ in range(vpp_size)]
        else:
            data_iterator = iter(minibatches)
        output = forward_backward(
            model=self.model,
            data_iterator=data_iterator,
            num_microbatches=len(minibatches),
            forward_step_func=_forward_step,
            seq_length=1,
            micro_batch_size=1,
            forward_only=not torch.is_grad_enabled(),
            collect_non_loss_data=not torch.is_grad_enabled()
        )
        output = broadcast_object(
            output,
            mpu.get_pipeline_model_parallel_last_rank(),
            mpu.get_pipeline_model_parallel_group().group
        )
        if torch.is_grad_enabled():
            self._load_optimizer_to_device(torch.cuda.current_device())
            _, grad_norm, _ = self.optimizer.step()
            self.optimizer.zero_grad()
            self._load_optimizer_to_device("cpu")
            for model in self.model:
                model.zero_grad_buffer()
            self.scheduler.step(1)
            metrics = {
                k: [item for metric in output for item in metric[k]]
                for k in output[0].keys()
            }
            return metrics, grad_norm
        else:
            return output

    def _get_ckpt(self) -> Dict[str, Dict[str, Any]]:

        ckpt = {}
        for vpp_rank, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            key = f"model{vpp_rank}" if len(self.model) > 1 else "model"
            if hasattr(model, "module"):
                model = model.module
            ckpt[key] = model.sharded_state_dict()

        ckpt = {
            "optimizer": self.optimizer.sharded_state_dict(ckpt),
            "scheduler": self.scheduler.state_dict()
        }
        return ckpt

    def load_ckpt(self, save_dir: str):
        
        ckpt = self._get_ckpt()
        sharded_strategy = get_default_load_sharded_strategy(save_dir)
        sharded_strategy = FullyParallelLoadStrategyWrapper(
            sharded_strategy,
            mpu.get_data_parallel_group(with_context_parallel=True)
        )
        ckpt = dist_checkpointing.load(
            ckpt,
            save_dir,
            sharded_strategy=sharded_strategy
        )
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

    def save_ckpt(self, save_dir: str):

        self.save_model(f"{save_dir}/model")
        sharded_strategy = get_default_save_sharded_strategy("torch_dist")
        sharded_strategy = FullyParallelSaveStrategyWrapper(
            sharded_strategy,
            mpu.get_data_parallel_group(with_context_parallel=True)
        )
        os.makedirs(f"{save_dir}/optimizer_scheduler", exist_ok=True)
        dist_checkpointing.save(
            self._get_ckpt(),
            f"{save_dir}/optimizer_scheduler",
            sharded_strategy=sharded_strategy
        )

    def save_model(self, save_dir: str):

        self._load_model_to_gpu()
        self.bridge.save_hf_pretrained(self.model, save_dir)
        self._offload_model_to_cpu()
        if dist.get_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
        dist.barrier()