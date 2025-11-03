from typing import Type
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def _param_init_fn(module: nn.Module):
    module.to_empty(device=torch.cuda.current_device(), recurse=False)

def prepare_dp_model(
    model: nn.Module,
    dtype: str,
    sync_module_states: bool,
    device_mesh: dist.DeviceMesh
) -> FSDP:

    def get_module_cls_from_name(name: str) -> Type[nn.Module]:
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__

    transformer_layer_cls = {
        get_module_cls_from_name(name)
        for name in model._no_split_modules
    }
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls
    )

    dtype: torch.dtype = getattr(torch, dtype)
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=mixed_precision,
        param_init_fn=_param_init_fn,
        sync_module_states=sync_module_states,
        device_mesh=device_mesh,
        device_id=torch.cuda.current_device()
    )