import functools
import torch
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaForTokenClassification,
    Qwen2ForCausalLM,
    Qwen2ForTokenClassification,
    Qwen3ForCausalLM,
    Qwen3ForTokenClassification
)

def prepare_llama_tp_layer(layer, device_mesh):

    parallelize_plan = {
        "input_layernorm": SequenceParallel(),
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(
            output_layouts=Shard(1)
        ),
        "post_attention_layernorm": SequenceParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(
            output_layouts=Shard(1)
        )
    }
    parallelize_module(
        module=layer,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_llama_tp_actor(model, device_mesh):

    for layer in model.model.layers:
        prepare_llama_tp_layer(layer, device_mesh)
        
    parallelize_plan = {
        "model.embed_tokens": ColwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel()
    }
    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_llama_tp_critic(model, device_mesh):

    for layer in model.model.layers:
        prepare_llama_tp_layer(layer, device_mesh)

    parallelize_plan = {
        "model.embed_tokens": ColwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "dropout": SequenceParallel(),
        "score": RowwiseParallel(
            input_layouts=Shard(1)
        )
    }
    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_tp_model(model, device_mesh):

    assert model.config.num_key_value_heads % device_mesh.size() == 0, \
        f"Key and value heads {model.config.num_key_value_heads} must be divisible by tensor parallelism size {device_mesh.size()}."

    if any([
        isinstance(model, cls)
        for cls in [
            LlamaForCausalLM,
            Qwen2ForCausalLM,
            Qwen3ForCausalLM
        ]
    ]):
        prepare_llama_tp_actor(model, device_mesh)
    elif any([
        isinstance(model, cls)
        for cls in [
            LlamaForTokenClassification,
            Qwen2ForTokenClassification,
            Qwen3ForTokenClassification
        ]
    ]):
        prepare_llama_tp_critic(model, device_mesh)
    else:
        raise NotImplementedError(
            f"Tensor parallelism is not supported for {model.__class__.__name__}."
        )
    
def prepare_dp_model(model, device_mesh):

    def get_module_cls_from_name(name):
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

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=mixed_precision,
        device_mesh=device_mesh,
        device_id=torch.cuda.current_device()
    )