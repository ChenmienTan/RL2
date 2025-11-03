import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)
from transformers import (
    LlamaForCausalLM,
    LlamaForTokenClassification,
    Qwen2ForCausalLM,
    Qwen2ForTokenClassification,
    Qwen3ForCausalLM,
    Qwen3ForTokenClassification
)

def _to_empty(module: nn.Module):
    module.to_empty(device=torch.cuda.current_device())

def _sync_module_states(
    module: nn.Module, attr: str, device_mesh: dist.DeviceMesh
):

    module.to(torch.cuda.current_device())
    dist.broadcast(
        getattr(module, attr),
        group=device_mesh.get_group(),
        group_src=0
    )

def _prepare_llama_tp_layer(layer: nn.Module, device_mesh: dist.DeviceMesh):

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

    if device_mesh.get_local_rank() != 0:
        _to_empty(layer)
    _sync_module_states(layer.input_layernorm, "weight", device_mesh)
    _sync_module_states(layer.post_attention_layernorm, "weight", device_mesh)

    parallelize_module(
        module=layer,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def _prepare_llama_tp_actor(model: nn.Module, device_mesh: dist.DeviceMesh):

    for layer in model.model.layers:
        _prepare_llama_tp_layer(layer, device_mesh)
        
    parallelize_plan = {
        "model.embed_tokens": ColwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel()
    }

    if device_mesh.get_local_rank() != 0:
        _to_empty(model.model.embed_tokens)
        _to_empty(model.model.rotary_emb)
        _to_empty(model.model.norm)
        _to_empty(model.lm_head)
    _sync_module_states(model.model.norm, "weight", device_mesh)
    _sync_module_states(model.model.rotary_emb, "inv_freq", device_mesh)

    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def _prepare_llama_tp_critic(model: nn.Module, device_mesh: dist.DeviceMesh):

    for layer in model.model.layers:
        _prepare_llama_tp_layer(layer, device_mesh)

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

    if device_mesh.get_local_rank() != 0:
        _to_empty(model.model.embed_tokens)
        _to_empty(model.model.rotary_emb)
        _to_empty(model.model.norm)
        _to_empty(model.dropout)
        _to_empty(model.score)
    _sync_module_states(model.model.norm, "weight", device_mesh)
    _sync_module_states(model.model.rotary_emb, "inv_freq", device_mesh)

    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_tp_model(model: nn.Module, device_mesh: dist.DeviceMesh):

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
        _prepare_llama_tp_actor(model, device_mesh)
    elif any([
        isinstance(model, cls)
        for cls in [
            LlamaForTokenClassification,
            Qwen2ForTokenClassification,
            Qwen3ForTokenClassification
        ]
    ]):
        _prepare_llama_tp_critic(model, device_mesh)
    else:
        raise NotImplementedError(
            f"Tensor parallelism is not supported for {model.__class__.__name__}."
        )