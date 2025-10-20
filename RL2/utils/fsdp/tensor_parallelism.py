import torch
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
    Qwen3ForTokenClassification,
    Qwen3MoeForCausalLM
)

def to_empty(module):
    module.to_empty(device=torch.cuda.current_device())

def sync_module_states(module, attr, device_mesh):

    module.to(torch.cuda.current_device())
    dist.broadcast(
        getattr(module, attr),
        group=device_mesh.get_group(),
        group_src=0
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

    if device_mesh.get_local_rank() != 0:
        to_empty(layer)
    sync_module_states(layer.input_layernorm, "weight", device_mesh)
    sync_module_states(layer.post_attention_layernorm, "weight", device_mesh)

    parallelize_module(
        module=layer,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_qwen3_moe_tp_layer(layer, device_mesh, num_experts):
    """
    Prepare Qwen3 MoE layer for tensor parallelism.

    Architecture:
    - Standard attention (same as dense models)
    - MoE MLP with:
      - gate: Linear(in_features, num_experts) - router, kept replicated
      - experts: ModuleList of num_experts MLPs, each parallelized
    """
    # Build parallelize plan for attention (same as dense models)
    parallelize_plan = {
        "input_layernorm": SequenceParallel(),
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(
            output_layouts=Shard(1)
        ),
        "self_attn.q_norm": SequenceParallel(),
        "self_attn.k_norm": SequenceParallel(),
        "post_attention_layernorm": SequenceParallel(),
    }

    # Add MoE-specific parallelization
    # The gate (router) stays replicated - each rank needs full routing decisions
    # Each expert's projections are column/row parallelized
    for expert_idx in range(num_experts):
        parallelize_plan[f"mlp.experts.{expert_idx}.gate_proj"] = ColwiseParallel()
        parallelize_plan[f"mlp.experts.{expert_idx}.up_proj"] = ColwiseParallel()
        parallelize_plan[f"mlp.experts.{expert_idx}.down_proj"] = RowwiseParallel(
            output_layouts=Shard(1)
        )

    if device_mesh.get_local_rank() != 0:
        to_empty(layer)
    sync_module_states(layer.input_layernorm, "weight", device_mesh)
    sync_module_states(layer.post_attention_layernorm, "weight", device_mesh)

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

    if device_mesh.get_local_rank() != 0:
        to_empty(model.model.embed_tokens)
        to_empty(model.model.rotary_emb)
        to_empty(model.model.norm)
        to_empty(model.lm_head)
    sync_module_states(model.model.norm, "weight", device_mesh)
    sync_module_states(model.model.rotary_emb, "inv_freq", device_mesh)

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

    if device_mesh.get_local_rank() != 0:
        to_empty(model.model.embed_tokens)
        to_empty(model.model.rotary_emb)
        to_empty(model.model.norm)
        to_empty(model.dropout)
        to_empty(model.score)
    sync_module_states(model.model.norm, "weight", device_mesh)
    sync_module_states(model.model.rotary_emb, "inv_freq", device_mesh)

    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_qwen3_moe_tp_actor(model, device_mesh):
    """
    Prepare Qwen3 MoE model for tensor parallelism.
    """
    num_experts = model.config.num_experts

    for layer in model.model.layers:
        prepare_qwen3_moe_tp_layer(layer, device_mesh, num_experts)

    parallelize_plan = {
        "model.embed_tokens": ColwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel()
    }

    if device_mesh.get_local_rank() != 0:
        to_empty(model.model.embed_tokens)
        to_empty(model.model.rotary_emb)
        to_empty(model.model.norm)
        to_empty(model.lm_head)
    sync_module_states(model.model.norm, "weight", device_mesh)
    sync_module_states(model.model.rotary_emb, "inv_freq", device_mesh)

    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_tp_model(model, device_mesh):

    assert model.config.num_key_value_heads % device_mesh.size() == 0, \
        f"Key and value heads {model.config.num_key_value_heads} must be divisible by tensor parallelism size {device_mesh.size()}."

    if isinstance(model, Qwen3MoeForCausalLM):
        prepare_qwen3_moe_tp_actor(model, device_mesh)
    elif any([
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