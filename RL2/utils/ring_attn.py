from typing import Optional, Dict, Any
import os
import functools
import torch
import torch.distributed as dist
import transformers
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import (
    _flash_supports_window_size,
    is_flash_attn_greater_or_equal
)
from ring_flash_attn.zigzag_ring_flash_attn_varlen import zigzag_ring_flash_attn_varlen_func
from ring_flash_attn.adapters.hf_adapter import flash_attention_forward
from RL2.utils.sequences import position_ids_to_cu_seqlens

DATA_PARAMS: Dict[str, Any] = {}

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.Tensor] = None,
    cu_seq_lens_k: Optional[torch.Tensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs
):
    use_sliding_windows = (
        _flash_supports_window_size
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs = (
        {"window_size": (sliding_window, sliding_window)}
        if use_sliding_windows
        else {}
    )

    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = (
                os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
            )

    flash_kwargs["deterministic"] = deterministic
    flash_kwargs["group"] = DATA_PARAMS["group"]

    return zigzag_ring_flash_attn_varlen_func(
        query_states.squeeze(0), 
        key_states.squeeze(0),
        value_states.squeeze(0),
        cu_seqlens=DATA_PARAMS["cu_seqlens"],
        max_seqlen=DATA_PARAMS["max_seqlen"],
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=True,
        **flash_kwargs
    )

transformers.modeling_flash_attention_utils._flash_attention_forward = _flash_attention_forward
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def ring_attn_preprocess(raw_minibatch, device_mesh):

    cu_seqlens = position_ids_to_cu_seqlens(
        raw_minibatch["position_ids"]
    )
    rank = device_mesh.get_local_rank()
    double_size = 2 * device_mesh.size()
    minibatch = {}
    for k, v in raw_minibatch.items():
        tensors = []
        for start_idx, end_idx in zip(
            cu_seqlens[:-1], cu_seqlens[1:]
        ):
            half_seqlen = (end_idx - start_idx) // double_size
            tensor = v[start_idx:end_idx]
            tensor = torch.cat((
                tensor[rank * half_seqlen:(rank + 1) * half_seqlen],
                tensor[(double_size - rank - 1) * half_seqlen: (double_size - rank) * half_seqlen]
            ))
            tensors.append(tensor)
        minibatch[k] = torch.cat(tensors)
    cu_seqlens = cu_seqlens // device_mesh.size()
    return minibatch, cu_seqlens

def pad_tokens_and_unsqueeze(minibatch, cu_seqlens, multiple_of):

    if len(minibatch["states"]) % multiple_of != 0:
        pad_tokens = multiple_of - len(minibatch["states"]) % multiple_of
        minibatch = {
            k: torch.cat((
                v,
                torch.zeros(
                    (pad_tokens),
                    dtype=v.dtype,
                    device=v.device
                )
            ))
            for k, v in minibatch.items()
        }
        cu_seqlens = torch.cat((
            cu_seqlens,
            torch.tensor(
                minibatch["position_ids"].size(),
                dtype=torch.int32,
                device=cu_seqlens.device
            )
        ))
    else:
        pad_tokens = 0
        
    minibatch = {
        k: v.unsqueeze(0)
        for k, v in minibatch.items()
    }

    return minibatch, cu_seqlens, pad_tokens

def squeeze_and_remove_pad_tokens(output, pad_tokens):

    if isinstance(output, tuple):
        return tuple(
            squeeze_and_remove_pad_tokens(tensor, pad_tokens)
            for tensor in output
        )
    
    output = output.squeeze(0)
    return output[:-pad_tokens] if pad_tokens > 0 else output
        
def gather_tensor_along_sp_dim(tensor, device_mesh):

    tensors = [
        torch.zeros_like(tensor)
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        tensors,
        tensor,
        group=device_mesh.get_group()
    )
    tensors[device_mesh.get_local_rank()] = tensor # necessary to retain grad
    half_seqlen = len(tensor) // 2
    inorder_tensors = [
        tensor[:half_seqlen] for tensor in tensors
    ]
    reversed_tensors = [
        tensor[half_seqlen:] for tensor in tensors
    ]
    return torch.cat(inorder_tensors + reversed_tensors[::-1])

def ring_attn_postprocess(output, cu_seqlens, device_mesh):

    if isinstance(output, tuple):
        return tuple(
            ring_attn_postprocess(tensor, cu_seqlens, device_mesh)
            for tensor in output
        )

    return torch.cat([
        gather_tensor_along_sp_dim(
            output[start_idx:end_idx], device_mesh
        )
        for start_idx, end_idx in zip(
            cu_seqlens[:-1], cu_seqlens[1:]
        )
    ])

def ring_attn_manager(func):

    @functools.wraps(func)
    def forward_with_ring_attn(self, minibatch, *args, **kwargs):

        minibatch, cu_seqlens = ring_attn_preprocess(
            minibatch, self.device_mesh["sp"]
        )
        minibatch, cu_seqlens, pad_tokens = pad_tokens_and_unsqueeze(
            minibatch, cu_seqlens, self.device_mesh["tp"].size()
        )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        DATA_PARAMS.update({
            "group": self.device_mesh["sp"].get_group(),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen
        })

        output = func(self, minibatch, *args, **kwargs)

        output = squeeze_and_remove_pad_tokens(output, pad_tokens)
        return ring_attn_postprocess(
            output, cu_seqlens, self.device_mesh["sp"]
        )
    
    return forward_with_ring_attn