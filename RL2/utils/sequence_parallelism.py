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
from ring_flash_attn.llama3_flash_attn_varlen import (
    llama3_flash_attn_varlen_func,
    llama3_flash_attn_prepare_cu_seqlens
)
from ring_flash_attn.adapters.hf_adapter import flash_attention_forward

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

    return llama3_flash_attn_varlen_func(
        query_states.squeeze(0),
        key_states.squeeze(0),
        value_states.squeeze(0),
        cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
        cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
        max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
        max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
        heads_k_stride=1,
        local_k_slice=DATA_PARAMS["local_k_slice"],
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=True,
        **flash_kwargs
    )

transformers.modeling_flash_attention_utils._flash_attention_forward = _flash_attention_forward
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def sequence_parallelism_manager(func):

    @functools.wraps(func)
    def forward_with_sequence_parallelism(
        worker, minibatch, *args, **kwargs
    ):
        shape = minibatch["states"].shape
        seq_lens = minibatch["eos_mask"].argmax(-1) + 1
        minibatch = {
            k: torch.cat([
                seq[:seq_len] for seq, seq_len in zip(v, seq_lens)
            ]).unsqueeze(0)
            for k, v in minibatch.items()
        }

        multiple_of = worker.device_mesh["sp"].size() * worker.device_mesh["tp"].size()
        if sum(seq_lens) % multiple_of != 0:
            pad_tokens = multiple_of - sum(seq_lens) % multiple_of
            seq_lens = torch.cat((
                seq_lens,
                torch.LongTensor([pad_tokens]).to(torch.cuda.current_device())
            ))
            minibatch = {
                k: torch.cat((
                    v,
                    torch.zeros((1, pad_tokens), dtype=v.dtype, device=v.device)
                ), -1)
                for k, v in minibatch.items()
            }

        cu_seqlens = torch.cumsum(
            torch.cat((
                torch.LongTensor([0]).to(torch.cuda.current_device()),
                seq_lens
            )),
            dim=0,
            dtype=torch.int32
        )
        rank = worker.device_mesh["sp"].get_local_rank()
        world_size = worker.device_mesh["sp"].size()
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens,
            True,
            rank,
            world_size
        )
        DATA_PARAMS.update({
            "group": worker.device_mesh["sp"].get_group(),
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "local_k_slice": local_k_slice,
        })
        
        minibatch = {
            k: torch.chunk(v, world_size, dim=-1)[rank]
            for k, v in minibatch.items()
        }
        output = func(worker, minibatch, *args, **kwargs)

        def postprocess(output):

            if isinstance(output, tuple):
                return tuple(
                    postprocess(tensor)
                    for tensor in output
                )
            
            tensors = [
                torch.zeros_like(output)
                for _ in range(world_size)
            ]
            dist.all_gather(
                tensors,
                output,
                group=worker.device_mesh["sp"].get_group()
            )
            tensors[rank] = output # necessary to retain grad
            tensor = torch.cat(tensors, -1).squeeze(0)

            output = torch.zeros(shape, device=torch.cuda.current_device())
            for row, start_idx, end_idx in zip(
                range(shape[0]), cu_seqlens[:-1], cu_seqlens[1:]
            ):
                output[row, :end_idx - start_idx] = tensor[start_idx:end_idx]

            return output

        return postprocess(output)
    
    return forward_with_sequence_parallelism