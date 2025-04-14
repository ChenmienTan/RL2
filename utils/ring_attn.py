from typing import Optional, Dict
import os
import torch
from transformers.modeling_flash_attention_utils import (
    # _flash_supports_window_size, # TODO: update this
    is_flash_attn_greater_or_equal
)
import transformers
from ring_flash_attn.zigzag_ring_flash_attn_varlen import zigzag_ring_flash_attn_varlen_func
from ring_flash_attn.adapters.hf_adapter import flash_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    
def get_ring_flash_attn(process_group, cu_seqlens, max_seqlen):

    def _flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float=0.0,
        position_ids: Optional[torch.Tensor]=None,
        softmax_scale: Optional[float]=None,
        sliding_window: Optional[int]=None,
        use_top_left_mask: bool=False,
        softcap: Optional[float]=None,
        deterministic: Optional[bool]=None,
        cu_seq_lens_q: Optional[torch.Tensor]=None,
        cu_seq_lens_k: Optional[torch.Tensor]=None,
        max_length_q: Optional[int]=None,
        max_length_k: Optional[int]=None,
        target_dtype: Optional[torch.dtype]=None,
        **kwargs
    ):
        
        # use_sliding_windows = (
        #     _flash_supports_window_size
        #     and sliding_window is not None
        #     and key_states.shape[1] > sliding_window
        # )
        # flash_kwargs = (
        #     {"window_size": (sliding_window, sliding_window)}
        #     if use_sliding_windows
        #     else {}
        # )
        flash_kwargs = {}

        if is_flash_attn_greater_or_equal("2.4.1"):
            if deterministic is None:
                deterministic = (
                    os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
                )

        flash_kwargs["deterministic"] = deterministic
        flash_kwargs["group"] = process_group

        return zigzag_ring_flash_attn_varlen_func(
            query_states.squeeze(0), 
            key_states.squeeze(0),
            value_states.squeeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=True,
            **flash_kwargs
        )
    
    return _flash_attention_forward


class RingAttentionContext:

    def __init__(self, device_mesh, seqlens: torch.Tensor):

        self.device_mesh = device_mesh
        self.cu_seqlens = torch.cumsum(
        torch.cat((
            torch.IntTensor([0]).to(torch.cuda.current_device()),
            seqlens
        )), 0)
        self.max_seqlen = seqlens.max().item()

    def __enter__(self):
        
        self.original_flash_attn = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        transformers.modeling_flash_attention_utils._flash_attention_forward = get_ring_flash_attn(
            self.device_mesh.get_group(),
            self.cu_seqlens,
            self.max_seqlen
        )
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

    def __exit__(self):

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = self.original_flash_attn