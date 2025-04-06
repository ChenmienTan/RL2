import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import inspect
from typing import Optional, Tuple
import transformers
import transformers.modeling_flash_attention_utils
from transformers.modeling_flash_attention_utils import (
    _flash_supports_window_size,
    is_flash_attn_greater_or_equal,
)
from ring_flash_attn import (
    llama3_flash_attn_varlen_func,
    llama3_flash_attn_prepare_cu_seqlens,
    zigzag_ring_flash_attn_varlen_func
)

RING_ATTN_GROUP = None


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except:
    ALL_ATTENTION_FUNCTIONS = None


DATA_PARAMS = {}
RING_ATTN_SWITCH = True


def check_params(f1, f2):
    return len(inspect.signature(f1).parameters) == len(
        inspect.signature(f2).parameters
    )


def use_ring_attn(flag):
    global RING_ATTN_SWITCH
    RING_ATTN_SWITCH = flag


def create_ring_flash_attention_forward(
    process_group: dist.ProcessGroup, heads_k_stride: int
):
    # before transformers 4.47
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
        deterministic: bool = None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_top_left_mask (`bool`, defaults to `False`):
                flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
            softcap (`float`, *optional*):
                Softcap for the attention logits, used e.g. in gemma2.
            deterministic (`bool`, *optional*):
                Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        """
        if not use_top_left_mask:
            causal = is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
            causal = is_causal and query_length != 1

        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
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
        assert (
            softcap is None
        ), "llama3_flash_attn_varlen_func does not support softcap yet."
        # flash_kwargs["softcap"] = softcap
        flash_kwargs["group"] = process_group

        # not sure why attention_mask can be not None...
        assert causal, "only causal attention is supported yet."
        batch_size = query_states.size(0)
        assert batch_size == 1, "varlen data should be processed in advance."

        attn_output = zigzag_ring_flash_attn_varlen_func(
            query_states.squeeze(dim=0),
            key_states.squeeze(dim=0),
            value_states.squeeze(dim=0),
            cu_seqlens=DATA_PARAMS["cu_seqlens"], # TODO: Not yet processed
            max_seqlen=DATA_PARAMS["max_seqlen"], # TODO: Not yet processed
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.unsqueeze(dim=0)

        return attn_output

    # transformers 4.47
    def _flash_attention_forward_v1(
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
        deterministic: bool = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
    ):
        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout,
            position_ids,
            softmax_scale,
            sliding_window,
            use_top_left_mask,
            softcap,
            deterministic,
        )

    def _flash_attention_forward_v2(
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
        deterministic: bool = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout,
            position_ids,
            softmax_scale,
            sliding_window,
            use_top_left_mask,
            softcap,
            deterministic,
        )

    return [
        _flash_attention_forward,
        _flash_attention_forward_v1,
        _flash_attention_forward_v2,
    ]


_use_top_left_mask = False


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(
                layer
                for layer in module.modules()
                if isinstance(layer, torch.nn.Linear)
            ).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = transformers.modeling_flash_attention_utils._flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None


def substitute_hf_flash_attn(process_group: dist.ProcessGroup, heads_k_stride: int):
    try:
        # substitute flash attn
        old_flash_attention_forward = (
            transformers.modeling_flash_attention_utils._flash_attention_forward
        )
        new_flash_attention_forward_list = create_ring_flash_attention_forward(
            process_group, heads_k_stride
        )
        for new_flash_attention_forward in new_flash_attention_forward_list:
            if check_params(old_flash_attention_forward, new_flash_attention_forward):
                transformers.modeling_flash_attention_utils._flash_attention_forward = (
                    lambda *args, **kwargs: (
                        new_flash_attention_forward(*args, **kwargs)
                        if RING_ATTN_SWITCH
                        else old_flash_attention_forward(*args, **kwargs)
                    )
                )
                break
        else:
            assert (
                False
            ), "The signature of the new flash attention forward function does not match the old one."
    except:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please use pip install -U transformers to upgrade to the latest version. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/zhuzilin/ring-flash-attention/issues"
        )

    if ALL_ATTENTION_FUNCTIONS is not None:
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward


def reset_ring_attn_position_ids(start, end, packed_seq_lens):
    """
    Calculate position ids for packed_seq_ids[start:end].
    For example, if the packed_seq_lens is [3, 2, 4, 1], start=2, end=8,
    the position ids will be [2, 0, 1, 0, 1, 2].

    Args:
        start: the start position
        end: the end position
        packed_seq_lens: the sequence lengths of packed sequences
    """
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start : seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids


def update_ring_attn_params(packed_seq_lens, total_seq_len):
    """
    Calculate the cu_seqlens for the current forward pass and pass the value to
    the substituted ring_flash_attn.

    Note that total_seq_len may be larger than the sum of packed_seq_lens because of padding.
    """
    assert RING_ATTN_GROUP is not None
    cu_seqlens = torch.cumsum(
        torch.tensor(packed_seq_lens, device=torch.cuda.current_device(), dtype=torch.int32),
        dim=-1,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len)
    DATA_PARAMS["cu_seqlens"] = cu_seqlens
    DATA_PARAMS["max_seqlen"] = total_seq_len

def convert_ring_attn_params(sequences, attention_mask, packed_seq_lens, ring_attn_group):
    # each rank within the ring group will process sequences[start:end]
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    total_seq_len = sequences.numel()
    local_seq_len = total_seq_len // ring_attn_size
    start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
    sequences = sequences[:, start:end]
    attention_mask = attention_mask[:, start:end]
    position_ids = reset_ring_attn_position_ids(start, end, packed_seq_lens)
    update_ring_attn_params(packed_seq_lens, total_seq_len)
    return sequences, attention_mask, position_ids


def pad_sequences(sequences, attention_mask, num_actions, packed_seq_lens, ring_attn_group, pad_token_id=0):
    # Pads the input sequences and attention mask to ensure that their lengths are multiples of the ring attention size.
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    if isinstance(sequences, torch.Tensor):
        seqlen = sequences.shape[-1]
        pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
        padded = torch.tensor([pad_token_id] * pad_len, device=sequences.device, dtype=sequences.dtype).unsqueeze(0)
        sequences = torch.cat([sequences, padded], dim=1)
        max_seq_id = attention_mask.max().item()
        attention_mask = torch.cat(
            [attention_mask, (max_seq_id + 1) * torch.ones(1, pad_len, device="cuda", dtype=torch.float)], dim=-1
        )
    elif isinstance(sequences, list):
        seqlen = len(sequences)
        pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
        sequences += [pad_token_id] * pad_len
        attention_mask += [len(packed_seq_lens) + 1] * pad_len
    else:
        raise "sequences is not available type"
    num_actions[-1] += pad_len
    packed_seq_lens[-1] += pad_len
    return pad_len, sequences, attention_mask, num_actions, packed_seq_lens


def unpad_sequences(
    pad_len,
    sequences,
    attention_mask,
    num_actions,
    packed_seq_lens,
    ring_attn_group,
    action_log_probs=None,
    values=None,
    kl=None,
):
    # Removes the padding from the input sequences, attention mask, and other optional tensors after padding.
    if pad_len > 0:
        sequences = sequences[:, :-pad_len]
        attention_mask = attention_mask[:, :-pad_len]
        num_actions[-1] -= pad_len
        packed_seq_lens[-1] -= pad_len
        if action_log_probs is not None:
            action_log_probs = action_log_probs[:, :-pad_len]
        if values is not None:
            values = values[:, :-pad_len]
        if kl is not None:
            kl = kl[:, :-pad_len]
    return sequences, attention_mask, num_actions, packed_seq_lens, action_log_probs, values, kl