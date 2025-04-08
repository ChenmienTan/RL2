import torch
import torch.nn as nn
from typing import Optional, Union
from transformers import AutoConfig, AutoModel, LlamaForCausalLM
from models.ring_attn_utils import convert_ring_attn_params, reset_position_ids
from utils import all_gather

# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=True,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, either "reward" or "critic".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        target_modules (list, optional): List of target modules for LoRA. Defaults to None.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.
        use_flash_attention_2 (bool, optional): Use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled. Defaults to None.
        init_value_head (bool, optional): Initialize the value head. Defaults to False.
        value_head_prefix (str, optional): Prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
        **kwargs,
    )

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model

def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="score"):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            # convert attention_mask to position_ids
            if ring_attn_group is not None:
                input_ids, position_ids = convert_ring_attn_params(
                    input_ids, position_ids, ring_attn_group
                )

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)
            if ring_attn_group is not None and values_allgather:
                values = all_gather(values, ring_attn_group).reshape(values.shape[0], -1)[:, :-1]
            else:
                values = values[:, :-1]
            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if num_actions is None:
                assert return_output
                return outputs

            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_values = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_values.append(values[:, start:end])
                offset += seq_len
            action_values = torch.cat(action_values, dim=1)

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel