import torch
import torch.nn as nn
from typing import Optional, Union
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, LlamaForCausalLM
from models.ring_attn_utils import convert_ring_attn_params
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
import torch.distributed as dist
from models.ring_attn_utils import all_gather_with_grad

def _get_actor_model(base_llm_model):
    class ActorModel(base_llm_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # No need to set base_model_prefix or create a new instance

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            use_cache=False,
            **kwargs
        ) -> torch.Tensor:
            if ring_attn_group is not None:
                input_ids, position_ids, total_seq_len = convert_ring_attn_params(
                    input_ids, position_ids, ring_attn_group
                )

            # Call parent class forward directly
            outputs = super().forward(
                input_ids, 
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                **kwargs
            )
            
            # Handle sequence parallel gathering if needed
            if ring_attn_group is not None:
                # Get local logits
                local_logits = outputs.logits
                
                # Get world size for the ring attention group
                world_size = dist.get_world_size(ring_attn_group)
                
                # Calculate expected sequence lengths for each rank
                base_len = total_seq_len // world_size
                remainder = total_seq_len % world_size
                
                # Use all_gather with gradient support
                gathered_logits = all_gather_with_grad(local_logits, ring_attn_group, 
                                                       [base_len + (remainder if r == world_size-1 else 0) 
                                                        for r in range(world_size)])
                
                # Concatenate the gathered logits along the sequence dimension (dim=1)
                outputs.logits = torch.cat(gathered_logits, dim=1)
            
            return outputs
            
    return ActorModel

def get_actor_model(model_name_or_path, **kwargs):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    base_class = AutoModelForCausalLM._model_mapping[type(config)]
    
    # Just use the base_class directly
    cls_class = _get_actor_model(base_class)
    
    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        **kwargs
    )
    
    return model 
