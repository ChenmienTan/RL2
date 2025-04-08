import torch
import torch.nn as nn
from typing import Optional, Union
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from models.ring_attn_utils import convert_ring_attn_params
import inspect
from transformers.modeling_outputs import CausalLMOutputWithPast

def _get_actor_model(base_pretrained_model, base_llm_model):
    class ActorModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # Create the actual model instance
            # self.model = base_llm_model(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

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
                input_ids, position_ids = convert_ring_attn_params(
                    input_ids, position_ids, ring_attn_group
                )

            outputs = getattr(self, self.base_model_prefix).forward(
                input_ids, 
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs
            )
            
            # If the model has a lm_head, apply it to get logits
            if hasattr(self, 'lm_head'):
                logits = self.lm_head(outputs.last_hidden_state)
            else:
                logits = outputs.last_hidden_state
                
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            )
            
    return ActorModel

def get_actor_model(model_name_or_path, **kwargs):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # config.attn_implementation = "flash_attention_2"
    
    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    print(f"base_pretrained_class: {base_pretrained_class}")
    print(f"base_class: {base_class}")
    # The order matters here - you want to inherit from AutoModelForCausalLM
    # and use AutoModel as the base model implementation
    cls_class = _get_actor_model(base_pretrained_class, base_class)
    
    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        **kwargs
    )
    
    # Print the forward function implementation
    print(f"Forward function implementation:")
    print(inspect.getsource(model.forward))
    
    return model 
