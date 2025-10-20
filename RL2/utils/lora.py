from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf

def wrap_peft_model(model, config):
    if config.tp_size > 1:
        raise ValueError("Tensor parallelism is not supported with LoRA.")
    if hasattr(model, 'lm_head'):
        task_type = TaskType.CAUSAL_LM
    elif hasattr(model, 'score'):
        task_type = TaskType.SEQ_CLS
    else:
        raise ValueError("Model has no LM head or score attribute.")
    
    lora_config = OmegaConf.to_container(config)
    lora_config = LoraConfig(
        **config,
        task_type=task_type
    )
    return get_peft_model(model, lora_config)