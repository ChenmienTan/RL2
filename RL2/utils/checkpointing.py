import os
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict
)
import transformers

def save_model_and_optimizer(worker, step=None, rm=False):

    path = worker.config.save_dir + (
        f"/step{step}" if step is not None else "/latest"
    )
    os.makedirs(path, exist_ok=True)
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=True
    )
    state_dict = get_model_state_dict(
        worker.model, options=options
    )
    if dist.get_rank() == 0:

        worker.tokenizer.save_pretrained(path)
        # We save model in half precision to save time.
        state_dict = {
            k: v.to(torch.bfloat16) for k, v in state_dict.items()
        }
        if hasattr(worker.config, "lora") and worker.config.lora.rank > 0:
            model_to_save = worker.model
        else:
            model_cls_name = worker.model.__class__.__name__.removeprefix("FSDP")
            if rm:
                model_cls_name = model_cls_name.replace(
                    "Token", "Sequence"
                )
            model_cls = getattr(transformers, model_cls_name)
            with torch.device("meta"):
                model_to_save = model_cls._from_config(
                    worker.model.config
                )
        model_to_save.save_pretrained(
            path, state_dict=state_dict
        )

    dist.barrier()

    torch.save(
        worker.optimizer.state_dict(),
        f"{path}/optimizer_rank{dist.get_rank()}.pt"
    )