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
        model_to_save = worker.model.module
        if rm:
            model_cls_name = model_to_save.__class__.__name__.replace(
                "Token", "Sequence"
            )
            model_cls = getattr(transformers, model_cls_name)
            with torch.device("meta"):
                model_to_save = model_cls._from_config(model_to_save.config)
        model_to_save.save_pretrained(
            path, state_dict=state_dict
        )

    dist.barrier()

    torch.save(
        worker.optimizer.state_dict(),
        f"{path}/optimizer_rank{dist.get_rank()}.pt"
    )