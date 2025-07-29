import os
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict
)
import transformers

def save(worker, step=None, rm=False):

    if step is not None:
        if worker.config.save_freq is None or step % worker.config.save_freq != 0:
            return
        path = f"{worker.config.save_dir}/step{step}"
    else:
        path = f"{worker.config.save_dir}/latest"

    os.makedirs(path, exist_ok=True)

    options = StateDictOptions(
        full_state_dict=True, cpu_offload=True
    )
    state_dict = get_model_state_dict(
        worker.model, options=options
    )
    if dist.get_rank() == 0:

        worker.tokenizer.save_pretrained(f"{path}/model")
        # unwrap the model
        model_to_save = worker.model.module
        if rm:
            # For RM, we load token classification model for simplicity 
            # but save sequence classification model for compatibility.
            model_cls_name = model_to_save.__class__.__name__.replace(
                "Token", "Sequence"
            )
            model_cls = getattr(transformers, model_cls_name)
            with torch.device("meta"):
                model_to_save = model_cls._from_config(model_to_save.config)
        model_to_save.save_pretrained(
            f"{path}/model", state_dict=state_dict
        )
    dist.barrier()

    # TODO (P1): save optimizer state distributionally
    state_dict = get_optimizer_state_dict(
        worker.model, worker.optimizer, options=options
    )
    if dist.get_rank() == 0:
        os.makedirs(f"{path}/optimizer", exist_ok=True)
        torch.save(
            state_dict,
            f"{path}/optimizer/state_dict.pt"
        )
    dist.barrier()