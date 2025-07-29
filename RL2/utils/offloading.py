import torch
from torch.distributed.fsdp._runtime_utils import _lazy_init

def load_model_to_device(worker, device):
    
    if not getattr(worker.config, "offload_model", False):
        return

    _lazy_init(worker.model, worker.model)
    for handle in worker.model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(device, non_blocking=True)
        flat_param._local_shard = flat_param.data

def load_optimizer_to_device(worker, device):

    if not getattr(worker.config, "offload_optimizer", False):
        return

    for param_group in worker.optimizer.param_groups:
        for param in param_group["params"]:
            state = worker.optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(
                        device, non_blocking=True
                    )