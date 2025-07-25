import torch
from torch.distributed.fsdp._runtime_utils import _lazy_init

def offload_model_to_cpu(model):

    _lazy_init(model, model)
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        flat_param._local_shard = flat_param.data

def load_model_to_gpu(model):
    
    _lazy_init(model, model)
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.cuda.current_device(), non_blocking=True)
        flat_param._local_shard = flat_param.data

def offload_optimizer_to_cpu(optimizer):

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)

def load_optimizer_to_gpu(optimizer):

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(
                        torch.cuda.current_device(), non_blocking=True
                    )