import torch

def offload_model_to_cpu(model):

    for param in model.parameters():
        param.data = param.data.to("cpu", non_blocking=True)

def load_model_to_gpu(model):
    
    for param in model.parameters():
        param.data = param.data.to(
            torch.cuda.current_device(), non_blocking=True
        )

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