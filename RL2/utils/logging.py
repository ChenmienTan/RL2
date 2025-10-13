import time
import inspect
import functools
import torch.distributed as dist
from tqdm import tqdm
import wandb
from RL2.utils.communication import gather_and_concat_list

def progress_bar(*args, **kwargs):
    return tqdm(
        *args,
        position=1,
        leave=False,
        disable=(dist.get_rank() != 0),
        **kwargs
    )

def time_logger(name):
    def decorator(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        assert "step" in param_names
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            step = kwargs.get("step", args[param_names.index("step")])
            start = time.time()
            output = func(*args, **kwargs)
            if dist.get_rank() == 0:
                wandb.log({
                    f"timing/{name}": time.time() - start
                }, step=step)
            return output
        return wrapper
    return decorator

def gather_and_log(
    metrics,
    step,
    process_group=None,
    metrics_to_sum=["loss"]
):

    if process_group is not None:
        metrics = {
            k: gather_and_concat_list(v, process_group)
            for k, v in metrics.items()
        }

    if dist.get_rank() == 0:

        metrics = {
            k: sum(v) / (1.0 if k in metrics_to_sum else len(v))
            for k, v in metrics.items()
        }
        tqdm.write(f"Step {step}, " + ", ".join([
            f"{k}: {v:.3g}" for k, v in metrics.items()
        ]))
        wandb.log(metrics, step=step)

def gather_and_reduce(lst, process_group):

    lst = gather_and_concat_list(lst, process_group)
    if dist.get_rank() == 0:
        return sum(lst)

def rank0_log(metrics, step):
    
    if not dist.get_rank() == 0:
        return
    
    metrics = {
        k: sum(v) / len(v)
        for k, v in metrics.items()
    }
    tqdm.write(f"Step {step}, " + ", ".join([
        f"{k}: {v:.3g}" for k, v in metrics.items()
    ]))
    wandb.log(metrics, step=step)