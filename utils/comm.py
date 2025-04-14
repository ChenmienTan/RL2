import os
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)

def sum_across_processes(value):

    value = torch.Tensor([value]).to(torch.cuda.current_device())
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.to("cpu").item()

def gather_and_concat_list(lst, device_mesh):

    lists = [None for _ in range(device_mesh.size())]
    dist.gather_object(
        lst,
        lists if device_mesh.get_rank() == 0 else None,
        dst=0
    )
    return sum(lists, []) if device_mesh.get_rank() == 0 else None