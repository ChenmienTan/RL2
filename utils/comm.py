import os
import math
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

def shard_across_processes(item, device_mesh):

    rank = device_mesh.get_local_rank()
    bsz = len(item)
    bsz_per_process = math.ceil(bsz / device_mesh.size())
    return item[rank * bsz_per_process:(rank + 1) * bsz_per_process]

def concat_across_processes(item, device_mesh):

    all_lists = [None for _ in range(device_mesh.size())]
    dist.all_gather_object(all_lists, item, group=device_mesh.get_group())
    return sum(all_lists, [])

def sum_across_processes(value, device_mesh=None):

    value = torch.Tensor([value]).to(torch.cuda.current_device())
    dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
        if device_mesh is not None else None
    )
    return value.to("cpu").item()