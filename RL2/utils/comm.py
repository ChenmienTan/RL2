import os
import math
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)

def split_and_scatter_list(lst, device_mesh=None):

    if device_mesh is None:
        world_size = dist.get_world_size()
        is_src = dist.get_rank() == 0
        src = 0
        group = None
        group_src = None
    else:
        world_size = device_mesh.size()
        is_src = device_mesh.get_local_rank() == 0
        src = None
        group = device_mesh.get_group()
        group_src = 0

    if is_src:
        data_per_dp = math.ceil(len(lst) / world_size)
    lists = [
        lst[rank * data_per_dp:(rank + 1) * data_per_dp]
        if is_src else None
        for rank in range(world_size)
    ]
    lst = [None]
    dist.scatter_object_list(
        lst,
        lists,
        src=src,
        group=group,
        group_src=group_src
    )
    return lst[0]

def gather_and_concat_list(lst, device_mesh=None):

    if device_mesh is None:
        world_size = dist.get_world_size()
        is_dst = dist.get_rank() == 0
        dst = 0
        group = None
        group_dst = None
    else:
        world_size = device_mesh.size()
        is_dst = device_mesh.get_local_rank() == 0
        dst = None
        group = device_mesh.get_group()
        group_dst = 0
    
    lists = [None] * world_size if is_dst else None
    dist.gather_object(
        lst,
        lists,
        dst=dst,
        group=group,
        group_dst=group_dst
    )

    return sum(lists, []) if is_dst else None