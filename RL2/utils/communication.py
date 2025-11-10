from typing import Any, Optional, List
import os
import socket
from datetime import timedelta
import torch
import torch.distributed as dist

def get_host() -> str:

    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def get_available_port() -> int:

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def broadcast_object(
    obj: Optional[Any],
    src: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
    group_src: Optional[int] = None
) -> Any:
    object_list = [obj]
    dist.broadcast_object_list(
        object_list,
        src=src,
        group=group,
        group_src=group_src
    )
    return object_list[0]

def gather_and_concat_list(
    lst: List[Any], process_group: dist.ProcessGroup
) -> Optional[List[Any]]:

    lists = (
        dist.get_world_size(process_group) * [None]
        if dist.get_rank(process_group) == 0
        else None
    )
    dist.gather_object(
        lst,
        lists,
        group=process_group,
        group_dst=0
    )
    return (
        [item for lst in lists for item in lst]
        if dist.get_rank(process_group) == 0
        else None
    )