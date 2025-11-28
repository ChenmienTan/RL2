from typing import Any, Optional, List, Dict
import os
import socket
import asyncio
import aiohttp
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

def _unwrap_process_group(
    process_group: dist.ProcessGroup
) -> dist.ProcessGroup:

    if hasattr(process_group, "group"):
        return process_group.group
    elif hasattr(process_group, "get_group"):
        return process_group.get_group()
    else:
        return process_group

def broadcast_object(
    obj: Optional[Any],
    src: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    group_src: Optional[int] = None
) -> Any:

    object_list = [obj]
    dist.broadcast_object_list(
        object_list,
        src=src,
        group=_unwrap_process_group(process_group),
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
        group=_unwrap_process_group(process_group),
        group_dst=0
    )
    return (
        [item for lst in lists for item in lst]
        if dist.get_rank(process_group) == 0
        else None
    )

async def async_request(
    url: str,
    method: str = "POST",
    payload: Dict[str, Any] = {},
    headers: None | Dict[str, Any] = None,
    timeout: int = 300,
    max_trials: int = 3,
    retry_delay: int = 1
) -> Dict[str, Any] | str | None:
    
    async with aiohttp.ClientSession() as session:
        for trial in range(max_trials):
            try:
                kwargs = {
                    "url": url,
                    "headers": headers,
                    "timeout": aiohttp.ClientTimeout(total=timeout)
                }
                if method == "POST":
                    req_ctx = session.post(json=payload, **kwargs)
                elif method == "GET":
                    req_ctx = session.get(**kwargs)
                else:
                    raise NotImplementedError
                
                async with req_ctx as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    if 'application/json' in content_type:
                        return await response.json(content_type=None)
                    else:
                        return await response.text()
            except Exception:
                if trial == max_trials - 1:
                    raise
                await asyncio.sleep(retry_delay)