from typing import Dict, Union, Tuple
import torch
import torch.distributed as dist

def _differentiable_all_reduce(
    tensor: torch.Tensor, process_group: dist.ProcessGroup
) -> torch.Tensor:

    detached_tensor = tensor.detach()
    dist.all_reduce(
        detached_tensor,
        op=dist.ReduceOp.SUM,
        group=process_group
    )
    return tensor + detached_tensor - tensor.detach()

def _compute_logsumexp(
    logits: torch.Tensor,
    process_group: dist.ProcessGroup,
    chunk_size: int = 1024
) -> torch.Tensor:
    """
    When using tensor parallelism, each process only has a shard of logits.
    We firstly compute logsumexp of the sharded logits on each process,
    and then perform logsumexp across processes, which is equivalent to 
    performing logsumexp over the entire vocabulary.
    """
    # Direct logsumexp over the entire sequence suffers high memory peak.
    # See https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881.
    logsumexps = []
    for start in range(0, logits.shape[1], chunk_size):
        logsumexp = torch.logsumexp(
            logits[:, start:start + chunk_size], -1
        )
        logsumexps.append(logsumexp)
    logsumexp = torch.cat(logsumexps, -1)

    logsumexps = [
        torch.zeros_like(logsumexp)
        for _ in range(dist.get_world_size(process_group))
    ]
    dist.all_gather(
        logsumexps,
        logsumexp,
        group=process_group
    )
    logsumexps[dist.get_rank(process_group)] = logsumexp # necessary to retain grad
    logsumexps = torch.cat([
        logsumexp.unsqueeze(-1) for logsumexp in logsumexps
    ], -1)
    return torch.logsumexp(logsumexps, -1)

def gather_action_logits(
    logits: torch.Tensor,
    actions: torch.Tensor,
    process_group: dist.ProcessGroup
) -> torch.Tensor:
    """
    When using tensor parallelism, each process only has a shard of logits.
    On each process, we gather logits for actions on the process, and then 
    perform AllReduce to collect the complete logits.
    """
    rank = dist.get_rank(process_group)
    start = rank * logits.shape[-1]
    end = (rank + 1) * logits.shape[-1]

    local_mask = (actions >= start) & (actions < end)
    local_actions = torch.where(
        local_mask, actions - start, 0
    )

    action_logits = torch.where(
        local_mask,
        torch.gather(
            logits,
            dim=-1,
            index=local_actions.unsqueeze(-1)
        ).squeeze(-1),
        0.0
    )

    return _differentiable_all_reduce(action_logits, process_group)

def _compute_entropy(
    logits: torch.Tensor,
    logsumexp: torch.Tensor,
    process_group: dist.ProcessGroup
) -> torch.Tensor:

    probs = torch.exp(logits - logsumexp.unsqueeze(-1))
    return logsumexp - _differentiable_all_reduce(
        (probs * logits).sum(-1), process_group
    )

def compute_logps_and_entropy(
    logits: torch.Tensor,
    minibatch: Dict[str, torch.Tensor],
    process_group: dist.ProcessGroup,
    prefix: str = "",
    return_entropy: bool = False
):  
    logsumexp = _compute_logsumexp(logits, process_group)
    action_logits = gather_action_logits(
        logits,
        minibatch["actions"],
        process_group
    )
    minibatch[f"{prefix}logps"] = (
        action_logits - logsumexp
    ) * minibatch["action_mask"]

    if return_entropy:
        minibatch["entropy"] = _compute_entropy(
            logits, logsumexp, process_group
        ) * minibatch["action_mask"]

def aggregate_values(
    tensor: Union[torch.Tensor, Tuple[torch.Tensor]],
    action_mask: torch.Tensor,
    avg_level: str,
    total_actions: int,
    total_sequences: int
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    
    if isinstance(tensor, tuple):
        return tuple(
            aggregate_values(
                t,
                action_mask,
                avg_level,
                total_actions,
                total_sequences
            )
            for t in tensor
        )

    if avg_level == "token":
        return tensor.sum() / total_actions
    elif avg_level == "sequence":
        return (
            tensor.sum(-1) / (
                action_mask.sum(-1) + torch.finfo(tensor.dtype).eps
            )
        ).sum() / total_sequences
    else:
        raise NotImplementedError