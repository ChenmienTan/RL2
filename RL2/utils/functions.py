import torch
import torch.distributed as dist

def differentiable_all_reduce(tensor, device_mesh):

    detached_tensor = tensor.detach()
    dist.all_reduce(
        detached_tensor,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return tensor + detached_tensor - tensor.detach()

def compute_logsumexp(logits, device_mesh, chunk_size=1024):

    # When using tensor parallelism, each device only has a shard of logits.
    # We firstly compute logsumexp of the sharded logits on each device,
    # and then perform logsumexp across devices, which is equivalent to 
    # performing logsumexp over the entire vocabulary.

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
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        logsumexps,
        logsumexp,
        group=device_mesh.get_group()
    )
    logsumexps[device_mesh.get_local_rank()] = logsumexp # necessary to retain grad
    logsumexps = torch.cat([
        logsumexp.unsqueeze(-1) for logsumexp in logsumexps
    ], -1)
    return torch.logsumexp(logsumexps, -1)

def gather_action_logits(logits, actions, device_mesh):

    # When using tensor parallelism, each device only has a shard of logits.
    # On each device, we gather logits for actions on the device, and then 
    # perform AllReduce to collect the complete logits.
    rank = device_mesh.get_local_rank()
    start_idx = rank * logits.shape[-1]
    end_idx = (rank + 1) * logits.shape[-1]

    local_mask = (actions >= start_idx) & (actions < end_idx)
    local_actions = torch.where(
        local_mask, actions - start_idx, 0
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

    return differentiable_all_reduce(action_logits, device_mesh)

def compute_entropy(logits, logsumexp, device_mesh):

    probs = torch.exp(logits - logsumexp.unsqueeze(-1))
    return logsumexp - differentiable_all_reduce(
        (probs * logits).sum(-1), device_mesh
    )

def aggregate_values(
    tensor,
    action_mask,
    avg_level,
    total_actions,
    total_sequences
):
    
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