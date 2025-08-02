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

def sequence_all_reduce(tensor, cu_seqlens, device_mesh):

    tensor = torch.stack([
        tensor[:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    return differentiable_all_reduce(tensor, device_mesh)

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

    local_vocab_size = torch.LongTensor(
        [logits.shape[-1]]
    ).to(torch.cuda.current_device())
    vocab_sizes = [
        torch.zeros_like(local_vocab_size)
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        vocab_sizes,
        local_vocab_size,
        group=device_mesh.get_group()
    )
    cu_vocab_sizes = torch.cumsum(
        torch.cat(
            [torch.zeros_like(local_vocab_size)] + vocab_sizes
        ), 0
    )
    action_device_mapping = (
        actions < cu_vocab_sizes[1:].unsqueeze(-1)
    ).to(torch.float32).argmax(0)
    local_action_indices = torch.where(
        action_device_mapping == rank
    )[0]
    local_actions = actions[:, local_action_indices] - cu_vocab_sizes[rank]
    action_logits = torch.zeros(
        actions.shape, device=torch.cuda.current_device()
    )
    action_logits[:, local_action_indices] = torch.gather(
        logits[:, local_action_indices],
        dim=-1,
        index=local_actions.unsqueeze(-1)
    ).squeeze(-1)

    return differentiable_all_reduce(action_logits, device_mesh)

def compute_entropy(logits, logsumexp, device_mesh):

    probs = torch.exp(logits - logsumexp.unsqueeze(-1))
    return logsumexp - differentiable_all_reduce(
        (probs * logits).sum(-1), device_mesh
    )