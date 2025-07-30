from typing import List
import math
import torch
import torch.distributed as dist
from RL2.utils.comm import split_and_scatter_list, gather_and_concat_list
from RL2.utils.seqlen_balance import get_seqlen_balanced_partitions

PAD_TRAJECTORIES = 0
SHUFFLE_INDICES = None

def group_data_list_into_all_rank_data_lists(worker, data_list, pair=False):
    
    # We use ZigZag Ring Attention to partition sequences, where 
    # the length of each sequence needs to be multiple of 2 * 
    # sp size and each rank sequentially get the head and tail.
    # See https://zhuanlan.zhihu.com/p/683714620.
    multiple_of = 2 * worker.device_mesh["sp"].size()
    for ex in data_list:
        if len(ex["states"]) % multiple_of == 0:
            continue
        pad_tokens = multiple_of - len(ex["states"]) % multiple_of
        for k, v in ex.items():
            ex[k] = torch.cat(
                (v, torch.zeros((pad_tokens), dtype=v.dtype))
            )

    # We pack trajectories into minibatches for higher throughput.
    # To accommodate all trajectories, at least n_minibatches 
    # minibatches are needed.
    seq_len_list = [len(ex["states"]) for ex in data_list]
    if pair:
        # When pair, every two adjacent trajectories will be 
        # colocated, so their length are summed.
        seq_len_list = torch.tensor(seq_len_list).view(-1, 2).sum(-1).tolist()
    max_length_per_dp = worker.device_mesh["sp"].size() * worker.device_mesh["tp"].size() * (
        worker.config.max_length_per_device
        if torch.is_grad_enabled()
        else worker.config.max_inference_length_per_device
    )
    assert max(seq_len_list) <= max_length_per_dp, \
        f"The longest trajectory has a total length of {max(seq_len_list)}," \
        f"which exceeds the maximum length per dp {max_length_per_dp}."
    n_minibatches = math.ceil(
        sum(seq_len_list) / max_length_per_dp
    )

    # Every dp should has identical number of minibatches, thus the 
    # total number of minibatches must be a multiple of dp size.
    multiple_of = worker.device_mesh["dp"].size()
    if n_minibatches % multiple_of != 0:
        n_minibatches += multiple_of - n_minibatches % multiple_of

    # Partition data into n_minibatches balanced minibatches.
    while True:

        global PAD_TRAJECTORIES
        if len(seq_len_list) < n_minibatches:
            # Perhaps the number of minibatches is larger than the number 
            # of trajectories so that there are not enough trajectories 
            # to fill all minibatches.
            PAD_TRAJECTORIES = n_minibatches - len(seq_len_list)
            trajectory_length = 2 * worker.device_mesh["sp"].size()
            trajectory = {
                k: torch.zeros((trajectory_length), dtype=v.dtype)
                for k, v in data_list[0].items()
            }
            data_list.extend(PAD_TRAJECTORIES * [trajectory])
            seq_len_list.extend(PAD_TRAJECTORIES * [trajectory_length])
        else:
            PAD_TRAJECTORIES = 0

        partitions: List[List[int]] = get_seqlen_balanced_partitions(
            seq_len_list, k_partitions=n_minibatches, equal_size=False
        )
        max_minibatch_length = max([
            sum([seq_len_list[p] for p in partition])
            for partition in partitions
        ])
        if max_minibatch_length <= max_length_per_dp:
            break
        n_minibatches += worker.device_mesh["dp"].size()
    n_minibatches_per_dp = n_minibatches // worker.device_mesh["dp"].size()

    if pair:
        partitions = [
            sum([[2 * p, 2 * p + 1] for p in partition], [])
            for partition in partitions
        ]
    global SHUFFLE_INDICES
    SHUFFLE_INDICES = sum(partitions, [])
    return [
        [
            [data_list[p] for p in partition]
            for partition in partitions[rank * n_minibatches_per_dp:(rank + 1) * n_minibatches_per_dp]
        ]
        for rank in range(worker.device_mesh["dp"].size())
    ]

def scatter_data_lists_along_sp_dim(data_lists, device_mesh):

    if device_mesh.get_local_rank() != 0:
        return split_and_scatter_list(None, device_mesh)[0]

    double_size = 2 * device_mesh.size()
    all_rank_data_lists = []
    for rank in range(device_mesh.size()):
        data_lists_for_rank = []
        for data_list in data_lists:
            scattered_data_list = []
            for ex in data_list:
                # To apply ZigZag Ring Attention, every trajectory is 
                # evenly partitioned into 2 * sp size segments and each 
                # rank sequentially get the head and tail.
                # See https://zhuanlan.zhihu.com/p/683714620.
                half_seqlen = len(ex["states"]) // double_size
                scattered_data_list.append({
                    k: torch.cat((
                        v[rank * half_seqlen:(rank + 1) * half_seqlen],
                        v[(double_size - rank - 1) * half_seqlen: (double_size - rank) * half_seqlen]
                    ))
                    for k, v in ex.items()
                })
            data_lists_for_rank.append(scattered_data_list)
        all_rank_data_lists.append(data_lists_for_rank)

    return split_and_scatter_list(all_rank_data_lists, device_mesh)[0]

def scatter_data_lists_along_tp_dim(data_lists, device_mesh):

    n_minibatches = torch.LongTensor([
        len(data_lists)
        if device_mesh.get_local_rank() == 0
        else 0
    ]).to(torch.cuda.current_device())
    kwargs = {
        "group": device_mesh.get_group(),
        "group_src": 0
    }
    dist.broadcast(n_minibatches, **kwargs)

    if device_mesh.get_local_rank() != 0:
        data_lists = n_minibatches.item() * [None]
    dist.broadcast_object_list(data_lists, **kwargs)

    return data_lists

def pack_data_lists_into_minibatches(data_lists, multiple_of):

    minibatches = []
    for data_list in data_lists:
        minibatch = {}
        for k in data_list[0].keys():
            tensors = [ex[k] for ex in data_list]
            # When using tensor parallelism, the length of minibatch 
            # must be multiple of tp size so that the sequence can be 
            # evenly sharded.
            minibatch_length = sum([len(tensor) for tensor in tensors])
            if minibatch_length % multiple_of != 0:
                pad_tokens = multiple_of - minibatch_length % multiple_of
                tensors.append(torch.zeros((pad_tokens), dtype=tensors[0].dtype))
            minibatch[k] = torch.cat(tensors).unsqueeze(0).to(
                torch.cuda.current_device()
            )
        # `update_params_of_ring_attn` requires `cu_seqlens` to mask 
        # the attention across trajectories within a minibatch. 
        seqlens = torch.IntTensor([len(tensor) for tensor in tensors])
        minibatch["cu_seqlens"] = torch.cumsum(
            torch.cat((torch.IntTensor([0]), seqlens)),
            0, dtype=torch.int32
        ).to(torch.cuda.current_device())
        minibatches.append(minibatch)
        
    return minibatches

def split_minibatches_into_data_list(minibatches):
    
    data_list = []
    for minibatch in minibatches:
        cu_seqlens = minibatch.pop("cu_seqlens")
        minibatch = {
            k: v.squeeze(0).to("cpu")
            for k, v in minibatch.items()
        }
        for start_idx, end_idx in zip(
            cu_seqlens[:-1], cu_seqlens[1:]
        ):
            data_list.append({
                k: v[start_idx:end_idx]
                for k, v in minibatch.items()
            })
    return data_list

def gather_data_list_along_sp_dim(data_list, device_mesh):

    data_lists = gather_and_concat_list([data_list], device_mesh)
    if device_mesh.get_local_rank() != 0:
        return

    data_list = []
    for exs in zip(*data_lists):
        half_seqlen = len(exs[0]["states"]) // 2
        ex = {
            k: torch.cat(
                [ex[k][:half_seqlen] for ex in exs] + [ex[k][half_seqlen:] for ex in exs[::-1]]
            )
            for k in exs[0].keys()
        }
        length = torch.argmax(ex["position_ids"]).item()
        if length == 0:
            continue
        ex = {
            k: v[:length + 1] for k, v in ex.items()
        }
        data_list.append(ex)
    return data_list

def resume_order_of_data_list(shuffled_data_list):

    data_list = len(shuffled_data_list) * [None]
    for idx, ex in zip(SHUFFLE_INDICES, shuffled_data_list):
        data_list[idx] = ex
    if PAD_TRAJECTORIES > 0:
        data_list = data_list[:-PAD_TRAJECTORIES]
    return data_list

def count_total_actions(minibatches, device_mesh):
        
    total_actions = sum(
        [minibatch["action_mask"].sum() for minibatch in minibatches]
    )
    total_actions = torch.Tensor(
        [total_actions]
    ).to(torch.cuda.current_device())
    for mesh_name in ["sp", "dp"]:
        dist.all_reduce(
            total_actions,
            op=dist.ReduceOp.SUM,
            group=device_mesh[mesh_name].get_group()
        )
    return total_actions.to("cpu").item()