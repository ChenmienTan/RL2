from typing import List
import math
import functools
import torch
import torch.distributed as dist
from RL2.utils.seqlen_balance import get_seqlen_balanced_partitions
from RL2.utils.comm import (
    split_and_scatter_list,
    boardcast_list,
    gather_and_concat_list
)

def pad_tensor_dict_to_multiple_of(tensor_dict, multiple_of):

    if len(tensor_dict["states"]) % multiple_of == 0:
        return tensor_dict
    pad_tokens = multiple_of - len(tensor_dict["states"]) % multiple_of
    tensor_dict = {
        k: torch.cat((
            v,
            torch.zeros((pad_tokens), dtype=v.dtype)
        ))
        for k, v in tensor_dict.items()
    }
    tensor_dict["position_ids"] = torch.arange(len(tensor_dict["states"]))
    return tensor_dict

def pack_tensor_dicts_to_minibatch(tensor_dicts):
    return {
        k: torch.cat([td[k] for td in tensor_dicts])
        for k in tensor_dicts[0].keys()
    }

def pack_tensor_dicts_to_minibatches(
    worker, tensor_dicts, pair=False
):

    # We pack sequences into minibatches for higher throughput.
    # There are two constrains:
    #   * The length of any minibatch cannot exceed `max_length_per_dp`
    #   * The number of minibatches must be multiple of dp size (so that
    #     each dp shares identical number of minibatches)
    # To satisfy the first constraint, the number of minibatches must be
    # at least `math.ceil(total_length / max_length_per_dp)`.
    # Starting from the first multiple of dp size that is no less than 
    # the value, we pack sequences into `n_minibatches` minibatches and 
    # check whether the first constraint is satisfied. If not, we increase 
    # `n_minibatches` by dp size (so that the second constraint is always 
    # satisfied) and repeat the loop.
    seq_len_list = [len(td["states"]) for td in tensor_dicts]
    if pair:
        # When pair, every two adjacent sequences will be colocated, so 
        # their length are summed.
        seq_len_list = torch.tensor(seq_len_list).view(-1, 2).sum(-1).tolist()
    max_length_per_dp = worker.device_mesh["sp"].size() * worker.device_mesh["tp"].size() * (
        worker.config.max_length_per_device
        if torch.is_grad_enabled()
        else worker.config.max_inference_length_per_device
    )
    assert max(seq_len_list) <= max_length_per_dp, \
        f"The longest sequence has a total length of {max(seq_len_list)}," \
        f"which exceeds the maximum length per dp {max_length_per_dp}."
    n_minibatches = math.ceil(
        sum(seq_len_list) / max_length_per_dp
    )
    multiple_of = worker.device_mesh["dp"].size()
    if n_minibatches % multiple_of != 0:
        n_minibatches += multiple_of - n_minibatches % multiple_of

    # Partition sequences into n_minibatches balanced minibatches.
    while True:

        global PAD_SEQUENCES
        if n_minibatches > len(seq_len_list):
            # We pack sequences into `n_minibatches` minibatches, where the
            # number of sequences must be no less than `n_minibatches`. If 
            # not, we pad the number of sequences to `n_minibatches`.
            PAD_SEQUENCES = n_minibatches - len(seq_len_list)
            sequence_length = 2 * worker.device_mesh["sp"].size()
            sequence = {
                k: torch.arange(sequence_length) if k == "position_ids"
                else torch.zeros((sequence_length), dtype=v.dtype)
                for k, v in tensor_dicts[0].items()
            }
            tensor_dicts.extend(PAD_SEQUENCES * [sequence])
            seq_len_list.extend(PAD_SEQUENCES * [sequence_length])
        else:
            PAD_SEQUENCES = 0

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

    if pair:
        partitions = [
            sum([[2 * p, 2 * p + 1] for p in partition], [])
            for partition in partitions
        ]
    global SHUFFLE_INDICES
    SHUFFLE_INDICES = sum(partitions, [])

    return [
        pack_tensor_dicts_to_minibatch([
            tensor_dicts[p] for p in partition
        ])
        for partition in partitions
    ]

def scatter_and_pack_tensor_dicts(
    worker, tensor_dicts, pack_minibatches=False, pair=False
):

    if pack_minibatches:
        # Pack minibatches into multiple batches, where each batch is 
        # used for an update.
        if dist.get_rank() == 0:
            assert len(tensor_dicts) >= worker.config.update_per_rollout, \
                f"The number of sequences {len(tensor_dicts)} is less than the number of updates {worker.config.update_per_rollout}."
            bsz = math.ceil(
                len(tensor_dicts) / worker.config.update_per_rollout
            )
            return [
                scatter_and_pack_tensor_dicts(
                    worker, tensor_dicts[update * bsz:(update + 1) * bsz]
                )
                for update in range(worker.config.update_per_rollout)
            ]
        else:
            return [
                scatter_and_pack_tensor_dicts(worker, None)
                for _ in range(worker.config.update_per_rollout)
            ]

    if worker.device_mesh["tp"].get_local_rank() == 0:
        if worker.device_mesh["sp"].get_local_rank() == 0:
            if worker.device_mesh["dp"].get_local_rank() == 0:
                # We use ZigZag Ring Attention to partition sequences, where 
                # the length of each sequence needs to be multiple of 2 * 
                # sp size and each rank sequentially get the head and tail.
                # See https://zhuanlan.zhihu.com/p/683714620.
                tensor_dicts = [
                    pad_tensor_dict_to_multiple_of(
                        td, 2 * worker.device_mesh["sp"].size()
                    )
                    for td in tensor_dicts
                ]
                minibatches = pack_tensor_dicts_to_minibatches(
                    worker, tensor_dicts, pair
                )
            minibatches = split_and_scatter_list(
                minibatches
                if worker.device_mesh["dp"].get_local_rank() == 0
                else None,
                worker.device_mesh["dp"]
            )
        minibatches = boardcast_list(
            minibatches
            if worker.device_mesh["sp"].get_local_rank() == 0
            else None,
            worker.device_mesh["sp"]
        )
    minibatches = boardcast_list(
        minibatches
        if worker.device_mesh["tp"].get_local_rank() == 0
        else None,
        worker.device_mesh["tp"]
    )
    return [
        {
            k: v.to(torch.cuda.current_device())
            for k, v in minibatch.items()
        }
        for minibatch in minibatches
    ]

def position_ids_to_cu_seqlens(position_ids):

    indices = torch.arange(
        len(position_ids),
        dtype=torch.int32,
        device=position_ids.device
    )
    return torch.cat((
        indices[position_ids == 0],
        torch.tensor(
            position_ids.size(),
            dtype=torch.int32,
            device=position_ids.device
        )
    ))

def split_minibatches_into_tensor_dicts(minibatches):
    
    tensor_dicts = []
    for minibatch in minibatches:
        cu_seqlens = position_ids_to_cu_seqlens(
            minibatch["position_ids"]
        )
        for start_idx, end_idx in zip(
            cu_seqlens[:-1], cu_seqlens[1:]
        ):
            tensor_dicts.append({
                k: v[start_idx:end_idx].to("cpu")
                for k, v in minibatch.items()
            })
        
    return tensor_dicts

def resume_order_of_tensor_dicts(raw_tensor_dicts):

    tensor_dicts = len(raw_tensor_dicts) * [None]
    for idx, td in zip(SHUFFLE_INDICES, raw_tensor_dicts):
        tensor_dicts[idx] = td
    if PAD_SEQUENCES > 0:
        tensor_dicts = tensor_dicts[:-PAD_SEQUENCES]
    return tensor_dicts

def unpack_and_gather_tensor_dicts(worker, minibatches):
        
    tensor_dicts = split_minibatches_into_tensor_dicts(minibatches)
    tensor_dicts = gather_and_concat_list(
        tensor_dicts, worker.device_mesh["dp"]
    )
    if dist.get_rank() == 0:
        return resume_order_of_tensor_dicts(tensor_dicts)

def data_manager(pack_minibatches=False, pair=False, gather=False):
    def decorator(func):
        @functools.wraps(func)
        def func_with_data_scatter_and_gather(
            worker, tensor_dicts, *args, **kwargs
        ):
            minibatches = scatter_and_pack_tensor_dicts(
                worker, tensor_dicts, pack_minibatches, pair
            )
            output = func(worker, minibatches, *args, **kwargs)
            if gather:
                output = unpack_and_gather_tensor_dicts(worker, output)
            return output
        return func_with_data_scatter_and_gather
    return decorator

def count_total(minibatches, key, device_mesh):

    if isinstance(key, tuple):
        return tuple(
            count_total(minibatches, k, device_mesh)
            for k in key
        )
        
    total = sum(
        [minibatch[key].sum() for minibatch in minibatches]
    )
    total = torch.Tensor(
        [total]
    ).to(torch.cuda.current_device())
    dist.all_reduce(
        total,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return total.to("cpu").item()