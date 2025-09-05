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

def _tensor_dict_to_minibatches(
    worker, tensor_dict, pair: bool
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
    seq_len_list = (tensor_dict["eos_mask"].argmax(-1) + 1).tolist()
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
            # The number of sequences must be no less than `n_minibatches`.
            # If not, we pad the number of sequences to `n_minibatches`.
            PAD_SEQUENCES = n_minibatches - len(seq_len_list)
            for k, v in tensor_dict.items():
                tensor_dict[k] = torch.cat((
                    v,
                    torch.zeros(
                        (
                            (2 if pair else 1) * PAD_SEQUENCES,
                            v.shape[-1]
                        ),
                        dtype=v.dtype
                    )
                ))
            seq_len_list.extend(PAD_SEQUENCES * [0])
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
        {
            k: v[partition] for k, v in tensor_dict.items()
        }
        for partition in partitions
    ]

def tensor_dict_to_minibatches(
    worker, tensor_dict, pack_minibatches: bool, pair: bool
):

    if pack_minibatches:
        # Pack minibatches into multiple batches, where each batch is 
        # used for an update.
        if dist.get_rank() == 0:
            return [
                tensor_dict_to_minibatches(
                    worker, {
                        k: torch.chunk(
                            v, worker.config.update_per_rollout, dim=0
                        )[update]
                        for k, v in tensor_dict.items()
                    }, False, pair
                )
                for update in range(worker.config.update_per_rollout)
            ]
        else:
            return [
                tensor_dict_to_minibatches(worker, None, False, pair)
                for _ in range(worker.config.update_per_rollout)
            ]

    if worker.device_mesh["tp"].get_local_rank() == 0:
        if worker.device_mesh["sp"].get_local_rank() == 0:
            if worker.device_mesh["dp"].get_local_rank() == 0:
                minibatches = _tensor_dict_to_minibatches(
                    worker, tensor_dict, pair
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

def minibatches_to_tensor_dict(worker, minibatches):
    
    minibatches = [
        {
            k: v.to("cpu")
            for k, v in minibatch.items()
        }
        for minibatch in minibatches
    ]
    minibatches = gather_and_concat_list(
        minibatches, worker.device_mesh["dp"]
    )
    if dist.get_rank() == 0:

        tensor_dict = {
            k: torch.cat([
                minibatch[k] for minibatch in minibatches
            ])
            for k in minibatches[0].keys()
        }

        reversed_indices = len(SHUFFLE_INDICES) * [None]
        for idx, shuffle_idx in enumerate(SHUFFLE_INDICES):
            reversed_indices[shuffle_idx] = idx
        tensor_dict = {
            k: v[reversed_indices]
            for k, v in tensor_dict.items()
        }

        if PAD_SEQUENCES > 0:
            tensor_dict = {
                k: v[:-PAD_SEQUENCES]
                for k, v in tensor_dict.items()
            }

        return tensor_dict

def data_manager(pack_minibatches=False, pair=False, gather=False):
    def decorator(func):
        @functools.wraps(func)
        def func_with_data_scatter_and_gather(
            worker, tensor_dict, *args, **kwargs
        ):
            minibatches = tensor_dict_to_minibatches(
                worker, tensor_dict, pack_minibatches, pair
            )
            output = func(worker, minibatches, *args, **kwargs)
            if gather:
                output = minibatches_to_tensor_dict(worker, output)
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