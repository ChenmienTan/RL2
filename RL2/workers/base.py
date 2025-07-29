from typing import List
import os
import math
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict
)
import transformers
from RL2.utils.models import prepare_tp_model, prepare_dp_model
from RL2.utils.seqlen_balance import get_seqlen_balanced_partitions
from RL2.utils.comm import split_and_scatter_list, gather_and_concat_list
from RL2.utils.offloading import load_model_to_device, load_optimizer_to_device

class Worker:

    def __init__(self, config, train: bool):

        self.config = config
        self.train = train

        self.prepare_device_mesh()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % (self.config.ddp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by ddp_size {self.config.ddp_size} * tp_size {self.config.tp_size}."
        self.fsdp_size = world_size // (self.config.ddp_size * self.config.tp_size)
        self.model_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("ddp", "fsdp", "tp"),
            mesh_shape=(self.config.ddp_size, self.fsdp_size, self.config.tp_size)
        )

        assert world_size % (self.config.sp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by sp_size {self.config.sp_size} * tp_size {self.config.tp_size}."
        self.dp_size = world_size // (self.config.sp_size * self.config.tp_size)
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "sp", "tp"),
            mesh_shape=(self.dp_size, self.config.sp_size, self.config.tp_size)
        )

    def prepare_model_optimizer(self):

        if self.train and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            prepare_tp_model(self.model, self.model_device_mesh["tp"])

        self.model = prepare_dp_model(
            self.model, self.model_device_mesh["ddp", "fsdp"]
        )

        if self.train:

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

            if self.config.optimizer_dir is not None:
                # TODO (P0): load integral state dict
                self.optimizer.load_state_dict(
                    torch.load(
                        f"{self.config.optimizer_dir}/optimizer_rank{dist.get_rank()}.pt"
                    )
                )
                load_optimizer_to_device(self, "cpu")

        load_model_to_device(self, "cpu")

    def scatter_and_pack_data_list(self, data_list, pack_minibatches=False, pair=False):

        if pack_minibatches:
            # Pack minibatches into multiple batches, where each batch is 
            # used for an update and contains multiple minibatches.
            if dist.get_rank() == 0:
                assert len(data_list) >= self.config.update_per_rollout, \
                    f"The number of trajectories {len(data_list)} is less than the number of updates {self.config.update_per_rollout}."
                n_trajectories_per_update = math.ceil(
                    len(data_list) / self.config.update_per_rollout
                )
                return [
                    self.scatter_and_pack_data_list(
                        data_list[update * n_trajectories_per_update:(update + 1) * n_trajectories_per_update]
                    )
                    for update in range(self.config.update_per_rollout)
                ]
            else:
                return [
                    self.scatter_and_pack_data_list(None)
                    for _ in range(self.config.update_per_rollout)
                ]

        if dist.get_rank() == 0:

            # We use ZigZag Ring Attention to partition sequences, where 
            # the length of each sequence needs to be multiple of 2 * 
            # sp size and each rank sequentially get the head and tail.
            # See https://zhuanlan.zhihu.com/p/683714620.
            multiple_of = 2 * self.device_mesh["sp"].size()
            for ex in data_list:
                if len(ex["states"]) % multiple_of == 0:
                    continue
                pad_tokens = multiple_of - len(ex["states"]) % multiple_of
                for k, v in ex.items():
                    ex[k] = torch.cat(
                        (v, torch.zeros((pad_tokens), dtype=v.dtype)),
                    -1)
                
            # We pack trajectories into minibatches for higher throughput.
            # To accommodate all trajectories, at least n_minibatches 
            # minibatches are needed.
            seq_len_list = [len(ex["states"]) for ex in data_list]
            if pair:
                # When pair, every two adjacent trajectories will be colocated, so their length are summed.
                seq_len_list = torch.tensor(seq_len_list).view(-1, 2).sum(-1).flatten().tolist()
            max_length_per_dp = (
                self.config.sp_size * self.config.tp_size * (
                    self.config.max_length_per_device
                    if torch.is_grad_enabled()
                    else self.config.max_inference_length_per_device
                )
            )
            assert max(seq_len_list) <= max_length_per_dp, \
                f"The longest trajectory has a total length of {max(seq_len_list)}," \
                f"which exceeds the maximum length per dp {max_length_per_dp}."
            n_minibatches = math.ceil(
                sum(seq_len_list) / max_length_per_dp
            )

            # Every dp should has identical number of minibatches, thus the 
            # total number of minibatches must be a multiple of dp size.
            multiple_of = self.device_mesh["dp"].size()
            if n_minibatches % multiple_of != 0:
                n_minibatches += (multiple_of - n_minibatches % multiple_of)

            if len(seq_len_list) < n_minibatches:
                # After letting n_minibatches to be multiple of dp size,
                # it may be larger than the number of trajectories so that
                # there are not enough trajectories to fill all minibatches.
                self.padding_trajectories = n_minibatches - len(seq_len_list)
                trajectory_length = 2 * self.device_mesh["sp"].size()
                trajectory = {
                    k: torch.zeros((trajectory_length), dtype=v.dtype)
                    for k, v in data_list[0].items()
                }
                data_list.extend(self.padding_trajectories * [trajectory])
                seq_len_list.extend(self.padding_trajectories * [trajectory_length])
            else:
                self.padding_trajectories = 0

            # Partition data into n_minibatches balanced minibatches.
            while True:
                partitions: List[List[int]] = get_seqlen_balanced_partitions(
                    seq_len_list, k_partitions=n_minibatches, equal_size=False
                )
                max_minibatch_length = max([
                    sum([seq_len_list[p] for p in partition])
                    for partition in partitions
                ])
                if max_minibatch_length <= max_length_per_dp:
                    break
                n_minibatches += self.device_mesh["dp"].size()
            n_minibatches_per_dp = n_minibatches // self.device_mesh["dp"].size()

            if pair:
                partitions = [
                    sum([[2 * p, 2 * p + 1] for p in partition], [])
                    for partition in partitions
                ]
            # We cache the shuffle indices to resume data order in 
            # `unpack_and_gather_data_list`.
            self.shuffle_indices = sum(partitions, [])
            # The n-th list contains data for rank n.
            data_lists = [
                [
                    [data_list[p] for p in partition]
                    for partition in partitions[rank * n_minibatches_per_dp:(rank + 1) * n_minibatches_per_dp]
                ]
                for rank in range(self.device_mesh["dp"].size())
                for _ in range(
                    self.device_mesh["sp", "tp"].size()
                )
            ]
        
        data_list = split_and_scatter_list(
            data_lists if dist.get_rank() == 0 else None,
        )[0]
        
        rank = self.device_mesh["sp"].get_local_rank()
        multiple_of = 2 * self.device_mesh["sp"].size()
        minibatches = []
        for data in data_list:
            minibatch = {}
            for k in data[0].keys():
                tensors = []
                for ex in data:
                    tensor = ex[k]
                    # To apply ZigZag Ring Attention, every trajectory is 
                    # evenly partitioned into 2 * sp size segments and each 
                    # rank sequentially get the head and tail.
                    # See https://zhuanlan.zhihu.com/p/683714620.
                    half_seqlen = len(tensor) // multiple_of
                    tensor = torch.cat((
                        tensor[rank * half_seqlen:(rank + 1) * half_seqlen],
                        tensor[(multiple_of - rank - 1) * half_seqlen: (multiple_of - rank) * half_seqlen]
                    ))
                    tensors.append(tensor)
                # When using tensor parallelism, the length of minibatch 
                # must be multiple of tp size so that the sequence can be 
                # evenly sharded.
                minibatch_length = sum([len(tensor) for tensor in tensors])
                if minibatch_length % self.config.tp_size != 0:
                    pad_tokens = self.config.tp_size - minibatch_length % self.config.tp_size
                    tensors.append(torch.zeros((pad_tokens), dtype=tensor.dtype))
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

    def unpack_and_gather_data_list(self, minibatches):

        data_list = []
        for minibatch in minibatches:
            cu_seqlens = minibatch.pop("cu_seqlens")
            for start_idx, end_idx in zip(
                cu_seqlens[:-1], cu_seqlens[1:]
            ):
                ex = {}
                for k, v in minibatch.items():
                    tensor = v.squeeze(0)[start_idx:end_idx]
                    tensors = [
                        torch.zeros_like(tensor)
                        for _ in range(self.device_mesh["sp"].size())
                    ]
                    dist.gather(
                        tensor,
                        tensors if self.device_mesh["sp"].get_local_rank() == 0 else None,
                        group=self.device_mesh["sp"].get_group(),
                        group_dst=0
                    )
                    # Devices with non-zero sp rank will process zero tensors.
                    mid_idx = len(tensor) // 2
                    inorder_tensors, reversed_tensors = [], []
                    for tensor in tensors:
                        inorder_tensors.append(tensor[:mid_idx])
                        reversed_tensors.append(tensor[mid_idx:])
                    ex[k] = torch.cat((
                        inorder_tensors + reversed_tensors[::-1]
                    )).to("cpu")

                length = torch.argmax(ex["position_ids"]).item()
                if length == 0:
                    continue
                ex = {
                    k: v[:length + 1] for k, v in ex.items()
                }
                data_list.append(ex)
        
        if self.device_mesh["sp"].get_local_rank() == 0 and self.device_mesh["tp"].get_local_rank() == 0:
            shuffled_data_list = gather_and_concat_list(
                data_list, self.device_mesh["dp"]
            )
            if dist.get_rank() == 0:
                data_list = len(shuffled_data_list) * [None]
                for idx, ex in zip(self.shuffle_indices, shuffled_data_list):
                    data_list[idx] = ex
                if self.padding_trajectories > 0:
                    data_list = data_list[:-self.padding_trajectories]
                return data_list
            
    def backward(self, loss):
        # https://github.com/ChenmienTan/RL2/issues/11
        (self.dp_size * self.config.sp_size * loss).backward()
    
    def optimizer_step(self):

        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )

        load_optimizer_to_device(
            self, torch.cuda.current_device()
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        load_optimizer_to_device(self, "cpu")

        return grad_norm.item()
    
    def save(self, step=None, rm=False):

        if step is not None and self.config.save_freq is not None and step % self.config.save_freq != 0:
            return
        
        path = self.config.save_dir + (
            f"/step{step}" if step is not None else "/latest"
        )
        os.makedirs(path, exist_ok=True)

        options = StateDictOptions(
            full_state_dict=True, cpu_offload=True
        )
        state_dict = get_model_state_dict(
            self.model, options=options
        )
        if dist.get_rank() == 0:

            self.tokenizer.save_pretrained(f"{path}/model")
            # unwrap the model
            model_to_save = self.model.module
            if rm:
                # For RM, we load token classification model for simplicity 
                # but save sequence classification model for compatibility.
                model_cls_name = model_to_save.__class__.__name__.replace(
                    "Token", "Sequence"
                )
                model_cls = getattr(transformers, model_cls_name)
                with torch.device("meta"):
                    model_to_save = model_cls._from_config(model_to_save.config)
            model_to_save.save_pretrained(
                f"{path}/model", state_dict=state_dict
            )

        dist.barrier()

        # TODO (P1): save optimizer state distributionally
        state_dict = get_optimizer_state_dict(
            self.model, self.optimizer, options=options
        )
        if dist.get_rank() == 0:
            
            os.makedirs(f"{path}/optimizer", exist_ok=True)
            torch.save(
                state_dict,
                f"{path}/optimizer/state_dict.pt"
            )

        dist.barrier()