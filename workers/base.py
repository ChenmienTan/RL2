from typing import List, Dict, Optional, Union
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision
)
import wandb
from utils.fsdp import (
    get_fsdp_wrap_policy,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer
)
from utils.seqlen_balance import get_seqlen_balanced_partitions
from utils.ring_attn import ring_attn_all_gather
from utils.comm import gather_and_concat_list
        

class Worker:

    def __init__(self, config, device_mesh, train: bool):

        self.config = config
        self.device_mesh = device_mesh
        self.train = train

        self.sp_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "sp"),
            mesh_shape=(
                device_mesh.size() // config.sp_size,
                config.sp_size
            )
        )

    def prepare_model_optimizer(self):

        if self.train and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        auto_wrap_policy = get_fsdp_wrap_policy(self.model)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        ) if self.train else None

        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            device_id=torch.cuda.current_device()
        )

        if self.train:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        self.offload_model_to_cpu()

    def offload_model_to_cpu(self):
        if self.config.offload_model:
            offload_fsdp_model_to_cpu(self.model)
    
    def load_model_to_gpu(self):
        if self.config.offload_model:
            load_fsdp_model_to_gpu(self.model)

    def offload_optimizer_to_cpu(self):
        if self.config.offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def load_optimizer_to_gpu(self):
        if self.config.offload_optimizer:
            load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())

    def scatter_and_pack_data_list(
        self,
        data_list: Optional[List[Dict[str, torch.Tensor]]],
        train: bool
    ) -> Union[List[List[Dict[str, torch.Tensor]]], List[Dict[str, torch.Tensor]]]:

        if self.device_mesh.get_rank() == 0:

            seq_len_list = [ex["states"].shape[-1] for ex in data_list]
            n_minibatches = math.ceil(
                sum(seq_len_list) / (
                    self.config.max_length_per_device * self.sp_device_mesh["sp"].size()
                )
            )
            # At least n_minibatches minibatches are needed.
            # Every dp should has identical number of 
            # minibatches, thus the total number of minibatches 
            # must be a multiple of world size. Additinally, at 
            # training, the number of minibatches on each dp must 
            # be a multiple of updates so that they can be evenly 
            # devided into multiple batches, with each being used 
            # for an update.
            multiple_of = self.sp_device_mesh["dp"].size() * (self.config.update_per_rollout if train else 1)
            if n_minibatches % multiple_of != 0:
                n_minibatches += (multiple_of - n_minibatches % multiple_of)
            
            partitions: List[List[int]] = get_seqlen_balanced_partitions(
                seq_len_list, k_partitions=n_minibatches, equal_size=False
            )
            self.shuffle_indices: List[int] = sum(partitions, [])
            # Cache this for `resume_and_gather_data_list`.
            data_lists: List[List[Dict[str, torch.Tensor]]] = [
                [data_list[p] for p in partition]
                for partition in partitions
            ] # Trajectories within an inner list will be packed into a minibatch.
            n_minibatches_per_process = n_minibatches // self.sp_device_mesh["dp"].size()
            data_lists: List[List[List[Dict[str, torch.Tensor]]]] = [
                data_lists[rank * n_minibatches_per_process:(rank + 1) * n_minibatches_per_process]
                for rank in range(
                    self.sp_device_mesh["dp"].size()
                )
                for _ in range(
                    self.sp_device_mesh["sp"].size()
                ) # The n-th list contains data lists for rank n.
            ]
        
        else:
            data_lists = [None for _ in range(self.device_mesh.size())]
            
        data_list = [None]
        dist.scatter_object_list(data_list, data_lists, src=0)
        data_list: List[List[Dict[str, torch.Tensor]]] = data_list[0]
        # Next we pack trajectories within the inner lists into minibatch.

        multiple_of = 2 * self.sp_device_mesh["sp"].size()
        rank = self.sp_device_mesh["sp"].get_local_rank()
        minibatches: List[Dict[str, torch.Tensor]] = []
        for data in data_list:
            minibatch: Dict[str, torch.Tensor] = {}
            for k in data[0].keys():
                tensors = []
                for ex in data:
                    tensor = ex[k]
                    # Zigzag ring attention is used to balance 
                    # the load across devices, where the sequence 
                    # length needs to be multiple of 2 * 
                    # world_size and each rank sequentially get 
                    # the head and tail.
                    # See https://zhuanlan.zhihu.com/p/683714620.
                    if tensor.shape[-1] % multiple_of != 0:
                        padding_tokens = multiple_of - tensor.shape[-1] % multiple_of
                        tensor = torch.cat(
                            (tensor, torch.zeros((1, padding_tokens), dtype=tensor.dtype)),
                        -1)
                    half_seqlen = tensor.shape[-1] // multiple_of
                    tensor = torch.cat((
                        tensor[:, rank * half_seqlen:(rank + 1) * half_seqlen],
                        tensor[:, (multiple_of - rank - 1) * half_seqlen: (multiple_of - rank) * half_seqlen]
                    ), -1)
                    tensors.append(tensor)
                minibatch[k] = torch.cat(tensors, -1).to(torch.cuda.current_device())
            seqlens = torch.IntTensor(
                [tensor.shape[-1] for tensor in tensors]
            )
            minibatch["cu_seqlens"] = torch.cumsum(
                torch.cat((torch.IntTensor([0]), seqlens)),
                0, dtype=torch.int32
            ).to(torch.cuda.current_device())
            # Required by `update_params_of_ring_attn`.
            minibatches.append(minibatch)

        if train:
            # Group minibatches into batches.
            n_minibatches_per_update = len(minibatches) // self.config.update_per_rollout
            return [
                minibatches[update * n_minibatches_per_update:(update + 1) * n_minibatches_per_update]
                for update in range(self.config.update_per_rollout)
            ]
        else:
            return minibatches
    
    def resume_and_gather_data_list(
        self,
        minibatches: List[Dict[str, torch.Tensor]]
    ) -> Optional[List[Dict[str, torch.Tensor]]]:

        data_list: List[Dict[str, torch.Tensor]] = []
        for minibatch in minibatches:
            cu_seqlens = self.sp_device_mesh["sp"].size() * minibatch["cu_seqlens"]
            minibatch = ring_attn_all_gather(minibatch, self.sp_device_mesh["sp"])
            for start_idx, end_idx in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                unpad_end_idx = torch.where(
                    minibatch["eos_mask"][:, start_idx:end_idx]
                )[1][0]
                data_list.append({
                    k: v[:, start_idx:start_idx + unpad_end_idx + 1].to("cpu")
                    for k, v in minibatch.items()
                })

        if self.sp_device_mesh["sp"].get_local_rank() == 0:
            shuffled_data_list = gather_and_concat_list(data_list, self.sp_device_mesh["dp"])

        if self.device_mesh.get_rank() == 0:
            data_list = [None for _ in range(len(shuffled_data_list))]
            for idx, data in zip(self.shuffle_indices, shuffled_data_list):
                data_list[idx] = data
            return data_list
        else:
            return None

    def log(self, metrics: Dict[str, List], step: int):

        metrics = {
            k: gather_and_concat_list(v, self.device_mesh)
            for k, v in metrics.items()
        }
        
        if self.device_mesh.get_rank() == 0:
            wandb.log({
                k: torch.Tensor(v).mean().item()
                for k, v in metrics.items()
            }, step=step)

    def save(self, path):

        self.load_model_to_gpu()
        with FSDP.summon_full_params(
            self.model,
            offload_to_cpu=True,
            rank0_only=True,
            writeback=False
        ):
            if self.device_mesh.get_rank() == 0:
                self.model.save_pretrained(path)
        self.offload_model_to_cpu()