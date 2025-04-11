from typing import List, Dict
import time
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision
)
import wandb
from utils import (
    all_gather,
    get_auto_wrap_policy,
    get_seqlen_balanced_partitions,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer
)


class TimeMemoryLogger:

    def __init__(self, device_mesh, op):
        self.device_mesh = device_mesh
        self.op = op

    def __enter__(self):
        self.start_time = time.time()
        start_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        if self.device_mesh.get_rank() == 0:
            print(f"Before {self.op}, {round(start_memory, 1)} GB memory is occupied.")

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        if self.device_mesh.get_rank() == 0:
            print(f"{self.op} takes {round(end_time - self.start_time, 1)} seconds.")
            print(f"After {self.op}, {round(end_memory, 1)} GB memory is occupied.")
        

class Worker:

    def __init__(self, config, device_mesh, train: bool):

        self.config = config
        self.device_mesh = device_mesh
        self.train = train

        if train:
            self.sp_device_mesh = dist.device_mesh.init_device_mesh(
                "cuda",
                mesh_dim_names=("dp", "sp"),
                mesh_shape=(device_mesh.size() // config.sp_size, config.sp_size)
            )

    def prepare_model_optimizer(self):

        if self.train and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        auto_wrap_policy = get_auto_wrap_policy(self.model)

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
            with TimeMemoryLogger(self.device_mesh, "offloading model to cpu"):
                offload_fsdp_model_to_cpu(self.model)
    
    def load_model_to_gpu(self):
        if self.config.offload_model:
            with TimeMemoryLogger(self.device_mesh, "loading model to cpu"):
                load_fsdp_model_to_gpu(self.model)

    def offload_optimizer_to_cpu(self):
        if self.config.offload_optimizer:
            with TimeMemoryLogger(self.device_mesh, "offloading optimizer to cpu"):
                offload_fsdp_optimizer(self.optimizer)

    def load_optimizer_to_gpu(self):
        if self.config.offload_optimizer:
            with TimeMemoryLogger(self.device_mesh, "loading optimizer to cpu"):
                load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())

    def pack_data_list_to_minibatches(self, data_list: List[Dict[str, torch.Tensor]], train: bool) -> List[Dict[str, torch.Tensor]]:

        seq_len_list = [ex["states"].shape[-1] for ex in data_list]
        n_minibatches = math.ceil(
            sum(seq_len_list) / (
                self.config.train_max_length_per_device * self.sp_device_mesh["sp"].size()
                if train else self.config.inference_max_length_per_device
            )
        )
        # At least n_minibatches minibatches are needed.
        # SP is used to shard cached activations, which is 
        # unnecessary at inference. Every dp should has identical 
        # number of minibatches, thus the total number of 
        # minibatches must be a multiple of world size. 
        # Additinally, at training, the number of minibatches on 
        # each dp must be a multiple of updates so that they can 
        # be evenly devided into multiple batches, with each being 
        # used for an update.
        multiple_of = self.sp_device_mesh["dp"].size() * self.config.update_per_rollout if train else self.device_mesh.size()
        if n_minibatches % multiple_of != 0:
            n_minibatches += (multiple_of - n_minibatches % multiple_of)
        
        partitions: List[List[int]] = get_seqlen_balanced_partitions(
            seq_len_list, k_partitions=n_minibatches, equal_size=False
        )
        self.shuffle_indices = sum(partitions, [])
        # cache this for `resume_minibatches_to_data_list`
        rank = self.sp_device_mesh["dp"].get_local_rank() if train else self.device_mesh.get_rank()
        n_minibatch_per_process = n_minibatches // (
            self.sp_device_mesh["dp"].size() if train else self.device_mesh.size()
        )
        partitions = partitions[rank * n_minibatch_per_process:(rank + 1) * n_minibatch_per_process]

        return [
            {   
                k: torch.cat(
                    [data_list[p][k] for p in partition],
                -1).to(torch.cuda.current_device())
                for k in data_list[0].keys()
            }
            for partition in partitions
        ]
    
    def resume_minibatches_to_data_list(self, minibatches: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        
        shuffled_data_list = []
        for minibatch in minibatches:
            end_indices = torch.where(minibatch["eos_mask"])[1]
            start_indices = torch.cat((
                torch.LongTensor([0]).to(torch.cuda.current_device()),
                end_indices[:-1] + 1
            ))
            for start_idx, end_idx in zip(start_indices, end_indices):
                shuffled_data_list.append({   
                    k: v[:, start_idx:end_idx + 1].to("cpu")
                    for k, v in minibatch.items()
                })
        
        shuffled_data_list = all_gather(shuffled_data_list, self.device_mesh)
        data_list = [None for _ in range(len(shuffled_data_list))]
        for idx, data in zip(self.shuffle_indices, shuffled_data_list):
            data_list[idx] = data

        return data_list

    def group_minibatches_into_batches(self, minibatches: List[Dict[str, torch.Tensor]]) -> List[List[Dict[str, torch.Tensor]]]:

        n_minibatches_per_update = len(minibatches) // self.config.update_per_rollout
        return [ # TODO: perhaps rank them
            minibatches[update * n_minibatches_per_update:(update + 1) * n_minibatches_per_update]
            for update in range(self.config.update_per_rollout)
        ]

    def log(self, metrics: Dict[str, List], step: int):

        metrics = {
            k: all_gather(v, self.device_mesh)
            for k, v in metrics.items()
        }
        
        if self.device_mesh.get_rank() == 0:
            wandb.log(
                {
                    k: torch.Tensor(v).mean().item()
                    for k, v in metrics.items()
                }, step=step    
            )

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