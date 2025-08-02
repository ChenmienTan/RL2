import math
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import transformers
from RL2.utils.models import prepare_tp_model, prepare_dp_model
from RL2.utils.comm import split_and_scatter_list, gather_and_concat_list
from RL2.utils.sequences import (
    group_data_list_into_all_rank_data_lists,
    scatter_data_lists_along_sp_dim,
    scatter_data_lists_along_tp_dim,
    pack_data_lists_into_minibatches,
    split_minibatches_into_data_list,
    gather_data_list_along_sp_dim,
    resume_order_of_data_list
)
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

        load_model_to_device(self, "cpu")

    def scatter_and_pack_data_list(
        self, data_list, pack_minibatches=False, pair=False
    ):

        if pack_minibatches:
            # Pack minibatches into multiple batches, where each batch is 
            # used for an update and contains multiple minibatches.
            if dist.get_rank() == 0:
                assert len(data_list) >= self.config.update_per_rollout, \
                    f"The number of trajectories {len(data_list)} is less than the number of updates {self.config.update_per_rollout}."
                bsz = math.ceil(
                    len(data_list) / self.config.update_per_rollout
                )
                return [
                    self.scatter_and_pack_data_list(
                        data_list[update * bsz:(update + 1) * bsz]
                    )
                    for update in range(self.config.update_per_rollout)
                ]
            else:
                return [
                    self.scatter_and_pack_data_list(None)
                    for _ in range(self.config.update_per_rollout)
                ]

        if self.device_mesh["tp"].get_local_rank() == 0:
            if self.device_mesh["sp"].get_local_rank() == 0:
                if self.device_mesh["dp"].get_local_rank() == 0:
                    all_rank_data_lists = group_data_list_into_all_rank_data_lists(
                        self, data_list, pair
                    )
                data_lists = split_and_scatter_list(
                    all_rank_data_lists
                    if self.device_mesh["dp"].get_local_rank() == 0
                    else None,
                    self.device_mesh["dp"]
                )[0]
            data_lists = scatter_data_lists_along_sp_dim(
                data_lists
                if self.device_mesh["sp"].get_local_rank() == 0
                else None,
                self.device_mesh["sp"]
            )
        data_lists = scatter_data_lists_along_tp_dim(
            data_lists
            if self.device_mesh["tp"].get_local_rank() == 0
            else None,
            self.device_mesh["tp"]
        )
        return pack_data_lists_into_minibatches(
            data_lists,
            self.device_mesh["tp"].size()
        )

    def unpack_and_gather_data_list(self, minibatches):
        
        if self.device_mesh["tp"].get_local_rank() != 0:
            return
        data_list = split_minibatches_into_data_list(minibatches)
        data_list = gather_data_list_along_sp_dim(
            data_list, self.device_mesh["sp"]
        )
        if self.device_mesh["sp"].get_local_rank() == 0:
            data_list = gather_and_concat_list(
                data_list, self.device_mesh["dp"]
            )
            if self.device_mesh["dp"].get_local_rank() == 0:
                return resume_order_of_data_list(data_list)
            
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
        self.scheduler.step()
        load_optimizer_to_device(self, "cpu")

        return grad_norm.item()