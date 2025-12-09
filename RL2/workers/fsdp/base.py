from typing import ContextManager, Dict, Union, List, Optional, Any
from omegaconf import OmegaConf, DictConfig
from accelerate import init_empty_weights
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.distributed.fsdp._runtime_utils import _lazy_init
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict
)
from transformers import get_scheduler
from RL2.workers import Worker
from RL2.utils.fsdp.data_parallelism import prepare_dp_model
from RL2.utils.fsdp.tensor_parallelism import prepare_tp_model
from RL2.utils.sequences import scatter_data, gather_data


class FSDPWorker(Worker):

    def __init__(self, config: DictConfig, train: bool):
        super().__init__(config, train)

        world_size = dist.get_world_size()
        assert world_size % (config.ddp_size * config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by ddp_size {config.ddp_size} * tp_size {config.tp_size}."
        self.model_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("ddp", "fsdp", "tp"),
            mesh_shape=(
                config.ddp_size,
                world_size // (config.ddp_size * config.tp_size),
                config.tp_size
            )
        )

        assert world_size % (config.cp_size * config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by cp_size {config.cp_size} * tp_size {config.tp_size}."
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "cp", "tp"),
            mesh_shape=(
                world_size // (config.cp_size * config.tp_size),
                config.cp_size,
                config.tp_size
            )
        )

    def _init_weight_context(self) -> ContextManager:
        # TODO: why offloading is incompatible with initialization on meta device?
        if any([
            dist.get_rank() == 0,
            self.device_mesh["tp"].size() > 1 and self.device_mesh["tp"].get_local_rank() == 0,
            getattr(self.config, "offload_model", False)
        ]):
            return torch.device("cpu")
        return init_empty_weights()

    def _prepare_model_optimizer(self):

        if self.train and self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            prepare_tp_model(self.model, self.model_device_mesh["tp"])

        self.model = prepare_dp_model(
            self.model,
            self.config.dtype,
            self.config.tp_size == 1,
            self.model_device_mesh["ddp", "fsdp"]
        )

        if self.train:

            optimizer_config = OmegaConf.to_container(self.config.optimizer)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_config
            )

        self._load_model_to_device("cpu")
    
    def prepare_scheduler(self, total_steps: int):

        num_training_steps = total_steps * getattr(
            self.config, "update_per_rollout", 1
        )
        scheduler_config = OmegaConf.to_container(self.config.scheduler)
        scheduler_name = scheduler_config.pop("name")
        num_warmup_steps = int(
            scheduler_config.pop("warmup_ratio") * num_training_steps
        )
        self.scheduler = get_scheduler(
            scheduler_name,
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=scheduler_config
        )

    def _scatter_data(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        pack_minibatches: bool = False,
        pair: bool = False
    ) -> Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]:

        max_length_per_dp = self.device_mesh["cp"].size() * self.device_mesh["tp"].size() * (
            self.config.max_length_per_device
            if torch.is_grad_enabled()
            else self.config.max_inference_length_per_device
        )
        return scatter_data(
            tensor_dict,
            self.device_mesh["dp"].get_group(),
            self.device_mesh["dp"].size(),
            max_length_per_dp,
            self.config.update_per_rollout if pack_minibatches else None,
            pair
        )

    def _gather_data(
        self, minibatches: List[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        return gather_data(minibatches, self.device_mesh["dp"].get_group())

    # TODO: maybe simplify this function
    def _load_model_to_device(self, device: Union[torch.device, str]):
    
        if not getattr(self.config, "offload_model", False):
            return

        torch.cuda.empty_cache()
        _lazy_init(self.model, self.model)
        for handle in self.model._all_handles:
            if handle._offload_params:
                continue
            flat_param = handle.flat_param
            handle.flat_param_to(device, non_blocking=True)
            flat_param._local_shard = flat_param.data
        torch.cuda.empty_cache()

    def _load_optimizer_to_device(self, device: Union[torch.device, str]):

        if not getattr(self.config, "offload_optimizer", False):
            return

        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(
                            device, non_blocking=True
                        )

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        # https://github.com/ChenmienTan/RL2/issues/11
        return self.device_mesh["dp"].size() * self.config.cp_size * loss
    
    def _optimizer_step(self) -> int:

        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )
        self._load_optimizer_to_device(
            torch.cuda.current_device()
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._load_optimizer_to_device("cpu")
        self.scheduler.step()
        return grad_norm.item()

    def _get_model_state_dict(
        self, full_state_dict: bool = False
    ) -> Dict[str, Any]:

        options = StateDictOptions(
            full_state_dict=full_state_dict,
            cpu_offload=True
        )
        self._load_model_to_device(torch.cuda.current_device())
        state_dict = get_model_state_dict(self.model, options=options)
        self._load_model_to_device("cpu")
        return state_dict

    def _get_ckpt(self) -> Dict[str, Dict[str, Any]]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

    def load_ckpt(self, checkpoint_id: str):

        ckpt = self._get_ckpt()
        dcp.load(ckpt, checkpoint_id=checkpoint_id)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

    def save_ckpt(self, save_dir: str):
        
        self.save_model(f"{save_dir}/model")
        dcp.save(
            self._get_ckpt(),
            checkpoint_id=f"{save_dir}/optimizer_scheduler"
        )

    def save_model(self, save_dir: str):

        state_dict = self._get_model_state_dict(full_state_dict=True)
        if dist.get_rank() == 0:

            self.tokenizer.save_pretrained(save_dir)
            self.model.module.save_pretrained(
                save_dir, state_dict=state_dict
            )

        dist.barrier()