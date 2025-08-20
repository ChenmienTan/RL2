import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import transformers
from RL2.utils.parallelism import prepare_tp_model, prepare_dp_model
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