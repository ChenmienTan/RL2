from typing import Dict, Any, Sequence
from omegaconf import OmegaConf, DictConfig
import glob
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import wandb
from RL2.workers import Worker


class Trainer:
    
    def __init__(self, config: DictConfig):
        
        OmegaConf.resolve(config)
        self.load_dir = config.trainer.load_ckpt_from
        if self.load_dir == "latest":
            load_dirs = glob.glob(f"{config.trainer.save_dir}/step*")
            self.load_dir = max(
                load_dirs, key=lambda dir: int(dir.split("/step")[-1])
            ) if load_dirs else None
        if self.load_dir is not None:
            if dist.get_rank() == 0:
                print(f"Loading checkpoint from {self.load_dir}")
            if hasattr(config, "actor"):
                config.actor.model_name = f"{self.load_dir}/actor/model"
            if hasattr(config, "critic"):
                config.critic.model_name = f"{self.load_dir}/critic/model"
            if hasattr(config, "rollout"):
                config.rollout.server_args.model_path = f"{self.load_dir}/actor/model"
        
        self.config = config

        if dist.get_rank() == 0:
            print(OmegaConf.to_yaml(config))
            if config.trainer.use_wandb:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config)
                )
            else:
                wandb.log = lambda *args, **kwargs: None

    def get_ckpt(self, step: int) -> Dict[str, Any]:
        return {
            "step": step,
            "dataloader": self.train_dataloader.state_dict()
        }

    def load_ckpt(self, workers: Sequence[Worker]) -> int:

        if self.load_dir is None:
            return 0
        for worker in workers:
            worker_name = "actor" if "Actor" in worker.__class__.__name__ else "critic"
            worker.load_ckpt(f"{self.load_dir}/{worker_name}/optimizer_scheduler")

        ckpt = self.get_ckpt(0)
        dcp.load(ckpt, checkpoint_id=f"{self.load_dir}/trainer")
        self.train_dataloader.load_state_dict(ckpt["dataloader"])
        return ckpt["step"]

    def save_ckpt(self, workers: Sequence[Worker], step: int):

        if self.config.trainer.save_freq is None or step % self.config.trainer.save_freq != 0:
            return

        save_dir = f"{self.config.trainer.save_dir}/step{step}"
        for worker in workers:
            worker_name = "actor" if "Actor" in worker.__class__.__name__ else "critic"
            worker.save_ckpt(f"{save_dir}/{worker_name}")

        dcp.save(
            self.get_ckpt(step),
            checkpoint_id=f"{save_dir}/trainer"
        )

    def save_model(self, workers: Sequence[Worker]):

        save_dir = self.config.trainer.save_dir
        if self.config.trainer.save_freq is not None:
            save_dir += "/latest"
        
        for worker in workers:
            worker_name = "actor" if "Actor" in worker.__class__.__name__ else "critic"
            worker.save_model(f"{save_dir}/{worker_name}")