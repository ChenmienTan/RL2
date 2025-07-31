from omegaconf import OmegaConf
import os
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict
)
import transformers
import wandb

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
        self.config = config

        if dist.get_rank() == 0:
            print(OmegaConf.to_yaml(config))
            if not config.trainer.disable_wandb:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config)
                )
            else:
                wandb.log = lambda *args, **kwargs: None
    
    def prepare_scheduler(self, worker):

        num_training_steps = self.config.trainer.n_epochs * len(self.dataloader)
        num_warmup_steps = int(worker.config.warmup_ratio * num_training_steps)

        return transformers.get_cosine_schedule_with_warmup(
            worker.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def load_ckpt(self, worker):

        dir = self.config.trainer.load_ckpt_from_dir
        if dir is None:
            return 0

        self.dataloader.load_state_dict(
            torch.load(
                f"{dir}/dataloader.pt"
            )
        )
        worker.model.load_state_dict(
            torch.load(
                f"{dir}/model/rank{dist.get_rank()}.pt"
            )
        )
        worker.optimizer.load_state_dict(
            torch.load(
                f"{dir}/optimizer/rank{dist.get_rank()}.pt"
            )
        )
        self.scheduler.load_state_dict(
            torch.load(
                f"{dir}/scheduler.pt"
            )
        )

        with open(f"{dir}/step.txt") as f:
            return int(f.read())
    
    def save_ckpt(self, worker, step):

        if self.config.trainer.save_freq is None or step % self.config.trainer.save_freq != 0:
            return
        
        dir = f"{self.config.trainer.save_dir}/step{step}"
        if dist.get_rank() == 0:

            os.makedirs(f"{dir}/model", exist_ok=True)
            os.makedirs(f"{dir}/optimizer", exist_ok=True)

            with open(f"{dir}/step.txt", "w") as f:
                f.write(str(step))

            torch.save(
                self.dataloader.state_dict(),
                f"{dir}/dataloader.pt"
            )
            torch.save(
                self.scheduler.state_dict(),
                f"{dir}/scheduler.pt"
            )

        torch.save(
            worker.model.state_dict(),
            f"{dir}/model/rank{dist.get_rank()}.pt"
        )
        torch.save(
            worker.optimizer.state_dict(),
            f"{dir}/optimizer/rank{dist.get_rank()}.pt"
        )

    def save_model(self, worker, rm=False):

        dir = f"{self.config.trainer.save_dir}/latest"
        options = StateDictOptions(
            full_state_dict=True, cpu_offload=True
        )
        state_dict = get_model_state_dict(
            worker.model, options=options
        )
        if dist.get_rank() == 0:

            worker.tokenizer.save_pretrained(dir)
            # unwrap the model
            model_to_save = worker.model.module
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
                dir, state_dict=state_dict
            )

        dist.barrier()