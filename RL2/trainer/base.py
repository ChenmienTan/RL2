from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict
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
    
    def get_ckpt(self, worker, step):

        options = StateDictOptions(cpu_offload=True)
        return {
            "step": step,
            "dataloader": self.dataloader.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model": get_model_state_dict(
                worker.model, options=options
            ),
            "optimizer": get_optimizer_state_dict(
                worker.model, worker.optimizer, options=options
            )
        }
    
    def load_ckpt(self, worker):

        if self.config.trainer.load_ckpt_from_dir is None:
            return 0

        ckpt = self.get_ckpt(worker, 0)
        dcp.load(
            ckpt,
            checkpoint_id=self.config.trainer.load_ckpt_from_dir
        )
        
        self.dataloader.load_state_dict(ckpt["dataloader"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        set_model_state_dict(
            worker.model, ckpt["model"]
        )
        set_optimizer_state_dict(
            worker.model, worker.optimizer, ckpt["optimizer"]
        )

        return ckpt["step"]
    
    def save_ckpt(self, worker, step):

        if self.config.trainer.save_freq is None or step % self.config.trainer.save_freq != 0:
            return
        
        dcp.save(
            self.get_ckpt(worker, step),
            checkpoint_id=f"{self.config.trainer.save_dir}/step{step}"
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