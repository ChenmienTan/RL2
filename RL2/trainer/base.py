from omegaconf import OmegaConf
import torch.distributed as dist
from transformers import get_scheduler
import wandb

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
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
    
    def prepare_scheduler(self, worker):

        num_training_steps = self.config.trainer.n_epochs * len(self.train_dataloader) * getattr(
            worker.config, "update_per_rollout", 1
        )
        num_warmup_steps = int(worker.config.warmup_ratio * num_training_steps)

        return get_scheduler(
            worker.config.scheduler,
            worker.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
