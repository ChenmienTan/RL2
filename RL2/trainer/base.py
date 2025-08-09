from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict
)
from transformers import (
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
import wandb
from RL2.utils.offloading import load_model_to_device

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

        num_training_steps = self.config.trainer.n_epochs * len(self.train_dataloader) * getattr(
            worker.config, "update_per_rollout", 1
        )
        num_warmup_steps = int(worker.config.warmup_ratio * num_training_steps)

        return get_cosine_schedule_with_warmup(
            worker.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def get_ckpt(self, workers, step):

        ckpt = {
            "step": step,
            "dataloader": self.train_dataloader.state_dict()
        }

        options = StateDictOptions(
            full_state_dict=False, cpu_offload=True
        )
        for idx, worker in enumerate(workers):

            load_model_to_device(worker, torch.cuda.current_device())
            worker_ckpt = {
                "model": get_model_state_dict(
                    worker.model, options=options
                ),
                "optimizer": worker.optimizer.state_dict(),
                "scheduler": worker.scheduler.state_dict()
            }
            load_model_to_device(worker, "cpu")
            ckpt.update({
                f"worker{idx}": worker_ckpt
            })

        return ckpt
    
    def load_ckpt(self, workers):

        if self.config.trainer.load_ckpt_from_dir is None:
            return 0

        ckpt = self.get_ckpt(workers, 0)
        dcp.load(
            ckpt,
            checkpoint_id=self.config.trainer.load_ckpt_from_dir
        )
        
        self.train_dataloader.load_state_dict(ckpt["dataloader"])
        for idx, worker in enumerate(workers):

            worker_ckpt = ckpt[f"worker{idx}"]
            load_model_to_device(worker, torch.cuda.current_device())
            set_model_state_dict(
                worker.model, worker_ckpt["model"]
            )
            load_model_to_device(worker, "cpu")
            worker.optimizer.load_state_dict(worker_ckpt["optimizer"])
            worker.scheduler.load_state_dict(worker_ckpt["scheduler"])

        return ckpt["step"]
    
    def save_ckpt(self, workers, step):

        if self.config.trainer.save_freq is None or step % self.config.trainer.save_freq != 0:
            return

        dcp.save(
            self.get_ckpt(workers, step),
            checkpoint_id=f"{self.config.trainer.save_dir}/step{step}"
        )

    def save_model(self, worker, rm=False):

        save_dir = f"{self.config.trainer.save_dir}/latest"
        options = StateDictOptions(
            full_state_dict=True, cpu_offload=True
        )
        # Ensure parameters are on the compute device before collecting state dict
        load_model_to_device(worker, torch.cuda.current_device())
        state_dict = get_model_state_dict(
            worker.model, options=options
        )
        # Move tensors back to CPU if offloading is enabled
        load_model_to_device(worker, "cpu")
        if dist.get_rank() == 0:

            worker.tokenizer.save_pretrained(save_dir)
            # unwrap the model
            model_to_save = worker.model.module
            if rm:
                # For RM, we load token classification model for simplicity 
                # but save sequence classification model for compatibility.
                with torch.device("meta"):
                    model_to_save = AutoModelForSequenceClassification.from_config(
                        model_to_save.config
                    )
            model_to_save.save_pretrained(
                save_dir, state_dict=state_dict
            )

        dist.barrier()