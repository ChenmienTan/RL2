poetry run torchrun \
    --nproc_per_node=2 \
    -m RL2.trainer.orpo \
    data.path=test.json \
    data.max_length=1024 \
    actor.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    actor.max_length_per_device=4096 \
    trainer.project=UltraFeedback \
    trainer.experiment_name=qwen-2_0_5_orpo