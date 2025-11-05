torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.experience.max_new_tokens=1024 \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.max_length_per_device=8192 \
    rollout.train_prompts_per_rollout=64 \
    rollout.test_prompts_per_prompt=64 \
    rollout.env_path=envs/gem.py \
    adv.global_norm=true \
    adv.norm_var=true \
    trainer.project=GEM \
    trainer.experiment_name=letter-counting_qwen3-1.7b_reinforce \
    trainer.total_steps=512 \
    trainer.save_freq=64