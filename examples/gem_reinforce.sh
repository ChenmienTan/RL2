torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    rollout.train.path=Chenmien/EmptyDataset \
    rollout.train.prompts_per_rollout=64 \
    rollout.train.sampling_params.max_new_tokens=1024 \
    rollout.test.path=Chenmien/EmptyDataset \
    rollout.test.prompts_per_rollout=64 \
    rollout.env_path=envs/gem.py \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.max_length_per_device=8192 \
    adv.norm_var=true \
    trainer.project=GEM \
    trainer.experiment_name=letter-counting_qwen3-1.7b_reinforce \
    trainer.total_steps=512 \
    trainer.save_freq=64