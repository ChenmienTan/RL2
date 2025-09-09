torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=null \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=1 \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.sp_size=2 \
    actor.max_length_per_device=8192 \
    actor.tis_coef=2.0 \
    actor.update_per_rollout=2 \
    actor.warmup_ratio=0.0 \
    actor.lr=1e-6 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.train_sampling_params.temperature=1.0 \
    rollout.env_path=envs/gem_env.py \
    rollout.max_turns=16 \
    adv.estimator=reinforce \
    adv.global_norm=true \
    adv.norm_var=true \
    trainer.project=GEM-RL \
    trainer.experiment_name=gem-qwen3-1.7b-reinforce \
    trainer.n_epochs=500 \
    trainer.test_freq=9999999 \
    trainer.save_freq=64 \
    trainer.use_wandb=true