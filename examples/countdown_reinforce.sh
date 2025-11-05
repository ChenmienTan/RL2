torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=train@Chenmien/Countdown \
    train_data.experience.responses_per_prompt=4 \
    train_data.experience.max_new_tokens=1024 \
    test_data.path=test@Chenmien/Countdown \
    actor.model_name=Qwen/Qwen2.5-3B-Instruct \
    actor.max_length_per_device=8192 \
    rollout.train_prompts_per_rollout=128 \
    "rollout.train_sampling_params.stop=['</answer>']" \
    rollout.env_path=envs/countdown.py \
    trainer.project=Countdown \
    trainer.experiment_name=qwen2.5-3b_reinforce \
    trainer.total_steps=512 \
    trainer.test_freq=8 \
    trainer.save_freq=32