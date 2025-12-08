torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    rollout.train.path=train@Chenmien/Countdown \
    rollout.train.prompts_per_rollout=128 \
    rollout.train.responses_per_prompt=4 \
    rollout.train.sampling_params.max_new_tokens=1024 \
    "rollout.train.sampling_params.stop=['</answer>']" \
    rollout.train.apply_chat_template=false \
    rollout.test.path=test@Chenmien/Countdown \
    rollout.env_path=envs/countdown.py \
    actor.model_name=Qwen/Qwen2.5-3B-Instruct \
    actor.max_length_per_device=8192 \
    trainer.project=Countdown \
    trainer.experiment_name=qwen2.5-3b_reinforce \
    trainer.total_steps=512 \
    trainer.test_freq=8 \
    trainer.save_freq=32