torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    rollout.train.path=Chenmien/OpenReasonerZero \
    rollout.train.prompts_per_rollout=128 \
    rollout.train.responses_per_prompt=64 \
    rollout.train.sampling_params.max_new_tokens=8192 \
    rollout.test.path=Chenmien/OlympiadBench \
    rollout.env_path=envs/orz.py \
    actor.model_name=Qwen/Qwen2.5-7B \
    actor.cp_size=2 \
    actor.max_length_per_device=8192 \
    actor.freeze_steps=4 \
    critic.model_name=Qwen/Qwen2.5-7B \
    critic.cp_size=2 \
    critic.max_length_per_device=8192 \
    adv.estimator=gae \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen2.5-7b-ppo \
    trainer.total_steps=512 \
    trainer.test_freq=8 \
    trainer.save_freq=32