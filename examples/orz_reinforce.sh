torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.responses_per_prompt=64 \
    train_data.max_new_tokens=8192 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=Qwen/Qwen2.5-7B \
    actor.cp_size=2 \
    actor.max_length_per_device=8192 \
    rollout.train_prompts_per_rollout=128 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen2.5-7b-reinforce \
    trainer.total_steps=512 \
    trainer.test_freq=8 \
    trainer.save_freq=32