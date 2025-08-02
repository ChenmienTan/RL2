torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    data.train_data_path=Chenmien/OpenReasonerZero \
    data.test_data_path=Chenmien/OlympiadBench \
    data.prompts_per_rollout=128 \
    data.responses_per_prompt=64 \
    actor.model_name=Qwen/Qwen2.5-7B \
    actor.sp_size=2 \
    actor.max_length_per_device=8192 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen2.5-7b-reinforce \
    trainer.test_freq=8 \
    trainer.save_freq=32