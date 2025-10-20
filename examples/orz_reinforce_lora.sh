torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=64 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=Qwen/Qwen3-4B \
    actor.cp_size=2 \
    actor.max_length_per_device=8192 \
    actor.use_lora=true \
    actor.lora.r=16 \
    actor.lora.lora_alpha=32 \
    actor.lora.lora_dropout=0.05 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen2.5-7b-reinforce \
    trainer.test_freq=8 \
    trainer.save_freq=32