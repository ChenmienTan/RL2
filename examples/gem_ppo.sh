#!/bin/bash

# GEM Environment PPO Training Example for RL2
# Aligned with VeRL training configuration (only matching parameters)

# Configuration variables
n_gpus=8
batch_size=128
env=rg:letter_counting

# Run PPO training with GEM environment
torchrun \
    --nproc_per_node=$n_gpus \
    -m RL2.trainer.ppo \
    \
    trainer.project=gem \
    trainer.experiment_name=rl2-qwen3-1.7b-base-${env} \
    trainer.n_epochs=15 \
    trainer.test_freq=9999999 \
    trainer.save_freq=9999999 \
    trainer.disable_wandb=false \
    \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.lr=1e-6 \
    \
    rollout.use_gem_env=true \
    rollout.model_name=Qwen/Qwen3-1.7B-Base \
    rollout.tp_size=1 \
    \
    gem_env.env_id=${env} \
    gem_env.wrappers="" \
    gem_env.num_env=16 \
    gem_env.async_env=true \
    gem_env.prompt_template=qwen3_general \
    gem_env.rollout_batch_size=${batch_size}