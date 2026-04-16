#!/usr/bin/env bash
set -euo pipefail

# Multi-Agent Debate Training Script
# Two agents debate a topic: Pro vs Con

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Dataset: Create your own debate topics dataset
# Format: {"topic": "AI will replace human jobs"}
TRAIN_PATH="${TRAIN_PATH:-train@your_org/DebateTopics}"
TEST_PATH="${TEST_PATH:-test@your_org/DebateTopics}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"

PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-16}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"  # Longer for debate arguments
MAX_LENGTH_PER_DEVICE="${MAX_LENGTH_PER_DEVICE:-8192}"

REWARD_MODE="${REWARD_MODE:-team}"  # Both agents rewarded equally
AGENT_ORDER="${AGENT_ORDER:-env_driven}"
TOTAL_STEPS="${TOTAL_STEPS:-512}"
TEST_FREQ="${TEST_FREQ:-8}"
SAVE_FREQ="${SAVE_FREQ:-32}"
PROJECT="${PROJECT:-Debate}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-multi_agent_debate}"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m RL2.trainer.ppo \
    rollout.train.path="${TRAIN_PATH}" \
    rollout.train.prompts_per_rollout="${PROMPTS_PER_ROLLOUT}" \
    rollout.train.responses_per_prompt="${RESPONSES_PER_PROMPT}" \
    rollout.train.sampling_params.max_new_tokens="${MAX_NEW_TOKENS}" \
    rollout.test.path="${TEST_PATH}" \
    rollout.env_path=envs/multi_agent_debate.py \
    rollout.multi_agent.enabled=true \
    rollout.multi_agent.shared_policy=true \
    rollout.multi_agent.reward_mode="${REWARD_MODE}" \
    rollout.multi_agent.agent_order="${AGENT_ORDER}" \
    actor.model_name="${MODEL_NAME}" \
    actor.max_length_per_device="${MAX_LENGTH_PER_DEVICE}" \
    trainer.project="${PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_steps="${TOTAL_STEPS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}"
