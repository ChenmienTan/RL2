#!/usr/bin/env bash
set -euo pipefail

# Cluster-friendly one-click launch script for the phase-1 shared-policy
# multi-agent countdown example. Override any variable below via env vars.

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

TRAIN_PATH="${TRAIN_PATH:-train@Chenmien/Countdown}"
TEST_PATH="${TEST_PATH:-test@Chenmien/Countdown}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"

PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-128}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_LENGTH_PER_DEVICE="${MAX_LENGTH_PER_DEVICE:-8192}"

REWARD_MODE="${REWARD_MODE:-team}"
AGENT_ORDER="${AGENT_ORDER:-env_driven}"
TOTAL_STEPS="${TOTAL_STEPS:-512}"
TEST_FREQ="${TEST_FREQ:-8}"
SAVE_FREQ="${SAVE_FREQ:-32}"
PROJECT="${PROJECT:-Countdown}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-3b_multi_agent_reinforce}"
STOP_TOKENS="${STOP_TOKENS:-['</answer>']}"

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
    "rollout.train.sampling_params.stop=${STOP_TOKENS}" \
    rollout.test.path="${TEST_PATH}" \
    rollout.env_path=envs/multi_agent_countdown.py \
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
