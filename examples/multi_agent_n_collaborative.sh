#!/usr/bin/env bash
set -euo pipefail

# N-Agent Collaborative Training Script
# Easily scale to any number of agents!

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Number of agents (easily configurable!)
NUM_AGENTS="${NUM_AGENTS:-3}"  # Try 3, 5, 10, or more!

# Policy mode (shared or independent)
SHARED_POLICY="${SHARED_POLICY:-true}"  # true=shared (default), false=independent

# Dataset
TRAIN_PATH="${TRAIN_PATH:-train@your_org/Problems}"
TEST_PATH="${TEST_PATH:-test@your_org/Problems}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"

# Important: Scale PROMPTS_PER_ROLLOUT with number of agents
# Rule: PROMPTS_PER_ROLLOUT >= NPROC_PER_NODE * NUM_AGENTS
PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-$((NPROC_PER_NODE * NUM_AGENTS * 2))}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_LENGTH_PER_DEVICE="${MAX_LENGTH_PER_DEVICE:-8192}"

REWARD_MODE="${REWARD_MODE:-team}"
AGENT_ORDER="${AGENT_ORDER:-env_driven}"
TOTAL_STEPS="${TOTAL_STEPS:-512}"
TEST_FREQ="${TEST_FREQ:-8}"
SAVE_FREQ="${SAVE_FREQ:-32}"
PROJECT="${PROJECT:-NAgentCollaborative}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${NUM_AGENTS}_agent_collaborative}"

echo "========================================="
echo "N-Agent Collaborative Training"
echo "========================================="
echo "Number of agents: ${NUM_AGENTS}"
echo "Policy mode: ${SHARED_POLICY}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Prompts per rollout: ${PROMPTS_PER_ROLLOUT}"
echo "========================================="

# Set NUM_AGENTS environment variable for the Python script
export NUM_AGENTS

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
    rollout.env_path=envs/multi_agent_n_collaborative.py \
    rollout.multi_agent.enabled=true \
    rollout.multi_agent.shared_policy="${SHARED_POLICY}" \
    rollout.multi_agent.reward_mode="${REWARD_MODE}" \
    rollout.multi_agent.agent_order="${AGENT_ORDER}" \
    actor.model_name="${MODEL_NAME}" \
    actor.max_length_per_device="${MAX_LENGTH_PER_DEVICE}" \
    trainer.project="${PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_steps="${TOTAL_STEPS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}"
