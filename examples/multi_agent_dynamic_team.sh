#!/usr/bin/env bash
set -euo pipefail

# Dynamic Team Training Script
# Configure different team types and sizes

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Team configuration
TEAM_TYPE="${TEAM_TYPE:-software}"  # software, research, creative, custom
NUM_RESEARCHERS="${NUM_RESEARCHERS:-2}"  # For research team
CUSTOM_ROLES="${CUSTOM_ROLES:-}"  # For custom team (comma-separated)

# Calculate number of agents based on team type
case "${TEAM_TYPE}" in
    software)
        NUM_AGENTS=4
        ;;
    research)
        NUM_AGENTS=$((NUM_RESEARCHERS + 2))  # researchers + reviewer + editor
        ;;
    creative)
        NUM_AGENTS=4
        ;;
    custom)
        if [ -z "${CUSTOM_ROLES}" ]; then
            echo "Error: CUSTOM_ROLES must be set for custom team type"
            exit 1
        fi
        NUM_AGENTS=$(echo "${CUSTOM_ROLES}" | tr ',' '\n' | wc -l)
        ;;
    *)
        echo "Error: Unknown TEAM_TYPE: ${TEAM_TYPE}"
        echo "Available types: software, research, creative, custom"
        exit 1
        ;;
esac

# Dataset
TRAIN_PATH="${TRAIN_PATH:-train@your_org/Tasks}"
TEST_PATH="${TEST_PATH:-test@your_org/Tasks}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"

# Scale prompts with team size
PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-$((NPROC_PER_NODE * NUM_AGENTS * 2))}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_LENGTH_PER_DEVICE="${MAX_LENGTH_PER_DEVICE:-8192}"

REWARD_MODE="${REWARD_MODE:-team}"
AGENT_ORDER="${AGENT_ORDER:-env_driven}"
TOTAL_STEPS="${TOTAL_STEPS:-512}"
TEST_FREQ="${TEST_FREQ:-8}"
SAVE_FREQ="${SAVE_FREQ:-32}"
PROJECT="${PROJECT:-DynamicTeam}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TEAM_TYPE}_team_${NUM_AGENTS}agents}"

echo "========================================="
echo "Dynamic Team Training"
echo "========================================="
echo "Team type: ${TEAM_TYPE}"
echo "Number of agents: ${NUM_AGENTS}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Prompts per rollout: ${PROMPTS_PER_ROLLOUT}"
if [ "${TEAM_TYPE}" = "research" ]; then
    echo "Researchers: ${NUM_RESEARCHERS}"
fi
if [ "${TEAM_TYPE}" = "custom" ]; then
    echo "Custom roles: ${CUSTOM_ROLES}"
fi
echo "========================================="

# Export configuration for Python script
export TEAM_TYPE
export NUM_RESEARCHERS
export CUSTOM_ROLES

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
    rollout.env_path=envs/multi_agent_dynamic_team.py \
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
