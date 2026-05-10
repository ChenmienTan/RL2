#!/usr/bin/env bash
#SBATCH --job-name=rl2-ma-countdown
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit example:
# sbatch \
#   --nodes=2 \
#   --gres=gpu:8 \
#   --job-name=rl2-ma-countdown \
#   examples/multi_agent_countdown_reinforce.slurm
#
# Override training settings without editing the file:
# sbatch --export=ALL,TRAIN_PATH=/path/train.jsonl,TEST_PATH=/path/test.jsonl,MODEL_NAME=Qwen/Qwen2.5-7B-Instruct examples/multi_agent_countdown_reinforce.slurm

ROOT_DIR="${ROOT_DIR:-$SLURM_SUBMIT_DIR}"
cd "${ROOT_DIR}"

mkdir -p logs

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-$SLURM_NNODES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

TRAIN_PATH="${TRAIN_PATH:-train@Chenmien/Countdown}"
TEST_PATH="${TEST_PATH:-test@Chenmien/Countdown}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-3b_multi_agent_countdown}"
PROJECT="${PROJECT:-Countdown}"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "TRAIN_PATH=${TRAIN_PATH}"
echo "TEST_PATH=${TEST_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
    bash -lc '
        set -euo pipefail
        export NODE_RANK="${SLURM_NODEID}"
        export MASTER_ADDR="'"${MASTER_ADDR}"'"
        export MASTER_PORT="'"${MASTER_PORT}"'"
        export NNODES="'"${NNODES}"'"
        export NPROC_PER_NODE="'"${NPROC_PER_NODE}"'"
        export TRAIN_PATH="'"${TRAIN_PATH}"'"
        export TEST_PATH="'"${TEST_PATH}"'"
        export MODEL_NAME="'"${MODEL_NAME}"'"
        export EXPERIMENT_NAME="'"${EXPERIMENT_NAME}"'"
        export PROJECT="'"${PROJECT}"'"
        bash examples/multi_agent_countdown_reinforce.sh
    '
