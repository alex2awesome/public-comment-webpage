#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_server
#SBATCH --output=logs/qwen_server_%j.out
#SBATCH --error=logs/qwen_server_%j.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Persistent Qwen3-32B server using sglang.
#
# Usage:
#   export PORT=8123                        # optional (default: hashed per DATASET_NAME+TARGET_MEASURE or 8000)
#   export MODEL_PATH="Qwen/Qwen3-32B"     # optional
#   export HOST="0.0.0.0"                 # optional
#   export TP=1                            # optional tensor parallelism
#   export DTYPE="bfloat16"               # optional
#   export MEM_FRACTION="0.8"             # optional
#   sbatch scripts/ablations/qwen/launch_qwen_server.sh

set -euo pipefail

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

mkdir -p logs

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
TP=${TP:-"1"}
DTYPE=${DTYPE:-"bfloat16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}

echo "Starting Qwen server: ${MODEL_PATH} on ${HOST}:${PORT} (tp=${TP}, dtype=${DTYPE})"
echo "API base will be: http://${HOST}:${PORT}/v1"

# Run in foreground so SLURM keeps the process alive
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port ${PORT} \
  --host ${HOST} \
  --tp ${TP} \
  --dtype ${DTYPE} \
  --mem-fraction-static ${MEM_FRACTION} \
  --trust-remote-code


