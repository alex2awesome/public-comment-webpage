#!/bin/bash

# Orchestrated data scaling run: ALL datasets on sphinx TP=2 (server) + workers

set -euo pipefail

cd /nlp/scr2/nlp/personal-rm/autometrics/scripts/data_scaling/qwen/

SEEDS=${SEEDS:-"42 43 44 45 46"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8691}
TP=${TP:-2}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/data_scaling/autometrics/qwen"}
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}
TRAIN_SIZES=${TRAIN_SIZES:-"5 10 20 40 80 160 320 640"}

ALL_DATASETS=true \
SEEDS="${SEEDS}" \
MODEL_PATH="${MODEL_PATH}" \
PORT="${PORT}" \
TP="${TP}" \
DTYPE="${DTYPE}" \
MEM_FRACTION="${MEM_FRACTION}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OPENAI_API_KEY="${OPENAI_API_KEY}" \
TRAIN_SIZES="${TRAIN_SIZES}" \
sbatch --export=ALL launch_qwen_and_submit_scaling_remote_sphinx2.sh | tee /dev/fd/2


