#!/bin/bash

# Orchestrated main run: ALL datasets on sphinx TP=2 (server) + sphinx workers

set -euo pipefail

cd /nlp/scr2/nlp/personal-rm/autometrics/scripts/main_autometrics/qwen/

DATASET_NAME="ALL"
SEEDS=${SEEDS:-"42 43 44 45 46"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8219}
TP=${TP:-2}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/main_runs/autometrics/qwen_run2"}
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

# Use the orchestrator's ALL mode and dataset list from the sphinx2 orchestrator script itself
ALL_DATASETS=true \
SEEDS="${SEEDS}" \
MODEL_PATH="${MODEL_PATH}" \
PORT="${PORT}" \
TP="${TP}" \
DTYPE="${DTYPE}" \
MEM_FRACTION="${MEM_FRACTION}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OPENAI_API_KEY="${OPENAI_API_KEY}" \
sbatch --export=ALL launch_qwen_and_submit_main_remote_sphinx2.sh | tee /dev/fd/2