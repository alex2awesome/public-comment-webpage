#!/bin/bash

# Orchestrated data scaling run: HelpSteer2/helpfulness on sphinx TP=2

set -euo pipefail

cd /nlp/scr2/nlp/personal-rm/autometrics/scripts/data_scaling/qwen/

DATASET_NAME=${DATASET_NAME:-"HelpSteer2"}
TARGET_MEASURE=${TARGET_MEASURE:-"helpfulness"}
SEEDS=${SEEDS:-"42 43 44 45 46"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8544}
TP=${TP:-2}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/data_scaling/autometrics/qwen"}
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}
TRAIN_SIZES=${TRAIN_SIZES:-"5 10 20 40 80 160 320 640"}

echo "Submitting scaling orchestrator for ${DATASET_NAME}/${TARGET_MEASURE} (SEEDS=[${SEEDS}]) on sphinx TP=${TP}"

DATASET_NAME="${DATASET_NAME}" \
TARGET_MEASURE="${TARGET_MEASURE}" \
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


