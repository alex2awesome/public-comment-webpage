#!/bin/bash

# Orchestrated main run: RealHumanEval/accepted on sphinx2 TP=2

set -euo pipefail

cd /nlp/scr2/nlp/personal-rm/autometrics/scripts/main_autometrics/qwen/

DATASET_NAME=${DATASET_NAME:-"RealHumanEval"}
TARGET_MEASURE=${TARGET_MEASURE:-"accepted"}
SEEDS=${SEEDS:-"42 43 44 45 46"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8956}
TP=${TP:-2}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/main_runs/autometrics/qwen_run2"}
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

echo "Submitting main orchestrator for ${DATASET_NAME}/${TARGET_MEASURE} (SEEDS=[${SEEDS}]) on sphinx2 TP=${TP}"

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
sbatch --export=ALL launch_qwen_and_submit_main_remote_sphinx2.sh | tee /dev/fd/2


