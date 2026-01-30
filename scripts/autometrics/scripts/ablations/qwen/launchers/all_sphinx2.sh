#!/bin/bash

# Orchestrated launcher: ALL datasets with one Qwen server (sphinx TP=2) + sphinx workers
# Submits ONE job that hosts the Qwen server and, once ready, submits ablation jobs back to it.
# Supports per-dataset filters via ONLY_DATASETS or DISABLE_DATASETS.
#
# Usage:
#   bash scripts/ablations/qwen/launchers/all_sphinx2.sh
#
# Customizable env vars (with defaults shown):
#   SEEDS="42 43 44 45 46"
#   FULL_SUITE="true"
#   DATASET_SPECS="CoGymTravelOutcome:outcomeRating:true EvalGenProduct:grade:true RealHumanEval:accepted:false Primock57:time_sec:false HelpSteer2:helpfulness:false SimpEval:score:false"
#   ONLY_DATASETS=""  # e.g., "CoGymTravelOutcome RealHumanEval"
#   DISABLE_DATASETS=""  # e.g., "EvalGenProduct"
#   MODEL_PATH="Qwen/Qwen3-32B" PORT=8123 TP=2 DTYPE=float16 MEM_FRACTION=0.8
#   OUTPUT_ROOT="results/ablations/qwen_remote"
#   OPENAI_API_KEY (if unset, script sets to "None")

set -euo pipefail

cd /nlp/scr2/nlp/personal-rm/autometrics/scripts/ablations/qwen/

SEEDS=${SEEDS:-"42 43 44 45 46"}
FULL_SUITE=${FULL_SUITE:-"true"}
DATASET_SPECS=${DATASET_SPECS:-"CoGymTravelOutcome:outcomeRating:true RealHumanEval:accepted:false Primock57:time_sec:false HelpSteer2:helpfulness:false SimpEval:score:false EvalGenProduct:grade:true"}
ONLY_DATASETS=${ONLY_DATASETS:-""}
DISABLE_DATASETS=${DISABLE_DATASETS:-""}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8123}
TP=${TP:-2}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/qwen_remote_run2"}
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

echo "Submitting orchestrator for ALL datasets (SEEDS=[${SEEDS}], FULL_SUITE=${FULL_SUITE}) on sphinx TP=${TP}"

DATASET_NAME="ALL" \
SEEDS="${SEEDS}" \
FULL_SUITE="${FULL_SUITE}" \
DATASET_SPECS="${DATASET_SPECS}" \
ONLY_DATASETS="${ONLY_DATASETS}" \
DISABLE_DATASETS="${DISABLE_DATASETS}" \
MODEL_PATH="${MODEL_PATH}" \
PORT="${PORT}" \
TP="${TP}" \
DTYPE="${DTYPE}" \
MEM_FRACTION="${MEM_FRACTION}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OPENAI_API_KEY="${OPENAI_API_KEY}" \
sbatch --export=ALL launch_qwen_and_submit_remote_sphinx2.sh | tee /dev/fd/2


