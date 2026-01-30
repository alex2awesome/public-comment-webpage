#!/bin/bash

# Example launcher: SimpEval with orchestrated mode on sphinx (server TP=2)
# Usage:
#   bash scripts/ablations/qwen/launchers/simpeval_score_orchestrated_sphinx2.sh

set -euo pipefail

cd /nlp/scr2/nlp/personal-rm/autometrics/scripts/ablations/qwen/

# Core experiment setup
DATASET_NAME="SimpEval"
TARGET_MEASURE=${TARGET_MEASURE:-"score"}
SEEDS=${SEEDS:-"42 43 44 45 46"}

# Run the full ablation suite
FULL_SUITE=${FULL_SUITE:-"true"}

# Optional single-run overrides (used only if FULL_SUITE != true)
METRICBANK_MODE=${METRICBANK_MODE:-"full"}
K=${K:-""}
N=${N:-""}
NO_METRIC_CARDS=${NO_METRIC_CARDS:-"false"}
FORCE_REINDEX=${FORCE_REINDEX:-"false"}
RESIZED=${RESIZED:-"false"}                  # true | false

# Qwen server options (sphinx node)
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8867}
TP=${TP:-2}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}

# Where the remote ablation results go
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/qwen_remote_run2"}

# OPENAI_API_KEY is required by downstream scripts; server ignores the key value
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

echo "Submitting orchestrator for SimpleVal (SEEDS=[$SEEDS], FULL_SUITE=$FULL_SUITE) on sphinx TP=${TP}"

DATASET_NAME="${DATASET_NAME}" \
TARGET_MEASURE="${TARGET_MEASURE}" \
SEEDS="${SEEDS}" \
FULL_SUITE="${FULL_SUITE}" \
METRICBANK_MODE="${METRICBANK_MODE}" \
K="${K}" \
N="${N}" \
NO_METRIC_CARDS="${NO_METRIC_CARDS}" \
FORCE_REINDEX="${FORCE_REINDEX}" \
MODEL_PATH="${MODEL_PATH}" \
PORT="${PORT}" \
TP="${TP}" \
DTYPE="${DTYPE}" \
MEM_FRACTION="${MEM_FRACTION}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OPENAI_API_KEY="${OPENAI_API_KEY}" \
RESIZED="${RESIZED}" \
sbatch --export=ALL launch_qwen_and_submit_remote_sphinx2.sh | tee /dev/fd/2
