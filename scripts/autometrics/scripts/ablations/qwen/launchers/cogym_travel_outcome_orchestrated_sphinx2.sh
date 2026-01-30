#!/bin/bash

# Orchestrated launcher: CoGymTravelOutcome with Qwen server (sphinx TP=2) + sphinx workers
# Submits ONE job that hosts the Qwen server and, once ready, submits ablation jobs back to it.
#
# Usage:
#   bash scripts/ablations/qwen/launchers/cogym_travel_outcome_orchestrated_sphinx2.sh
#
# Customize below as needed.

set -euo pipefail

# Ensure we run from orchestrator script directory so relative paths resolve
cd /nlp/scr2/nlp/personal-rm/autometrics/scripts/ablations/qwen/

# Core experiment setup
DATASET_NAME=${DATASET_NAME:-"CoGymTravelOutcome"}
TARGET_MEASURE=${TARGET_MEASURE:-"outcomeRating"}
SEEDS=${SEEDS:-"42 43 44 45 46"}

# Run the full ablation suite in the specified order (k desc; n desc; no-metric-cards; generated_only; existing_only)
FULL_SUITE=${FULL_SUITE:-"true"}

# Optional single-run overrides (used only if FULL_SUITE != true)
METRICBANK_MODE=${METRICBANK_MODE:-"full"}   # full | existing_only | generated_only
K=${K:-""}                                   # 5 | 10 | 20 | 30
N=${N:-""}                                   # 1 | 3 | 5 | 10
NO_METRIC_CARDS=${NO_METRIC_CARDS:-"false"}  # true | false
FORCE_REINDEX=${FORCE_REINDEX:-"false"}      # true | false
RESIZED=${RESIZED:-"true"}                  # true | false

# Qwen server options (sphinx node, TP=2) — ensure different port from EvalGen (8123)
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8219}
TP=${TP:-2}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}

# Where the remote ablation results go (on the sphinx runs)
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/qwen_remote_run2"}

# OPENAI_API_KEY is required by downstream scripts; Qwen server ignores the key value.
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

echo "Submitting orchestrator for CoGymTravelOutcome (SEEDS=[$SEEDS], FULL_SUITE=$FULL_SUITE) on sphinx TP=${TP}"

# Pass values with spaces safely by exporting via environment and using --export=ALL
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


