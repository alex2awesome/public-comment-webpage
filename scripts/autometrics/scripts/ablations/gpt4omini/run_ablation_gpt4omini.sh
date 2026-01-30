#!/bin/bash

#SBATCH --job-name=gpt4omini_ablation
#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --partition=jag-lo
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=logs/gpt4omini_ablation_%j.out
#SBATCH --error=logs/gpt4omini_ablation_%j.err
#SBATCH -x jagupard[19-20,26-31]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Usage (single ablation run):
#   export DATASET_NAME="EvalGenProduct"
#   export TARGET_MEASURE="quality"
#   export SEED="42"
#   export METRICBANK_MODE="full"                # full|existing_only|generated_only
#   export K="30"                                 # optional, e.g., 5|10|20|30
#   export N="5"                                  # optional, e.g., 1|3|5|10
#   export NO_METRIC_CARDS="false"               # true|false
#   export FORCE_REINDEX="false"                 # true|false (use true for desc/existing/generated)
#   export OUTPUT_ROOT="results/ablations/gpt4omini"
#   sbatch scripts/ablations/gpt4omini/run_ablation_gpt4omini.sh

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

cd /nlp/scr2/nlp/personal-rm/autometrics

set -euo pipefail

if [ -z "${DATASET_NAME:-}" ]; then echo "ERROR: DATASET_NAME is required"; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then echo "ERROR: TARGET_MEASURE is required"; exit 1; fi
if [ -z "${SEED:-}" ]; then echo "ERROR: SEED is required"; exit 1; fi
if [ -z "${OPENAI_API_KEY:-}" ]; then echo "ERROR: OPENAI_API_KEY is required"; exit 1; fi

MODEL_NAME=${MODEL_NAME:-"openai/gpt-4o-mini"}
API_BASE=${API_BASE:-"https://api.openai.com/v1"}
METRICBANK_MODE=${METRICBANK_MODE:-"full"}
K=${K:-""}
N=${N:-""}
NO_METRIC_CARDS=${NO_METRIC_CARDS:-"false"}
FORCE_REINDEX=${FORCE_REINDEX:-"false"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/main_ablations/gpt4omini"}

mkdir -p logs

ABLA_TAG="${METRICBANK_MODE}"
if [ -n "$K" ]; then ABLA_TAG="${ABLA_TAG}_k${K}"; fi
if [ -n "$N" ]; then ABLA_TAG="${ABLA_TAG}_n${N}"; fi
if [ "$NO_METRIC_CARDS" = "true" ]; then ABLA_TAG="${ABLA_TAG}_desc"; fi

OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}/${ABLA_TAG}"
mkdir -p "$OUT_DIR"

echo "Launching ablation (GPT-4o-mini)"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Seed:    $SEED"
echo "Mode:    $METRICBANK_MODE"
echo "k:       ${K:-default}"
echo "n:       ${N:-default}"
echo "desc:    $NO_METRIC_CARDS"
echo "reindex: $FORCE_REINDEX"
echo "Model:   $MODEL_NAME"
echo "API:     $API_BASE"
echo "Out:     $OUT_DIR"

export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_ablation_gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/ablation_gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"

if [ -f "$OUT_DIR/score_pearson_${SEED}.txt" ] && [ -f "$OUT_DIR/log_${SEED}.json" ]; then
  echo "Results already exist in $OUT_DIR for seed $SEED. Skipping."
  exit 0
fi

PY_ARGS=(
  analysis/ablations/run_autometrics_ablation.py
  "$DATASET_NAME"
  "$TARGET_MEASURE"
  "$SEED"
  "$OUT_DIR"
  --model-name "$MODEL_NAME"
  --api-base "$API_BASE"
  --metricbank "$METRICBANK_MODE"
)

if [ -n "$K" ]; then PY_ARGS+=( --k "$K" ); fi
if [ -n "$N" ]; then PY_ARGS+=( --n "$N" ); fi
if [ "$NO_METRIC_CARDS" = "true" ]; then PY_ARGS+=( --no-metric-cards ); fi
if [ "$FORCE_REINDEX" = "true" ]; then PY_ARGS+=( --force-reindex ); fi

python "${PY_ARGS[@]}"

STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "✅ Ablation completed"
else
  echo "❌ Ablation failed with status $STATUS"
fi


