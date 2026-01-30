#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --partition=john-lo
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=dna_eval_robustness
#SBATCH --output=/nlp/scr2/nlp/personal-rm/autometrics/scripts/robustness/logs/dna_eval_robustness_%j.out
#SBATCH --error=/nlp/scr2/nlp/personal-rm/autometrics/scripts/robustness/logs/dna_eval_robustness_%j.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# CPU-only robustness runner for DNAEval on unified robustness CSVs.
# Required env vars:
#   CSV_PATH           Path to the input CSV (unified columns: input, model_output, ref1...)
#   DATASET_NAME       Dataset key (e.g., SimpEval, HelpSteer2, RealHumanEval, CoGymTravelOutcome)
#   TARGET_MEASURE     Measure key (e.g., score, helpfulness, accepted, outcomeRating)
# Optional env vars:
#   API_BASE           OpenAI-compatible base URL (default: http://sphinx3.stanford.edu:8544/v1)
#   LLM_MODEL          Friendly or raw: litellm_proxy/Qwen/Qwen3-32B | qwen3_32b | openai/gpt-4o-mini | ... (default: litellm_proxy/Qwen/Qwen3-32B)
#   SEED               Default: 42
#   METRIC_NAME        Name for output column (default: auto)
#   OUTPUT_CSV         If set, write results to this file instead of in-place
#   OPENAI_API_KEY     Required by downstream libs; value is ignored by proxy servers (default: None)

set -euo pipefail

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
set +u
conda activate autometrics
set -u

cd /nlp/scr2/nlp/personal-rm/autometrics

mkdir -p scripts/robustness/logs

if [ -z "${CSV_PATH:-}" ]; then echo "❌ CSV_PATH required"; exit 1; fi
if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required"; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required"; exit 1; fi

API_BASE=${API_BASE:-"http://sphinx3.stanford.edu:8544/v1"}
LLM_MODEL=${LLM_MODEL:-"litellm_proxy/Qwen/Qwen3-32B"}
SEED=${SEED:-42}
OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

# Map friendly/raw model IDs to run_robustness.py choices
MODEL_FOR_SCRIPT="${LLM_MODEL}"
case "${LLM_MODEL}" in
  "litellm_proxy/Qwen/Qwen3-32B"|"Qwen/Qwen3-32B"|"qwen3_32b")
    MODEL_FOR_SCRIPT="qwen3_32b";;
  "openai/gpt-4o-mini"|"gpt4o_mini")
    MODEL_FOR_SCRIPT="gpt4o_mini";;
  "openai/gpt-5-mini"|"gpt5_mini")
    MODEL_FOR_SCRIPT="gpt5_mini";;
  "litellm_proxy/meta-llama/Llama-3.3-70B-Instruct"|"llama3_70b")
    MODEL_FOR_SCRIPT="llama3_70b";;
  *)
    :;;
esac

export OPENAI_API_KEY

echo "Running DNAEval robustness"
echo "CSV:     $CSV_PATH"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Model:   $LLM_MODEL (mapped -> ${MODEL_FOR_SCRIPT})"
echo "API:     $API_BASE"
echo "Seed:    $SEED"

# Per-job unique caches (include dataset/measure/csv basename/seed)
CSV_BASENAME="$(basename "$CSV_PATH" .csv)"
CACHE_TAG="robust_dna_${DATASET_NAME}_${TARGET_MEASURE}_${CSV_BASENAME}_seed${SEED}"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/${CACHE_TAG}"
export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/${CACHE_TAG}"
mkdir -p "$(dirname "${DSPY_CACHEDIR}")" "$(dirname "${AUTOMETRICS_CACHE_DIR}")" 2>/dev/null || true

# Rename SLURM job to include dataset/measure/csv
scontrol update JobId=${SLURM_JOB_ID} JobName="dna_${DATASET_NAME}_${TARGET_MEASURE}_${CSV_BASENAME}" >/dev/null 2>&1 || true

PY_ARGS=(
  analysis/robustness/run_robustness.py
  --csv "$CSV_PATH"
  --mode dna_eval
  --dataset "$DATASET_NAME"
  --measure "$TARGET_MEASURE"
  --llm-model "$MODEL_FOR_SCRIPT"
  --api-base "$API_BASE"
  --seed "$SEED"
)

if [ -n "${METRIC_NAME:-}" ]; then
  PY_ARGS+=( --metric-name "$METRIC_NAME" )
fi

if [ -n "${OUTPUT_CSV:-}" ]; then
  PY_ARGS+=( --output "$OUTPUT_CSV" )
fi

python "${PY_ARGS[@]}"


