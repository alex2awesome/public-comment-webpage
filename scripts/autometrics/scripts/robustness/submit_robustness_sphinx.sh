#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --partition=sphinx
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=robust_sphinx
#SBATCH --output=/nlp/scr2/nlp/personal-rm/autometrics/scripts/robustness/logs/robust_sphinx_%j.out
#SBATCH --error=/nlp/scr2/nlp/personal-rm/autometrics/scripts/robustness/logs/robust_sphinx_%j.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Unified GPU robustness submitter for sphinx.
# MODE: BEST | AUTOMETRICS

set -euo pipefail

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
set +u
conda activate autometrics
set -u

cd /nlp/scr2/nlp/personal-rm/autometrics

MODE=${MODE:-"BEST"}
SEED=${SEED:-42}

mkdir -p scripts/robustness/logs

case "$MODE" in
  BEST)
    if [ -z "${CSV_PATH:-}" ]; then echo "❌ CSV_PATH required"; exit 1; fi
    if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required"; exit 1; fi
    if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required"; exit 1; fi
    echo "[SPHINX] Running BEST Existing on GPU"
    CSV_BASENAME="$(basename "${CSV_PATH}" .csv)"
    CACHE_TAG="robust_best_${DATASET_NAME}_${TARGET_MEASURE}_${CSV_BASENAME}_seed${SEED}"
    export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/${CACHE_TAG}"
    export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/${CACHE_TAG}"
    mkdir -p "$(dirname "${DSPY_CACHEDIR}")" "$(dirname "${AUTOMETRICS_CACHE_DIR}")" 2>/dev/null || true
    # Rename job to include dataset/measure/csv
    scontrol update JobId=${SLURM_JOB_ID} JobName="best_${DATASET_NAME}_${TARGET_MEASURE}_${CSV_BASENAME}" >/dev/null 2>&1 || true
    python analysis/robustness/run_robustness.py \
      --csv "${CSV_PATH}" \
      --mode best_existing \
      --dataset "${DATASET_NAME}" \
      --measure "${TARGET_MEASURE}" \
      --seed "${SEED}" \
      ${METRIC_CLASS:+--metric-class "${METRIC_CLASS}"} \
      ${METRIC_NAME:+--metric-name "${METRIC_NAME}"} \
      ${OUTPUT_CSV:+--output "${OUTPUT_CSV}"}
    ;;
  AUTOMETRICS)
    if [ -z "${CSV_PATH:-}" ]; then echo "❌ CSV_PATH required"; exit 1; fi
    if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required"; exit 1; fi
    if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required"; exit 1; fi
    echo "[SPHINX] Running Autometrics Static Regression on GPU (per CSV)"
    CSV_BASENAME="$(basename "${CSV_PATH}" .csv)"
    CACHE_TAG="robust_autometrics_${DATASET_NAME}_${TARGET_MEASURE}_${CSV_BASENAME}_seed${SEED}"
    export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/${CACHE_TAG}"
    export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/${CACHE_TAG}"
    mkdir -p "$(dirname "${DSPY_CACHEDIR}")" "$(dirname "${AUTOMETRICS_CACHE_DIR}")" 2>/dev/null || true
    # Rename job to include dataset/measure/csv
    scontrol update JobId=${SLURM_JOB_ID} JobName="autometrics_${DATASET_NAME}_${TARGET_MEASURE}_${CSV_BASENAME}" >/dev/null 2>&1 || true
    # Map MODEL_NAME to llm-model flag for family inference (optional)
    MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}
    MODEL_FOR_SCRIPT="${MODEL_NAME}"
    case "${MODEL_NAME}" in
      "litellm_proxy/Qwen/Qwen3-32B"|"Qwen/Qwen3-32B"|"qwen3_32b") MODEL_FOR_SCRIPT="qwen3_32b";;
      "openai/gpt-4o-mini"|"gpt4o_mini") MODEL_FOR_SCRIPT="gpt4o_mini";;
      "openai/gpt-5-mini"|"gpt5_mini") MODEL_FOR_SCRIPT="gpt5_mini";;
      "litellm_proxy/meta-llama/Llama-3.3-70B-Instruct"|"llama3_70b") MODEL_FOR_SCRIPT="llama3_70b";;
      *) :;;
    esac
    python analysis/robustness/run_robustness.py \
      --csv "${CSV_PATH}" \
      --mode autometrics \
      --dataset "${DATASET_NAME}" \
      --measure "${TARGET_MEASURE}" \
      --seed "${SEED}" \
      --llm-model "${MODEL_FOR_SCRIPT}" \
      ${AUTOMETRICS_FAMILY:+--autometrics-family "${AUTOMETRICS_FAMILY}"} \
      ${METRIC_NAME:+--metric-name "${METRIC_NAME}"} \
      ${OUTPUT_CSV:+--output "${OUTPUT_CSV}"}
    ;;
  *)
    echo "❌ Unknown MODE='$MODE' (use BEST | AUTOMETRICS)"; exit 1;;
esac


