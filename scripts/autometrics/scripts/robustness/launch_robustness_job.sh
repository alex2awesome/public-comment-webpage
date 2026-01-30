#!/bin/bash

# Generic launcher for robustness runs.
# Usage examples:
#  MODE=llm_judge DATASET_NAME=SimpEval TARGET_MEASURE=score CSV_PATH=... API_BASE=http://host:port/v1 bash scripts/robustness/launch_robustness_job.sh
#  MODE=best_existing DATASET_NAME=SimpEval TARGET_MEASURE=score CSV_PATH=... bash scripts/robustness/launch_robustness_job.sh
#  MODE=dna_eval DATASET_NAME=SimpEval TARGET_MEASURE=score CSV_PATH=... API_BASE=http://host:port/v1 bash scripts/robustness/launch_robustness_job.sh
#  MODE=autometrics DATASET_NAME=SimpEval TARGET_MEASURE=score OUTPUT_DIR=... API_BASE=http://host:port/v1 bash scripts/robustness/launch_robustness_job.sh

set -euo pipefail

MODE=${MODE:-"llm_judge"}
LLM_MODEL=${LLM_MODEL:-"qwen3_32b"}
SEED=${SEED:-42}
PARTITION_CPU=${PARTITION_CPU:-"john-lo"}
TARGET_CLUSTER=${TARGET_CLUSTER:-"sc"} # sc | sphinx | jag

LOG_DIR="/nlp/scr2/nlp/personal-rm/autometrics/scripts/robustness/logs"
mkdir -p "${LOG_DIR}"

if [ "${MODE}" != "autometrics" ]; then
  if [ -z "${CSV_PATH:-}" ]; then echo "❌ CSV_PATH required"; exit 1; fi
fi
if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required"; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required"; exit 1; fi

submit_for_csv() {
  local csv="$1"
  local base=""
  if [ -n "${csv}" ]; then
    base="$(basename "${csv}" .csv)"
  fi
  local prefix="${MODE}_${DATASET_NAME}_${TARGET_MEASURE}"
  if [ -n "${base}" ]; then
    prefix="${prefix}_${base}"
  fi
  case "$MODE" in
    llm_judge)
      sbatch --partition="${PARTITION_CPU}" \
        --job-name="robust_llm_judge_${DATASET_NAME}_${TARGET_MEASURE}" \
        --output="${LOG_DIR}/${prefix}_%j.out" \
        --error="${LOG_DIR}/${prefix}_%j.err" \
        --export=ALL,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",API_BASE="${API_BASE:-}",LLM_MODEL="${LLM_MODEL}",SEED="${SEED}",METRIC_NAME="${METRIC_NAME:-}",OUTPUT_CSV="${OUTPUT_CSV:-}" \
        scripts/robustness/run_llm_judge_robustness.sh | tee /dev/fd/2
      ;;
    best_existing)
      case "${TARGET_CLUSTER}" in
        sc)
          sbatch --output="${LOG_DIR}/${prefix}_%j.out" --error="${LOG_DIR}/${prefix}_%j.err" --export=ALL,MODE=BEST,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",SEED="${SEED}",METRIC_CLASS="${METRIC_CLASS:-}",METRIC_NAME="${METRIC_NAME:-}",OUTPUT_CSV="${OUTPUT_CSV:-}" scripts/robustness/submit_robustness_sc.sh ;;
        sphinx)
          sbatch --output="${LOG_DIR}/${prefix}_%j.out" --error="${LOG_DIR}/${prefix}_%j.err" --export=ALL,MODE=BEST,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",SEED="${SEED}",METRIC_CLASS="${METRIC_CLASS:-}",METRIC_NAME="${METRIC_NAME:-}",OUTPUT_CSV="${OUTPUT_CSV:-}" scripts/robustness/submit_robustness_sphinx.sh ;;
        jag)
          sbatch --output="${LOG_DIR}/${prefix}_%j.out" --error="${LOG_DIR}/${prefix}_%j.err" --export=ALL,MODE=BEST,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",SEED="${SEED}",METRIC_CLASS="${METRIC_CLASS:-}",METRIC_NAME="${METRIC_NAME:-}",OUTPUT_CSV="${OUTPUT_CSV:-}" scripts/robustness/submit_robustness_jag.sh ;;
        *) echo "❌ Unknown TARGET_CLUSTER='${TARGET_CLUSTER}' (use sc|sphinx|jag)"; exit 1;;
      esac
      ;;
    dna_eval)
      sbatch --partition="${PARTITION_CPU}" \
        --job-name="robust_dna_${DATASET_NAME}_${TARGET_MEASURE}" \
        --output="${LOG_DIR}/${prefix}_%j.out" \
        --error="${LOG_DIR}/${prefix}_%j.err" \
        --export=ALL,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",API_BASE="${API_BASE:-}",LLM_MODEL="${LLM_MODEL}",SEED="${SEED}",METRIC_NAME="${METRIC_NAME:-}",OUTPUT_CSV="${OUTPUT_CSV:-}" \
        scripts/robustness/run_dna_eval_robustness.sh | tee /dev/fd/2
      ;;
    autometrics)
      case "${TARGET_CLUSTER}" in
        sc)
          sbatch --output="${LOG_DIR}/${prefix}_%j.out" --error="${LOG_DIR}/${prefix}_%j.err" --export=ALL,MODE=AUTOMETRICS,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",SEED="${SEED}",API_BASE="${API_BASE:-}",MODEL_NAME="${MODEL_NAME:-}",OPENAI_API_KEY="${OPENAI_API_KEY:-}" scripts/robustness/submit_robustness_sc.sh ;;
        sphinx)
          sbatch --output="${LOG_DIR}/${prefix}_%j.out" --error="${LOG_DIR}/${prefix}_%j.err" --export=ALL,MODE=AUTOMETRICS,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",SEED="${SEED}",API_BASE="${API_BASE:-}",MODEL_NAME="${MODEL_NAME:-}",OPENAI_API_KEY="${OPENAI_API_KEY:-}" scripts/robustness/submit_robustness_sphinx.sh ;;
        jag)
          sbatch --output="${LOG_DIR}/${prefix}_%j.out" --error="${LOG_DIR}/${prefix}_%j.err" --export=ALL,MODE=AUTOMETRICS,CSV_PATH="${csv}",DATASET_NAME="${DATASET_NAME}",TARGET_MEASURE="${TARGET_MEASURE}",SEED="${SEED}",API_BASE="${API_BASE:-}",MODEL_NAME="${MODEL_NAME:-}",OPENAI_API_KEY="${OPENAI_API_KEY:-}" scripts/robustness/submit_robustness_jag.sh ;;
        *) echo "❌ Unknown TARGET_CLUSTER='${TARGET_CLUSTER}' (use sc|sphinx|jag)"; exit 1;;
      esac
      ;;
    *)
      echo "❌ Unknown MODE='$MODE'. Use one of: llm_judge | best_existing | dna_eval | autometrics" >&2
      exit 1
      ;;
  esac
}

if [ -d "${CSV_PATH:-}" ]; then
  for f in "${CSV_PATH}"/*.csv; do
    [ -e "$f" ] || continue
    submit_for_csv "$f"
  done
else
  submit_for_csv "${CSV_PATH:-}"
fi


