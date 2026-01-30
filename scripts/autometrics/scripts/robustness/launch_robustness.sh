#!/bin/bash

set -euo pipefail

# Dead simple launcher
# Usage examples:
#   scripts/robustness/launch_robustness.sh Helpsteer2 LLMJudge cpu
#   scripts/robustness/launch_robustness.sh EvalGenProduct Best sphinx
#   scripts/robustness/launch_robustness.sh SimpEval DNAEval cpu
#   scripts/robustness/launch_robustness.sh RealHumanEval Autometrics jag

REPO_ROOT="/nlp/scr2/nlp/personal-rm/autometrics"
cd "${REPO_ROOT}"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <Dataset> <Mode> <Target>" >&2
  echo "  Dataset: Helpsteer2 | SimpEval | RealHumanEval | CoGymTravelOutcome | EvalGenProduct | Primock57" >&2
  echo "  Mode:    LLMJudge | DNAEval | Best | AutoMetrics" >&2
  echo "  Target:  cpu | sc | sphinx | jag" >&2
  exit 1
fi

dataset_token="$1"
mode_token="$2"
target_token="$3"

shopt -s nocasematch

# Map dataset token → DATASET_NAME, TARGET_MEASURE, CSV_DIR
DATASET_NAME=""
TARGET_MEASURE=""
CSV_DIR=""

case "${dataset_token}" in
  helpsteer2)
    DATASET_NAME="HelpSteer2"
    TARGET_MEASURE="helpfulness"
    CSV_DIR="${REPO_ROOT}/outputs/robustness/csvs/helpsteer2_helpfulness"
    ;;
  simpeval)
    DATASET_NAME="SimpEval"
    TARGET_MEASURE="score"
    CSV_DIR="${REPO_ROOT}/outputs/robustness/csvs/simpeval_score"
    ;;
  realhumaneval)
    DATASET_NAME="RealHumanEval"
    TARGET_MEASURE="accepted"
    CSV_DIR="${REPO_ROOT}/outputs/robustness/csvs/realhumaneval_accepted"
    ;;
  cogym|cogymtraveloutcome)
    DATASET_NAME="CoGymTravelOutcome"
    TARGET_MEASURE="outcomeRating"
    CSV_DIR="${REPO_ROOT}/outputs/robustness/csvs/cogymtraveloutcome_outcomerating"
    ;;
  evalgenproduct)
    DATASET_NAME="EvalGenProduct"
    TARGET_MEASURE="grade"
    CSV_DIR="${REPO_ROOT}/outputs/robustness/csvs/evalgenproduct_grade"
    ;;
  primock57)
    DATASET_NAME="Primock57"
    TARGET_MEASURE="time_sec"
    CSV_DIR="${REPO_ROOT}/outputs/robustness/csvs/primock57_time_sec"
    ;;
  *)
    echo "❌ Unknown dataset token: ${dataset_token}" >&2
    exit 1
    ;;
esac

# Map mode token → internal MODE
MODE=""
case "${mode_token}" in
  llmjudge|llm_judge)
    MODE="llm_judge" ;;
  dnaeval|dna|dna_eval)
    MODE="dna_eval" ;;
  best|best_existing)
    MODE="best_existing" ;;
  autometrics|auto)
    MODE="autometrics" ;;
  *)
    echo "❌ Unknown mode token: ${mode_token}" >&2
    exit 1
    ;;
esac

# Map target token → CPU vs GPU cluster
TARGET_CLUSTER=""
IS_CPU="false"
case "${target_token}" in
  cpu)
    IS_CPU="true" ;;
  sc)
    TARGET_CLUSTER="sc" ;;
  sphinx)
    TARGET_CLUSTER="sphinx" ;;
  jag|jagupard)
    TARGET_CLUSTER="jag" ;;
  *)
    echo "❌ Unknown target token: ${target_token}" >&2
    exit 1
    ;;
esac

# Defaults
SEED=${SEED:-42}
API_BASE=${API_BASE:-"http://sphinx3.stanford.edu:8544/v1"}
LLM_MODEL=${LLM_MODEL:-"litellm_proxy/Qwen/Qwen3-32B"}

echo "Dataset: ${DATASET_NAME} (${dataset_token})"
echo "Measure: ${TARGET_MEASURE}"
echo "Mode:    ${MODE} (${mode_token})"
echo "Target:  ${IS_CPU:+cpu}${TARGET_CLUSTER:+/gpu-${TARGET_CLUSTER}}"
if [ -n "${CSV_DIR}" ]; then echo "CSV dir: ${CSV_DIR}"; fi

if [ "${MODE}" = "llm_judge" ] || [ "${MODE}" = "dna_eval" ]; then
  if [ "${IS_CPU}" != "true" ]; then
    echo "⚠️  ${MODE} is recommended on CPU; continuing on GPU launcher if set."
  fi
  CSV_PATH="${CSV_DIR}" \
  DATASET_NAME="${DATASET_NAME}" \
  TARGET_MEASURE="${TARGET_MEASURE}" \
  MODE="${MODE}" \
  API_BASE="${API_BASE}" \
  LLM_MODEL="${LLM_MODEL}" \
  SEED="${SEED}" \
  bash scripts/robustness/launch_robustness_job.sh
elif [ "${MODE}" = "best_existing" ]; then
  if [ "${IS_CPU}" = "true" ]; then
    echo "❌ Best Existing should run on GPU (sc|sphinx|jag)." >&2
    exit 1
  fi
  CSV_PATH="${CSV_DIR}" \
  DATASET_NAME="${DATASET_NAME}" \
  TARGET_MEASURE="${TARGET_MEASURE}" \
  MODE="${MODE}" \
  TARGET_CLUSTER="${TARGET_CLUSTER}" \
  SEED="${SEED}" \
  bash scripts/robustness/launch_robustness_job.sh
elif [ "${MODE}" = "autometrics" ]; then
  if [ "${IS_CPU}" = "true" ]; then
    echo "❌ AutoMetrics should run on GPU (sc|sphinx|jag)." >&2
    exit 1
  fi
  CSV_PATH="${CSV_DIR}" \
  DATASET_NAME="${DATASET_NAME}" \
  TARGET_MEASURE="${TARGET_MEASURE}" \
  MODE="${MODE}" \
  TARGET_CLUSTER="${TARGET_CLUSTER}" \
  API_BASE="${API_BASE}" \
  LLM_MODEL="${LLM_MODEL}" \
  SEED="${SEED}" \
  bash scripts/robustness/launch_robustness_job.sh
else
  echo "❌ Unhandled MODE: ${MODE}" >&2
  exit 1
fi


