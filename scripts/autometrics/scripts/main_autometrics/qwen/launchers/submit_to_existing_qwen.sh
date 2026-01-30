#!/bin/bash

# Submit main autometrics jobs to an already running Qwen server.
#
# Usage examples:
#   submit_to_existing_qwen.sh --url http://sphinx11.stanford.edu:8219 \
#     --dataset RealHumanEval --seed 45
#
#   submit_to_existing_qwen.sh --url sphinx11.stanford.edu:8219 \
#     --dataset Primock57 --seed 42 --seed 43
#
# Notes:
# - The provided URL can be with or without scheme and with or without "/v1".
#   Examples accepted: http://host:8219, host:8219, http://host:8219/v1, @http://host:8219
# - OPENAI_API_KEY is required by downstream script; if unset, we pass "None".
# - Dataset-to-measure mapping is handled below (see dataset_to_measure()).

set -euo pipefail

# Absolute repo root for consistent paths
REPO_ROOT="/nlp/scr2/nlp/personal-rm/autometrics"

print_usage() {
  cat <<EOF
Submit main autometrics jobs to an existing Qwen server (no new orchestrator).

Required:
  --url URL            Existing server URL (e.g., http://host:8219 or host:8219)
  --dataset NAME       Dataset name (e.g., RealHumanEval, Primock57)
  --seed SEED          Seed value (repeat for multiple, or use --seeds)

Optional:
  --seeds "S1 S2 ..."  Space- or comma-separated seeds
  --model-name NAME    Model name to propagate (default Qwen/Qwen3-32B)
  --output-root PATH   Output root (default results/main_runs/autometrics/qwen)
  --ablation-tag TAG   MAIN_ABLATION_TAG for cache naming (default full_k30_n5)

Examples:
  $0 --url http://sphinx11.stanford.edu:8219 --dataset RealHumanEval --seed 45
  $0 --url sphinx11.stanford.edu:8219 --dataset Primock57 --seed 42 --seed 43
EOF
}

canonicalize_api_base() {
  local raw="$1"
  # Strip optional leading '@'
  raw="${raw#@}"
  # Add scheme if missing
  if [[ ! "$raw" =~ ^https?:// ]]; then
    raw="http://${raw}"
  fi
  # Trim trailing '/'
  raw="${raw%/}"
  # Ensure suffix '/v1'
  if [[ "$raw" == */v1 ]]; then
    echo "$raw"
  else
    echo "${raw}/v1"
  fi
}

dataset_to_measure() {
  # Echo the measure for a given dataset; return nonzero if unknown
  case "$1" in
    CoGymTravelOutcome) echo "outcomeRating" ;;
    EvalGenProduct)     echo "grade" ;;
    RealHumanEval)      echo "accepted" ;;
    Primock57)          echo "time_sec" ;;
    HelpSteer2)         echo "helpfulness" ;;
    SimpEval)           echo "score" ;;
    TauBench)           echo "reward" ;;
    TauBenchBigger)     echo "reward" ;;
    TauBenchHighTemperature) echo "reward" ;;
    *) return 1 ;;
  esac
}

URL=""
DATASET_NAME=""
SEEDS_LIST=()
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_runs/autometrics/qwen_run2}"
ABLA_TAG="${MAIN_ABLATION_TAG:-full_k30_n5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url|--api-base)
      URL="$2"; shift 2 ;;
    --dataset)
      DATASET_NAME="$2"; shift 2 ;;
    --seed)
      SEEDS_LIST+=("$2"); shift 2 ;;
    --seeds)
      # Accept space- or comma-separated
      IFS=', ' read -r -a tmp <<< "$2"
      for s in "${tmp[@]}"; do
        [[ -n "$s" ]] && SEEDS_LIST+=("$s")
      done
      shift 2 ;;
    --model-name)
      MODEL_NAME="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --ablation-tag)
      ABLA_TAG="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage; exit 1 ;;
  esac
done

if [[ -z "${URL}" || -z "${DATASET_NAME}" || ${#SEEDS_LIST[@]} -eq 0 ]]; then
  echo "❌ Missing required args." >&2
  print_usage
  exit 1
fi

TARGET_MEASURE=""
if ! TARGET_MEASURE=$(dataset_to_measure "${DATASET_NAME}"); then
  echo "❌ Unknown dataset: ${DATASET_NAME}" >&2
  echo "   Supported: CoGymTravelOutcome, EvalGenProduct, RealHumanEval, Primock57, HelpSteer2, SimpEval" >&2
  exit 1
fi

API_BASE=$(canonicalize_api_base "${URL}")
echo "[Submitter] Using API base: ${API_BASE}"

# Provide OPENAI_API_KEY downstream; Qwen server ignores it, but script requires non-empty
DOWNSTREAM_OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

cd "${REPO_ROOT}"

mkdir -p logs 2>/dev/null || true

for SEED in "${SEEDS_LIST[@]}"; do
  OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}"
  mkdir -p "${OUT_DIR}" 2>/dev/null || true

  job_name="${DATASET_NAME}_qwen_main_seed${SEED}"
  envs="ALL,DATASET_NAME=${DATASET_NAME},TARGET_MEASURE=${TARGET_MEASURE},SEED=${SEED},MODEL_NAME=${MODEL_NAME},QWEN_API_BASE=${API_BASE},OPENAI_API_KEY=${DOWNSTREAM_OPENAI_API_KEY},OUTPUT_ROOT=${OUTPUT_ROOT},MAIN_ABLATION_TAG=${ABLA_TAG}"

  jid=$(sbatch \
    --job-name="${job_name}" \
    --output="logs/${job_name}_%j.out" \
    --error="logs/${job_name}_%j.err" \
    --export=${envs} \
    scripts/main_autometrics/qwen/run_autometrics_qwen_remote_cpu.sh | awk '{print $4}')

  echo "[Submitter] Submitted dataset=${DATASET_NAME} measure=${TARGET_MEASURE} seed=${SEED} -> job ${jid}"
done


# EXAMPLE USAGE:
# bash scripts/main_autometrics/qwen/launchers/submit_to_existing_qwen.sh --url http://sphinx9.stanford.edu:8956 --dataset RealHumanEval --seed 45
# bash scripts/main_autometrics/qwen/launchers/submit_to_existing_qwen.sh --url http://sphinx3.stanford.edu:8219 --dataset TauBench --seed 42