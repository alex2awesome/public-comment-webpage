#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sphinx
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_scaling_orchestrator_sphinx2
#SBATCH --output=logs/qwen_scaling_orchestrator_sphinx2_%j.out
#SBATCH --error=logs/qwen_scaling_orchestrator_sphinx2_%j.err
#SBATCH -x sphinx[1-2,4-6]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Orchestrator (sphinx2 2xGPU): starts a persistent Qwen server and submits data scaling jobs
# across seeds, modes (full vs generated-only), and train sizes.

set -euo pipefail

set +u
. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang
set -u

cd /nlp/scr2/nlp/personal-rm/autometrics

mkdir -p scripts/data_scaling/qwen/logs || true

choose_tmp_base() {
  local candidates=()
  if [ -n "${TMP_BASE:-}" ]; then candidates+=("${TMP_BASE}"); fi
  candidates+=("/nlp/scr2/nlp/personal-rm/autometrics/.tmp")
  candidates+=("${HOME}/.tmp")
  candidates+=("/tmp")
  local cand
  for cand in "${candidates[@]}"; do
    [ -z "${cand}" ] && continue
    mkdir -p "${cand}" 2>/dev/null || true
    if [ -d "${cand}" ] && [ -w "${cand}" ]; then
      echo "${cand}"
      return 0
    fi
  done
  echo "/tmp"
}
TMP_BASE_CHOSEN="$(choose_tmp_base)"
JOB_TMPDIR="${TMP_BASE_CHOSEN}/slurm_${SLURM_JOB_ID}"
mkdir -p "${JOB_TMPDIR}" 2>/dev/null || true
if [ -d "${JOB_TMPDIR}" ] && [ -w "${JOB_TMPDIR}" ]; then
  export TMPDIR="${JOB_TMPDIR}"
else
  export TMPDIR="${TMP_BASE_CHOSEN}"
fi
echo "[Scaling Orchestrator] Using TMPDIR=${TMPDIR}"

# Per-run Torch/TRITON caches for the server process
TORCH_EXTENSIONS_DIR="${JOB_TMPDIR}/torch_extensions"
TRITON_CACHE_DIR="${JOB_TMPDIR}/triton_cache"
export TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR
rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true

cleanup_torch_triton_cache() {
  rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup_torch_triton_cache EXIT INT TERM

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8219}
HOST=${HOST:-$(hostname -f)}
TP=${TP:-"2"}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/data_scaling/autometrics/qwen"}

echo "[Scaling Orchestrator] Starting Qwen server on ${HOST}:${PORT}"

LOG_DIR="/nlp/scr2/nlp/personal-rm/autometrics/scripts/data_scaling/qwen/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/sglang_server_${SLURM_JOB_ID}.log"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port ${PORT} \
  --host ${HOST} \
  --tp ${TP} \
  --dtype ${DTYPE} \
  --mem-fraction-static ${MEM_FRACTION} \
  --trust-remote-code > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
API_BASE="http://${HOST}:${PORT}/v1"
echo "[Scaling Orchestrator] Waiting for server readiness at ${API_BASE} …"
tail -n +1 -F "${SERVER_LOG}" &
TAIL_PID=$!

TIMEOUT=90
START_TIME=$(date +%s)
until curl -s "${API_BASE}/get_model_info" > /dev/null; do
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "[Scaling Orchestrator] ❌ Server process exited unexpectedly."
    tail -n 100 "${SERVER_LOG}" || true
    kill ${TAIL_PID} 2>/dev/null || true
    exit 1
  fi
  echo "[Scaling Orchestrator] Still waiting …"
  sleep 10
  if [ $(( $(date +%s) - START_TIME )) -gt $((TIMEOUT * 60)) ]; then
    echo "[Scaling Orchestrator] ❌ Timeout waiting for server."
    tail -n 100 "${SERVER_LOG}" || true
    kill ${TAIL_PID} 2>/dev/null || true
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
  fi
done

echo "[Scaling Orchestrator] ✅ Server ready at ${API_BASE}"
kill ${TAIL_PID} 2>/dev/null || true

# Seeds, sizes, and datasets
SEEDS=${SEEDS:-"42 43 44 45 46"}
TRAIN_SIZES_DEFAULT="5 10 20 40 80 160 320 640"
TRAIN_SIZES=${TRAIN_SIZES:-"${TRAIN_SIZES_DEFAULT}"}

# Allow ALL mode similar to main orchestrator
ALL_MODE=false
if [ "${DATASET_NAME:-}" = "ALL" ] || [ "${ALL_DATASETS:-false}" = "true" ]; then
  ALL_MODE=true
fi

# Helper: submit a single job
submit_one() {
  local mode="$1"; shift # "full" or "genonly"
  local seed="$1"; shift
  local train_size="$1"; shift

  local out_dir="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}"
  local size_token="sz${train_size}"
  local mode_token="$([ "${mode}" = "genonly" ] && echo genonly || echo fullbank)"
  local derived_dir="${out_dir}/${DATASET_NAME}_${TARGET_MEASURE}_seed${seed}_${size_token}_${mode_token}"

  # Skip if already completed
  local score_file="${derived_dir}/score_pearson_${seed}.txt"
  local log_file="${derived_dir}/log_${seed}.json"
  if [ -f "${score_file}" ] && [ -f "${log_file}" ]; then
    echo "[Scaling Orchestrator] ✅ Already done: ${DATASET_NAME}/${TARGET_MEASURE} seed=${seed} size=${train_size} mode=${mode}"
    return 0
  fi

  local envs="ALL,DATASET_NAME=${DATASET_NAME},TARGET_MEASURE=${TARGET_MEASURE},SEED=${seed},TRAIN_SIZE=${train_size},MODEL_NAME=${MODEL_PATH},QWEN_API_BASE=${API_BASE},OPENAI_API_KEY=${OPENAI_API_KEY:-None},OUTPUT_ROOT=${OUTPUT_ROOT}"

  if [ "${mode}" = "genonly" ]; then
    local job_name="${DATASET_NAME}_qwen_scaling_genonly"
    jid=$(sbatch \
      --job-name="${job_name}" \
      --output="scripts/data_scaling/qwen/logs/${job_name}_%j.out" \
      --error="scripts/data_scaling/qwen/logs/${job_name}_%j.err" \
      --export=${envs} \
      scripts/data_scaling/qwen/run_data_scaling_qwen_remote_generated_only_cpu.sh | awk '{print $4}')
    echo "[Scaling Orchestrator] Submitted GENONLY seed=${seed} size=${train_size} => job ${jid}"
  else
    local job_name="${DATASET_NAME}_qwen_scaling_full"
    jid=$(sbatch \
      --job-name="${job_name}" \
      --output="scripts/data_scaling/qwen/logs/${job_name}_%j.out" \
      --error="scripts/data_scaling/qwen/logs/${job_name}_%j.err" \
      --export=${envs} \
      scripts/data_scaling/qwen/run_data_scaling_qwen_remote_full.sh | awk '{print $4}')
    echo "[Scaling Orchestrator] Submitted FULL seed=${seed} size=${train_size} => job ${jid}"
  fi
}

run_for_dataset() {
  local ds="$1"; shift
  local tm="$1"; shift
  echo "[Scaling Orchestrator] Dataset=${ds} Measure=${tm}"
  DATASET_NAME="${ds}"
  TARGET_MEASURE="${tm}"
  for s in ${SEEDS}; do
    for n in ${TRAIN_SIZES}; do
      submit_one full "${s}" "${n}"
      submit_one genonly "${s}" "${n}"
    done
  done
}

if [ "${ALL_MODE}" = true ]; then
  DATASET_SPECS_LIST=${DATASET_SPECS:-"CoGymTravelOutcome:outcomeRating EvalGenProduct:grade RealHumanEval:accepted Primock57:time_sec HelpSteer2:helpfulness SimpEval:score"}
  for spec in ${DATASET_SPECS_LIST}; do
    IFS=':' read -r ds tm <<< "${spec}"
    run_for_dataset "${ds}" "${tm}"
  done
else
  if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required (or set DATASET_NAME=ALL)"; kill ${SERVER_PID} || true; exit 1; fi
  if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required (or set DATASET_NAME=ALL)"; kill ${SERVER_PID} || true; exit 1; fi
  run_for_dataset "${DATASET_NAME}" "${TARGET_MEASURE}"
fi

echo "[Scaling Orchestrator] Server will remain running. Cancel this job to shut it down (scancel ${SLURM_JOB_ID})."
echo "[Scaling Orchestrator] QWEN_API_BASE=${API_BASE}"

wait ${SERVER_PID}


