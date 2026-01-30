#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sphinx
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_main_orchestrator_sphinx2
#SBATCH --output=logs/qwen_main_orchestrator_sphinx2_%j.out
#SBATCH --error=logs/qwen_main_orchestrator_sphinx2_%j.err
#SBATCH -x sphinx[1-2,4-5]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Orchestrator (sphinx2 2xGPU): starts a persistent Qwen server and submits main jobs
# that connect to it via QWEN_API_BASE.

set -euo pipefail

set +u
. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang
set -u

cd /nlp/scr2/nlp/personal-rm/autometrics

## Prefer a writable, job-local TMPDIR. Try in order: $TMP_BASE, repo-local on scr2, $HOME, then /tmp
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
echo "[Main Orchestrator (sphinx2)] Using TMPDIR=${TMPDIR}"

## Force per-run Torch/TRITON extension caches and ensure cleanup for the server process
TORCH_EXTENSIONS_DIR="${JOB_TMPDIR}/torch_extensions"
TRITON_CACHE_DIR="${JOB_TMPDIR}/triton_cache"
export TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR

# Clean on startup (in case of retries) and create fresh dirs
rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
echo "[Main Orchestrator (sphinx2)] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[Main Orchestrator (sphinx2)] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

# Always wipe caches on orchestrator exit, cancellation, or failure
cleanup_torch_triton_cache() {
  rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup_torch_triton_cache EXIT INT TERM

mkdir -p logs

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8219}
HOST=${HOST:-$(hostname -f)}
TP=${TP:-"2"}
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/main_runs/autometrics/qwen_run2"}

echo "[Main Orchestrator (sphinx2)] Starting Qwen server on ${HOST}:${PORT}"

LOG_DIR="/nlp/scr2/nlp/personal-rm/autometrics/scripts/main_autometrics/qwen/logs"
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
echo "[Main Orchestrator (sphinx2)] Waiting for server readiness at ${API_BASE} …"
tail -n +1 -F "${SERVER_LOG}" &
TAIL_PID=$!

TIMEOUT=90
START_TIME=$(date +%s)
until curl -s "${API_BASE}/get_model_info" > /dev/null; do
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "[Main Orchestrator (sphinx2)] ❌ Server process exited unexpectedly."
    tail -n 100 "${SERVER_LOG}" || true
    kill ${TAIL_PID} 2>/dev/null || true
    exit 1
  fi
  echo "[Main Orchestrator (sphinx2)] Still waiting …"
  sleep 10
  if [ $(( $(date +%s) - START_TIME )) -gt $((TIMEOUT * 60)) ]; then
    echo "[Main Orchestrator (sphinx2)] ❌ Timeout waiting for server."
    tail -n 100 "${SERVER_LOG}" || true
    kill ${TAIL_PID} 2>/dev/null || true
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
  fi
done

echo "[Main Orchestrator (sphinx2)] ✅ Server ready at ${API_BASE}"
kill ${TAIL_PID} 2>/dev/null || true

# Allow ALL mode
SEEDS=${SEEDS:-"42 43 44 45 46"}
ALL_MODE=false
if [ "${DATASET_NAME:-}" = "ALL" ] || [ "${ALL_DATASETS:-false}" = "true" ]; then
  ALL_MODE=true
else
  if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required (or set DATASET_NAME=ALL)"; kill ${SERVER_PID} || true; exit 1; fi
  if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required (or set DATASET_NAME=ALL)"; kill ${SERVER_PID} || true; exit 1; fi
fi

DOWNSTREAM_OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

echo "[Main Orchestrator (sphinx2)] Submitting main jobs (one per seed/config)"

submit_one() {
  local seed="$1"; shift

  local out_dir="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}"
  local log_file="${out_dir}/log_${seed}.json"
  local score_file="${out_dir}/score_pearson_${seed}.txt"
  if [ -f "${log_file}" ] && [ -f "${score_file}" ]; then
    echo "[Main Orchestrator (sphinx2)] ✅ Already done: ${DATASET_NAME}/${TARGET_MEASURE} seed=${seed}"
    return 0
  fi

  local envs="ALL,DATASET_NAME=${DATASET_NAME},TARGET_MEASURE=${TARGET_MEASURE},SEED=${seed},MODEL_NAME=${MODEL_PATH},QWEN_API_BASE=${API_BASE},OPENAI_API_KEY=${DOWNSTREAM_OPENAI_API_KEY},OUTPUT_ROOT=${OUTPUT_ROOT}"

  local job_name="${DATASET_NAME}_qwen_main"
  jid=$(sbatch \
    --job-name="${job_name}" \
    --output="logs/${job_name}_%j.out" \
    --error="logs/${job_name}_%j.err" \
    --export=${envs} \
    scripts/main_autometrics/qwen/run_autometrics_qwen_remote.sh | awk '{print $4}')
  echo "[Main Orchestrator (sphinx2)] Submitted job_name=${job_name} seed=${seed} => job ${jid}"
}

should_skip_dataset() {
  local name="$1"
  if [ -n "${ONLY_DATASETS:-}" ]; then
    for d in ${ONLY_DATASETS}; do
      if [ "$d" = "$name" ]; then return 1; fi
    done
    return 0
  fi
  if [ -n "${DISABLE_DATASETS:-}" ]; then
    for d in ${DISABLE_DATASETS}; do
      if [ "$d" = "$name" ]; then return 0; fi
    done
  fi
  return 1
}

if [ "$ALL_MODE" = true ]; then
  DATASET_SPECS_LIST=${DATASET_SPECS:-"CoGymTravelOutcome:outcomeRating:false EvalGenProduct:grade:false RealHumanEval:accepted:false Primock57:time_sec:false HelpSteer2:helpfulness:false SimpEval:score:false"}
  for spec in ${DATASET_SPECS_LIST}; do
    IFS=':' read -r ds tm _ <<< "${spec}"
    if should_skip_dataset "${ds}"; then
      echo "[Main Orchestrator (sphinx2)] Skipping dataset ${ds} due to ONLY/DISABLE filters"
      continue
    fi
    DATASET_NAME="${ds}"
    TARGET_MEASURE="${tm}"
    echo "[Main Orchestrator (sphinx2)] Dataset=${DATASET_NAME} Measure=${TARGET_MEASURE}"
    for s in ${SEEDS}; do
      submit_one "$s"
    done
  done
else
  for s in ${SEEDS}; do
    submit_one "$s"
  done
fi

echo "[Main Orchestrator (sphinx2)] Server will remain running. Cancel this job to shut it down (scancel ${SLURM_JOB_ID})."
echo "[Main Orchestrator (sphinx2)] QWEN_API_BASE=${API_BASE}"

wait ${SERVER_PID}


