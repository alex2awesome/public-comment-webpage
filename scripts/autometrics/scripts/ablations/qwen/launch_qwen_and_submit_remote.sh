#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=200GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_orchestrator
#SBATCH --output=logs/qwen_orchestrator_%j.out
#SBATCH --error=logs/qwen_orchestrator_%j.err
#SBATCH --constraint=[141G|80G]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Orchestrator: starts a persistent Qwen server (on this node) and submits one or more
# jagupard ablation jobs that connect to it via QWEN_API_BASE. Useful when H200/H100
# are scarce but jagupard GPUs are plentiful.
#
# Usage examples:
#   # Single ablation
#   export DATASET_NAME="EvalGenProduct" TARGET_MEASURE="quality" SEEDS="42 43" \
#          METRICBANK_MODE="full" K=30 N=5 NO_METRIC_CARDS=false FORCE_REINDEX=false
#   sbatch scripts/ablations/qwen/launch_qwen_and_submit_remote.sh
#
#   # Full suite (runs the ordered set for each SEED):
#   export DATASET_NAME="EvalGenProduct" TARGET_MEASURE="quality" SEEDS="42 43" FULL_SUITE="true"
#   sbatch scripts/ablations/qwen/launch_qwen_and_submit_remote.sh
#
# Optional env:
#   MODEL_PATH (default Qwen/Qwen3-32B), PORT (default 8123), HOST (auto-detected),
#   OUTPUT_ROOT (defaults to results/ablations/qwen_remote), SERVER_TAG (for logging only),
#   OPENAI_API_KEY (if unset, we set it to "None" for downstream jobs)

set -euo pipefail

set +u
. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang
set -u

cd /nlp/scr2/nlp/personal-rm/autometrics

mkdir -p logs

# Prefer a writable, job-local TMPDIR. Try in order: $TMP_BASE, repo-local on scr2, $HOME, then /tmp
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
echo "[Orchestrator] Using TMPDIR=${TMPDIR}"

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8123}
# Prefer FQDN for cross-node access
HOST=${HOST:-$(hostname -f)}
TP=${TP:-"2"}
DTYPE=${DTYPE:-"bfloat16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/qwen_remote_run2"}

echo "[Orchestrator] Starting Qwen server on ${HOST}:${PORT}"
echo "[Orchestrator] Model: ${MODEL_PATH} (tp=${TP}, dtype=${DTYPE})"
echo "[Orchestrator] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Preflight: ensure we have at least TP visible GPUs
# Prefer SLURM/CUDA_VISIBLE_DEVICES visibility; fallback to nvidia-smi if unset
NUM_VISIBLE_GPUS=0
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_VISIBLE_GPUS="${#_cvd[@]}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_VISIBLE_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
fi

if [ "${TP}" -gt "${NUM_VISIBLE_GPUS}" ]; then
  echo "[Orchestrator] ❌ Requested tp=${TP} but only ${NUM_VISIBLE_GPUS} visible GPU(s)."
  echo "[Orchestrator]    Check --gres, partition constraints, or CUDA_VISIBLE_DEVICES."
  exit 1
fi

# Prepare server log location and ensure directory exists
LOG_DIR="/nlp/scr2/nlp/personal-rm/autometrics/scripts/ablations/qwen/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/sglang_server_${SLURM_JOB_ID}.log"
echo "[Orchestrator] Server logs will stream to: ${SERVER_LOG}"

# Start server in background and capture PID (respect SLURM-assigned GPUs; do not override CUDA_VISIBLE_DEVICES)
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port ${PORT} \
  --host ${HOST} \
  --tp ${TP} \
  --dtype ${DTYPE} \
  --mem-fraction-static ${MEM_FRACTION} \
  --trust-remote-code > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "[Orchestrator] Qwen server PID: ${SERVER_PID}"

API_BASE="http://${HOST}:${PORT}/v1"
echo "[Orchestrator] Waiting for server readiness at ${API_BASE} …"
echo "[Orchestrator] Tailing server logs (Ctrl+C will stop tail but not the server) …"
tail -n +1 -F "${SERVER_LOG}" &
TAIL_PID=$!

TIMEOUT=90
START_TIME=$(date +%s)
until curl -s "${API_BASE}/get_model_info" > /dev/null; do
  # Check if server process died
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "[Orchestrator] ❌ Server process exited unexpectedly. Showing last 100 log lines:"
    tail -n 100 "${SERVER_LOG}" || true
    kill ${TAIL_PID} 2>/dev/null || true
    exit 1
  fi
  echo "[Orchestrator] Still waiting …"
  sleep 10
  if [ $(( $(date +%s) - START_TIME )) -gt $((TIMEOUT * 60)) ]; then
    echo "[Orchestrator] ❌ Timeout waiting for server. Showing last 100 log lines:"
    tail -n 100 "${SERVER_LOG}" || true
    kill ${TAIL_PID} 2>/dev/null || true
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
  fi
done

echo "[Orchestrator] ✅ Server ready at ${API_BASE}"
kill ${TAIL_PID} 2>/dev/null || true

# Prepare downstream job submissions
if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required"; kill ${SERVER_PID} || true; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required"; kill ${SERVER_PID} || true; exit 1; fi
SEEDS=${SEEDS:-"42"}

# Ensure downstream jobs have some API key (Qwen server ignores it)
DOWNSTREAM_OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

echo "[Orchestrator] Submitting jobs to jagupard (one per seed/config)"

submit_one() {
  local seed="$1"; shift
  local mb_mode="$1"; shift
  local k_val="$1"; shift
  local n_val="$1"; shift
  local no_cards="$1"; shift
  local force_reidx="$1"; shift
  local resized="$1"; shift

  local envs="ALL,DATASET_NAME=${DATASET_NAME},TARGET_MEASURE=${TARGET_MEASURE},SEED=${seed},MODEL_NAME=${MODEL_PATH},QWEN_API_BASE=${API_BASE},OPENAI_API_KEY=${DOWNSTREAM_OPENAI_API_KEY},METRICBANK_MODE=${mb_mode},OUTPUT_ROOT=${OUTPUT_ROOT}"
  if [ -n "${k_val}" ]; then envs=",${envs},K=${k_val}"; fi
  if [ -n "${n_val}" ]; then envs=",${envs},N=${n_val}"; fi
  if [ "${no_cards}" = "true" ]; then envs=",${envs},NO_METRIC_CARDS=true"; fi
  if [ "${force_reidx}" = "true" ]; then envs=",${envs},FORCE_REINDEX=true"; fi
  if [ "${resized}" = "true" ]; then envs=",${envs},RESIZED=true"; fi

  # Submit and capture job id
  jid=$(sbatch --export=${envs} scripts/ablations/qwen/run_ablation_qwen_remote.sh | awk '{print $4}')
  echo "[Orchestrator] Submitted seed=${seed} mode=${mb_mode} k=${k_val:-default} n=${n_val:-default} desc=${no_cards} reindex=${force_reidx} resized=${resized} => job ${jid}"
}

JOBS=()
resized=${RESIZED:-"false"}

if [ "${FULL_SUITE:-false}" = "true" ]; then
  for s in ${SEEDS}; do
    # Retrieval k: 30,20,10,5
    for k in 30 20 10 5; do
      submit_one "$s" "full" "$k" "" "false" "false" "${resized}"; done
    # Regression n with k fixed at 30: 10,5,3,1
    for n in 20 10 5 3 1; do
      submit_one "$s" "full" "30" "$n" "false" "false" "${resized}"; done
    # No Metric Cards with k=20 (force reindex)
    submit_one "$s" "full" "20" "" "true" "true" "${resized}";
    # Generated only (force reindex)
    submit_one "$s" "generated_only" "" "" "false" "true" "${resized}";
    # Existing only (force reindex)
    submit_one "$s" "existing_only" "" "" "false" "true" "${resized}";
  done
else
  # Single config run(s) for provided K/N/etc.
  mb_mode=${METRICBANK_MODE:-"full"}
  k_val=${K:-""}
  n_val=${N:-""}
  no_cards=${NO_METRIC_CARDS:-"false"}
  force_reidx=${FORCE_REINDEX:-"false"}
  for s in ${SEEDS}; do
    submit_one "$s" "$mb_mode" "$k_val" "$n_val" "$no_cards" "$force_reidx" "${resized}";
  done
fi

echo "[Orchestrator] Server will remain running. Cancel this job to shut it down (scancel ${SLURM_JOB_ID})."
echo "[Orchestrator] QWEN_API_BASE=${API_BASE}"

# Keep the server alive until this orchestrator job is cancelled
wait ${SERVER_PID}


