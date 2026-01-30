#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sphinx
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_orchestrator_sphinx2
#SBATCH --output=logs/qwen_orchestrator_sphinx2_%j.out
#SBATCH --error=logs/qwen_orchestrator_sphinx2_%j.err
#SBATCH -x sphinx[1-2,5-6]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Orchestrator (sphinx2 2xGPU): starts a persistent Qwen server on this node and submits
# one or more sphinx2 ablation jobs that connect to it via QWEN_API_BASE.
#
# Optional env:
#   MODEL_PATH (default Qwen/Qwen3-32B)
#   PORT (default 8123)
#   HOST (auto-detected)
#   TP (default 2)
#   DTYPE (default float16)
#   MEM_FRACTION (default 0.8)
#   OUTPUT_ROOT (default results/ablations/qwen_remote)
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

# Force per-run Torch/TRITON extension caches and ensure cleanup for the server process
TORCH_EXTENSIONS_DIR="${JOB_TMPDIR}/torch_extensions"
TRITON_CACHE_DIR="${JOB_TMPDIR}/triton_cache"
export TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR

# Clean on startup (in case of retries) and create fresh dirs
rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
echo "[Orchestrator] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[Orchestrator] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

# Always wipe caches on orchestrator exit, cancellation, or failure
cleanup_torch_triton_cache() {
  rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup_torch_triton_cache EXIT INT TERM

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8123}
# Prefer FQDN for cross-node access
HOST=${HOST:-$(hostname -f)}
TP=${TP:-"2"}
# Default to float16 for jagupard consumer GPUs
DTYPE=${DTYPE:-"float16"}
MEM_FRACTION=${MEM_FRACTION:-"0.8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/qwen_remote_run2"}

echo "[Orchestrator] Starting Qwen server on ${HOST}:${PORT}"
echo "[Orchestrator] Model: ${MODEL_PATH} (tp=${TP}, dtype=${DTYPE})"
echo "[Orchestrator] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Preflight: ensure we have at least TP visible GPUs
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

# Start server in background and capture PID (respect SLURM-assigned GPUs)
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
# Allow ALL mode to submit multiple datasets in one orchestrator
SEEDS=${SEEDS:-"42"}
ALL_MODE=false
if [ "${DATASET_NAME:-}" = "ALL" ] || [ "${ALL_DATASETS:-false}" = "true" ]; then
  ALL_MODE=true
else
  if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required (or set DATASET_NAME=ALL)"; kill ${SERVER_PID} || true; exit 1; fi
  if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required (or set DATASET_NAME=ALL)"; kill ${SERVER_PID} || true; exit 1; fi
fi

# Ensure downstream jobs have some API key (Qwen server ignores it)
DOWNSTREAM_OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}

echo "[Orchestrator] Submitting jobs to sphinx (one per seed/config)"

submit_one() {
  local seed="$1"; shift
  local mb_mode="$1"; shift
  local k_val="$1"; shift
  local n_val="$1"; shift
  local no_cards="$1"; shift
  local force_reidx="$1"; shift
  local resized="$1"; shift

  # Pre-submit existence check to avoid queue spam
  local abla_tag="${mb_mode}"
  if [ -n "${k_val}" ]; then abla_tag="${abla_tag}_k${k_val}"; fi
  if [ -n "${n_val}" ]; then abla_tag="${abla_tag}_n${n_val}"; fi
  if [ "${no_cards}" = "true" ]; then abla_tag="${abla_tag}_desc"; fi

  local base_dir
  if [ "${resized}" = "true" ]; then
    base_dir="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}_resized"
  else
    base_dir="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}"
  fi
  local out_dir="${base_dir}/${abla_tag}"

  local metric_to_check="${DONE_CHECK_METRIC:-pearson}"
  local log_file="${out_dir}/log_${seed}.json"
  local score_file="${out_dir}/score_${metric_to_check}_${seed}.txt"
  if [ -f "${log_file}" ] && [ -f "${score_file}" ]; then
    echo "[Orchestrator] ✅ Already done: ${DATASET_NAME}/${TARGET_MEASURE}/${abla_tag} seed=${seed} (${metric_to_check})"
    return 0
  fi

  local envs="ALL,DATASET_NAME=${DATASET_NAME},TARGET_MEASURE=${TARGET_MEASURE},SEED=${seed},MODEL_NAME=${MODEL_PATH},QWEN_API_BASE=${API_BASE},OPENAI_API_KEY=${DOWNSTREAM_OPENAI_API_KEY},METRICBANK_MODE=${mb_mode},OUTPUT_ROOT=${OUTPUT_ROOT}"
  if [ -n "${k_val}" ]; then envs=",${envs},K=${k_val}"; fi
  if [ -n "${n_val}" ]; then envs=",${envs},N=${n_val}"; fi
  if [ "${no_cards}" = "true" ]; then envs=",${envs},NO_METRIC_CARDS=true"; fi
  if [ "${force_reidx}" = "true" ]; then envs=",${envs},FORCE_REINDEX=true"; fi
  if [ "${resized}" = "true" ]; then envs=",${envs},RESIZED=true"; fi

  # Construct a descriptive job name: {dataset}_qwen_{ablation_settings}
  local abla_tag="${mb_mode}"
  if [ -n "${k_val}" ]; then abla_tag="${abla_tag}_k${k_val}"; fi
  if [ -n "${n_val}" ]; then abla_tag="${abla_tag}_n${n_val}"; fi
  if [ "${no_cards}" = "true" ]; then abla_tag="${abla_tag}_desc"; fi
  if [ "${resized}" = "true" ]; then abla_tag="${abla_tag}_resized"; fi
  local job_name="${DATASET_NAME}_qwen_${abla_tag}"

  # Submit and capture job id
  jid=$(sbatch \
    --job-name="${job_name}" \
    --output="logs/${job_name}_%j.out" \
    --error="logs/${job_name}_%j.err" \
    --export=${envs} \
    scripts/ablations/qwen/run_ablation_qwen_remote.sh | awk '{print $4}')
  echo "[Orchestrator] Submitted job_name=${job_name} seed=${seed} mode=${mb_mode} k=${k_val:-default} n=${n_val:-default} desc=${no_cards} reindex=${force_reidx} => job ${jid}"
}

JOBS=()

# Dataset iteration logic
should_skip_dataset() {
  local name="$1"
  # If ONLY_DATASETS is set, skip any not in the list
  if [ -n "${ONLY_DATASETS:-}" ]; then
    for d in ${ONLY_DATASETS}; do
      if [ "$d" = "$name" ]; then return 1; fi
    done
    return 0
  fi
  # If DISABLE_DATASETS is set, skip those in the list
  if [ -n "${DISABLE_DATASETS:-}" ]; then
    for d in ${DISABLE_DATASETS}; do
      if [ "$d" = "$name" ]; then return 0; fi
    done
  fi
  return 1
}

if [ "$ALL_MODE" = true ]; then
  # Whitespace-separated list of dataset specs: name:measure:resized_flag
  DATASET_SPECS_LIST=${DATASET_SPECS:-"CoGymTravelOutcome:outcomeRating:true EvalGenProduct:grade:true RealHumanEval:accepted:false Primock57:time_sec:false HelpSteer2:helpfulness:false SimpEval:score:false"}
  for spec in ${DATASET_SPECS_LIST}; do
    IFS=':' read -r ds tm rsz <<< "${spec}"
    if should_skip_dataset "${ds}"; then
      echo "[Orchestrator] Skipping dataset ${ds} due to ONLY/DISABLE filters"
      continue
    fi
    DATASET_NAME="${ds}"
    TARGET_MEASURE="${tm}"
    resized="${rsz}"
    echo "[Orchestrator] Dataset=${DATASET_NAME} Measure=${TARGET_MEASURE} Resized=${resized}"

    if [ "${FULL_SUITE:-false}" = "true" ]; then
      for s in ${SEEDS}; do
        for k in 30; do # 30 20 10 5; do
          submit_one "$s" "full" "$k" "" "false" "false" "${resized}"; done
        # for n in 20 10 5 3 1; do
        #   submit_one "$s" "full" "30" "$n" "false" "false" "${resized}"; done
        # submit_one "$s" "full" "20" "" "true" "true" "${resized}";
        # submit_one "$s" "generated_only" "" "" "false" "true" "${resized}";
        # submit_one "$s" "existing_only" "" "" "false" "true" "${resized}";
      done
    else
      mb_mode=${METRICBANK_MODE:-"full"}
      k_val=${K:-""}
      n_val=${N:-""}
      no_cards=${NO_METRIC_CARDS:-"false"}
      force_reidx=${FORCE_REINDEX:-"false"}
      for s in ${SEEDS}; do
        submit_one "$s" "$mb_mode" "$k_val" "$n_val" "$no_cards" "$force_reidx" "${resized}"
      done
    fi
  done
else
  resized=${RESIZED:-"false"}
  if [ "${FULL_SUITE:-false}" = "true" ]; then
    for s in ${SEEDS}; do
      for k in 30; do # 30 20 10 5; do
        submit_one "$s" "full" "$k" "" "false" "false" "${resized}"; done
      # for n in 20 10 5 3 1; do
      #   submit_one "$s" "full" "30" "$n" "false" "false" "${resized}"; done
      # submit_one "$s" "full" "20" "" "true" "true" "${resized}";
      # submit_one "$s" "generated_only" "" "" "false" "true" "${resized}";
      # submit_one "$s" "existing_only" "" "" "false" "true" "${resized}";
    done
  else
    mb_mode=${METRICBANK_MODE:-"full"}
    k_val=${K:-""}
    n_val=${N:-""}
    no_cards=${NO_METRIC_CARDS:-"false"}
    force_reidx=${FORCE_REINDEX:-"false"}
    for s in ${SEEDS}; do
      submit_one "$s" "$mb_mode" "$k_val" "$n_val" "$no_cards" "$force_reidx" "${resized}"
    done
  fi
fi

echo "[Orchestrator] Server will remain running. Cancel this job to shut it down (scancel ${SLURM_JOB_ID})."
echo "[Orchestrator] QWEN_API_BASE=${API_BASE}"

# Keep the server alive until this orchestrator job is cancelled
wait ${SERVER_PID}


