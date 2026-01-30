#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_remote_scaling_full_sc
#SBATCH --output=logs/qwen_remote_scaling_full_sc_%j.out
#SBATCH --error=logs/qwen_remote_scaling_full_sc_%j.err
#SBATCH --constraint=[141G|80G]
#SBATCH -x tiger[1-8]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Data scaling run (FULL mode) on jagupard GPUs, pointing to an existing Qwen server (set QWEN_API_BASE).
# Required env vars: DATASET_NAME, TARGET_MEASURE, SEED, TRAIN_SIZE, QWEN_API_BASE

set -euo pipefail

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
set +u
conda activate autometrics
set -u

cd /nlp/scr2/nlp/personal-rm/autometrics

if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required"; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required"; exit 1; fi
if [ -z "${SEED:-}" ]; then echo "❌ SEED required"; exit 1; fi
if [ -z "${OPENAI_API_KEY:-}" ]; then echo "❌ OPENAI_API_KEY required"; exit 1; fi
if [ -z "${QWEN_API_BASE:-}" ]; then echo "❌ QWEN_API_BASE required (e.g., http://host:port/v1)"; exit 1; fi

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/data_scaling/autometrics/qwen"}

mkdir -p logs

# Base output dir for scaling runs; data_scaling.py will create per-size/mode subfolders
OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}"
mkdir -p "${OUT_DIR}"

SIZE_TOKEN="sz${TRAIN_SIZE:-Full}"
MODE_TOKEN="fullbank"

echo "Running Qwen data scaling (FULL)"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Seed:    $SEED"
echo "Model:   $MODEL_NAME"
echo "API:     $QWEN_API_BASE"
echo "Out:     $OUT_DIR"
echo "TrainSz: ${TRAIN_SIZE:-Full}"

# Respect SLURM's GPU allocation; do not override CUDA_VISIBLE_DEVICES here
export OPENAI_API_BASE="$QWEN_API_BASE"
export AUTOMETRICS_LM_GENERATOR="Qwen/Qwen3-32B"
export AUTOMETRICS_LM_JUDGE="Qwen/Qwen3-32B"

# Caching: experiment-specific cache dirs (include size/mode tokens)
ABLA_TAG="full_k30_n5"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_scaling_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${SIZE_TOKEN}_${MODE_TOKEN}_seed${SEED}"
export AUTOMETRICS_CACHE_DIR="/nlp/scr2/nlp/personal-rm/autometrics/autometrics_cache_scaling/autometrics_cache_${DATASET_NAME}_${TARGET_MEASURE}_seed${SEED}_${SIZE_TOKEN}_${MODE_TOKEN}"

# If caches do not exist, clone from ablation caches as warm-start
ensure_caches() {
  local ds="$1"; shift
  local tm="$1"; shift
  local seed="$1"; shift

  # If the dataset is RealHumanEval, use the Specific cache that already computed INFORMRewardModel on the whole dataset.
  if [ "${ds}" = "RealHumanEval" ]; then
    local src_auto="/nlp/scr2/nlp/personal-rm/autometrics/autometrics_cache_scaling/autometrics_cache_RealHumanEval_accepted_seed44_fullbank"
  else
    local src_auto="/nlp/scr2/nlp/personal-rm/autometrics/autometrics_cache_ablations/autometrics_cache_${ds}_${tm}_seed${seed}_k30_n5"
  fi
  local src_dspy="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_ablation_qwen_${ds}_${tm}_full_k30_n5_seed${seed}"

  if [ ! -d "${AUTOMETRICS_CACHE_DIR}" ]; then
    if [ -d "${src_auto}" ]; then
      echo "[Scaling FULL] Cloning Autometrics cache from ${src_auto} → ${AUTOMETRICS_CACHE_DIR}"
      mkdir -p "$(dirname "${AUTOMETRICS_CACHE_DIR}")"
      cp -r "${src_auto}" "${AUTOMETRICS_CACHE_DIR}"
    else
      echo "[Scaling FULL] No source Autometrics cache found at ${src_auto}; starting fresh."
    fi
  fi

  if [ ! -d "${DSPY_CACHEDIR}" ]; then
    if [ -d "${src_dspy}" ]; then
      echo "[Scaling FULL] Cloning DSPy cache from ${src_dspy} → ${DSPY_CACHEDIR}"
      mkdir -p "$(dirname "${DSPY_CACHEDIR}")"
      cp -r "${src_dspy}" "${DSPY_CACHEDIR}"
    else
      echo "[Scaling FULL] No source DSPy cache found at ${src_dspy}; starting fresh."
    fi
  fi
}

ensure_caches "${DATASET_NAME}" "${TARGET_MEASURE}" "${SEED}"

# TMPDIR handling
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
echo "[Scaling FULL] Using TMPDIR=${TMPDIR}"

# Torch/TRITON caches per-run
TORCH_EXTENSIONS_DIR="/nlp/scr3/nlp/20questions/torch_extensions/autometrics_scaling_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${SIZE_TOKEN}_${MODE_TOKEN}_seed${SEED}"
TRITON_CACHE_DIR="/nlp/scr3/nlp/20questions/triton_cache/autometrics_scaling_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${SIZE_TOKEN}_${MODE_TOKEN}_seed${SEED}"
export TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR
rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true

cleanup_torch_triton_cache() {
  rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup_torch_triton_cache EXIT INT TERM

PY_ARGS=(
  "${DATASET_NAME}"
  "${TARGET_MEASURE}"
  "${SEED}"
  "${OUT_DIR}"
  --model-name "${MODEL_NAME}"
  --api-base "${QWEN_API_BASE}"
)

if [ -n "${TRAIN_SIZE:-}" ]; then
  PY_ARGS+=(--train-size "${TRAIN_SIZE}")
fi

python analysis/data_scaling/data_scaling.py "${PY_ARGS[@]}"


