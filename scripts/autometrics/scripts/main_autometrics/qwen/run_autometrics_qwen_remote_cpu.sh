#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:0
#SBATCH --open-mode=append
#SBATCH --partition=john-lo
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_main_remote_cpu
#SBATCH --output=logs/qwen_main_remote_cpu_%j.out
#SBATCH --error=logs/qwen_main_remote_cpu_%j.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Main run on CPU partition pointing to an existing Qwen server (set QWEN_API_BASE).
# Required env vars: DATASET_NAME, TARGET_MEASURE, SEED, QWEN_API_BASE

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
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/main_runs/autometrics/qwen_run2"}

mkdir -p logs

# Output dir for main runs (no ablation tags)
OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}"
mkdir -p "${OUT_DIR}"

echo "Running Qwen main (remote CPU)"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Seed:    $SEED"
echo "Model:   $MODEL_NAME"
echo "API:     $QWEN_API_BASE"
echo "Out:     $OUT_DIR"

export OPENAI_API_BASE="$QWEN_API_BASE"
export AUTOMETRICS_LM_GENERATOR="Qwen/Qwen3-32B"
export AUTOMETRICS_LM_JUDGE="Qwen/Qwen3-32B"

# Mirror ablation cache naming; default to K=30, n=5 cache
ABLA_TAG="${MAIN_ABLATION_TAG:-full_k30_n5}"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"

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
echo "[Main Remote CPU] Using TMPDIR=${TMPDIR}"

# Force per-run Torch/TRITON extension caches keyed by dataset/measure/seed/ablation
TORCH_EXTENSIONS_DIR="/nlp/scr3/nlp/20questions/torch_extensions/autometrics_main_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
TRITON_CACHE_DIR="/nlp/scr3/nlp/20questions/triton_cache/autometrics_main_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
export TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR

# Clean on startup (in case of retries) and create fresh dirs
rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
echo "[Main Remote CPU] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[Main Remote CPU] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

# Always wipe caches on job exit, cancellation, or failure
cleanup_torch_triton_cache() {
  rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup_torch_triton_cache EXIT INT TERM

# Skip if already completed
if [ -f "$OUT_DIR/score_pearson_${SEED}.txt" ] && [ -f "$OUT_DIR/log_${SEED}.json" ]; then
  echo "Results already exist in $OUT_DIR for seed $SEED. Skipping."
  exit 0
fi

python analysis/main_experiments/run_main_autometrics.py \
  "$DATASET_NAME" \
  "$TARGET_MEASURE" \
  "$SEED" \
  "$OUT_DIR" \
  --model-name "$MODEL_NAME" \
  --api-base "$QWEN_API_BASE"


