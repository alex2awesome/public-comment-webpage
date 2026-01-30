#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:4
#SBATCH --open-mode=append
#SBATCH --partition=jag-lo
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_remote_ablation
#SBATCH --output=logs/qwen_remote_ablation_%j.out
#SBATCH --error=logs/qwen_remote_ablation_%j.err
#SBATCH -x jagupard[19-20,26-31]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Add this later
#### #SBATCH --mem=40G
#### #SBATCH --gres=gpu:1
#### #SBATCH -x jagupard[19-20,26-31]

# Run ablations on jagupard-grade GPUs pointing to a separately-hosted Qwen server.
# This script DOES NOT start or stop the Qwen server.
#
# Required env vars:
#   DATASET_NAME, TARGET_MEASURE, SEED, QWEN_API_BASE
# Optional env vars:
#   METRICBANK_MODE, K, N, NO_METRIC_CARDS, FORCE_REINDEX, OUTPUT_ROOT, RESIZED

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
METRICBANK_MODE=${METRICBANK_MODE:-"full"}
K=${K:-""}
N=${N:-""}
NO_METRIC_CARDS=${NO_METRIC_CARDS:-"false"}
FORCE_REINDEX=${FORCE_REINDEX:-"false"}
RESIZED=${RESIZED:-"false"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/main_ablations/qwen_run2"}
HOST=${HOST:-$(hostname -f)}

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
echo "[Ablation] Using TMPDIR=${TMPDIR}"

ABLA_TAG="${METRICBANK_MODE}"
if [ -n "$K" ]; then ABLA_TAG="${ABLA_TAG}_k${K}"; fi
if [ -n "$N" ]; then ABLA_TAG="${ABLA_TAG}_n${N}"; fi
if [ "$NO_METRIC_CARDS" = "true" ]; then ABLA_TAG="${ABLA_TAG}_desc"; fi
# if [ "$RESIZED" = "true" ]; then ABLA_TAG="${ABLA_TAG}_resized"; fi # Removed to keep the same cache dirs for resized datasets

# Force per-run Torch/TRITON extension caches keyed by dataset/measure/seed/ablation
TORCH_EXTENSIONS_DIR="/nlp/scr3/nlp/20questions/torch_extensions/autometrics_ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
TRITON_CACHE_DIR="/nlp/scr3/nlp/20questions/triton_cache/autometrics_ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
export TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR

# Clean on startup (in case of retries) and create fresh dirs
rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
echo "[Ablation] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[Ablation] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

# Always wipe caches on job exit, cancellation, or failure
cleanup_torch_triton_cache() {
  rm -rf "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup_torch_triton_cache EXIT INT TERM

if [ "$RESIZED" = "true" ]; 
then 
  OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}_resized/${ABLA_TAG}"
else 
  OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}/${ABLA_TAG}"
fi
mkdir -p "$OUT_DIR"

echo "Running Qwen remote ablation"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Seed:    $SEED"
echo "Mode:    $METRICBANK_MODE"
echo "k:       ${K:-default}"
echo "n:       ${N:-default}"
echo "desc:    $NO_METRIC_CARDS"
echo "reindex: $FORCE_REINDEX"
echo "resized: $RESIZED"
echo "Model:   $MODEL_NAME"
echo "API:     $QWEN_API_BASE"
echo "Out:     $OUT_DIR"
echo "Run on:  $HOST"

# Respect SLURM's GPU allocation; do not override CUDA_VISIBLE_DEVICES here
export OPENAI_API_BASE="$QWEN_API_BASE"
export AUTOMETRICS_LM_GENERATOR="Qwen/Qwen3-32B"
export AUTOMETRICS_LM_JUDGE="Qwen/Qwen3-32B"

export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"

if [ -f "$OUT_DIR/score_pearson_${SEED}.txt" ] && [ -f "$OUT_DIR/log_${SEED}.json" ]; then
  echo "Results already exist in $OUT_DIR for seed $SEED. Skipping."
  exit 0
fi

PY_ARGS=(
  analysis/ablations/run_autometrics_ablation.py
  "$DATASET_NAME"
  "$TARGET_MEASURE"
  "$SEED"
  "$OUT_DIR"
  --model-name "$MODEL_NAME"
  --api-base "$QWEN_API_BASE"
  --metricbank "$METRICBANK_MODE"
)

if [ -n "$K" ]; then PY_ARGS+=( --k "$K" ); fi
# Pass --n if N is provided (integer). No 'auto' mode anymore.
if [[ -n "$N" && "$N" =~ ^[0-9]+$ ]]; then PY_ARGS+=( --n "$N" ); fi
if [ "$NO_METRIC_CARDS" = "true" ]; then PY_ARGS+=( --no-metric-cards ); fi
if [ "$FORCE_REINDEX" = "true" ]; then PY_ARGS+=( --force-reindex ); fi
if [ "$RESIZED" = "true" ]; then PY_ARGS+=( --resized ); fi

COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True python "${PY_ARGS[@]}"
STATUS=$?

if [ $STATUS -eq 0 ]; then
  echo "✅ Ablation completed"
else
  echo "❌ Ablation failed with status $STATUS"
fi


