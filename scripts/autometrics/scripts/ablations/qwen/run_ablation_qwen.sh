#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=340GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_ablation
#SBATCH --output=logs/qwen_ablation_%j.out
#SBATCH --error=logs/qwen_ablation_%j.err
#SBATCH --constraint=141G
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Single ablation run for Qwen3-32B (server on GPU0, pipeline on GPU1)
# Required env vars:
#   DATASET_NAME, TARGET_MEASURE, SEED
# Optional env vars:
#   METRICBANK_MODE (full|existing_only|generated_only, default full)
#   K, N
#   NO_METRIC_CARDS (true|false), FORCE_REINDEX (true|false)
#   OUTPUT_ROOT (default results/ablations/qwen)
#   RESIZED (true|false)

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

model="Qwen/Qwen3-32B"

assign_port() {
    local dataset_name="$1"; local target_measure="$2"
    local base_port=8000
    local hash_value=$(echo "${dataset_name}_${target_measure}" | md5sum | cut -d' ' -f1 | head -c 8)
    local hash_decimal=$((16#$hash_value))
    local port_offset=$((hash_decimal % 100))
    echo $((base_port + port_offset))
}

if [ -z "${DATASET_NAME:-}" ]; then echo "❌ DATASET_NAME required"; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then echo "❌ TARGET_MEASURE required"; exit 1; fi
if [ -z "${SEED:-}" ]; then echo "❌ SEED required"; exit 1; fi
if [ -z "${OPENAI_API_KEY:-}" ]; then echo "❌ OPENAI_API_KEY required (unused for server but required by script)"; exit 1; fi

METRICBANK_MODE=${METRICBANK_MODE:-"full"}
K=${K:-""}
N=${N:-""}
NO_METRIC_CARDS=${NO_METRIC_CARDS:-"false"}
FORCE_REINDEX=${FORCE_REINDEX:-"false"}
RESIZED=${RESIZED:-"false"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ablations/main_ablations/qwen_run2"}

mkdir -p logs

port=$(assign_port "$DATASET_NAME" "$TARGET_MEASURE")

ABLA_TAG="${METRICBANK_MODE}"
if [ -n "$K" ]; then ABLA_TAG="${ABLA_TAG}_k${K}"; fi
if [ -n "$N" ]; then ABLA_TAG="${ABLA_TAG}_n${N}"; fi
if [ "$NO_METRIC_CARDS" = "true" ]; then ABLA_TAG="${ABLA_TAG}_desc"; fi
# if [ "$RESIZED" = "true" ]; then ABLA_TAG="${ABLA_TAG}_resized"; fi # Removed to keep the same cache dirs for resized datasets

OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_${TARGET_MEASURE}/${ABLA_TAG}"
if [ "$RESIZED" = "true" ]; then OUT_DIR="${OUT_DIR}_resized"; fi
mkdir -p "$OUT_DIR"

echo "Starting Qwen server on port $port…"
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path ${model} --port ${port} --host 0.0.0.0 --tp 1 --dtype bfloat16 --mem-fraction-static 0.8 --trust-remote-code > /dev/null 2>&1 &

TIMEOUT=90
START_TIME=$(date +%s)
while ! curl -s http://localhost:${port}/v1/get_model_info > /dev/null; do
  echo "Waiting for server to start…"; sleep 20
  if [ $(( $(date +%s) - START_TIME )) -gt $((TIMEOUT * 60)) ]; then echo "Server timeout"; exit 1; fi
done

conda activate autometrics

API_BASE=http://localhost:${port}/v1

export CUDA_VISIBLE_DEVICES=1
export OPENAI_API_KEY="None"
export OPENAI_API_BASE="$API_BASE"
export AUTOMETRICS_LM_GENERATOR="Qwen/Qwen3-32B"
export AUTOMETRICS_LM_JUDGE="Qwen/Qwen3-32B"

export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"

if [ -f "$OUT_DIR/score_pearson_${SEED}.txt" ] && [ -f "$OUT_DIR/log_${SEED}.json" ]; then
  echo "Results already exist in $OUT_DIR for seed $SEED. Skipping."; pkill -f "sglang.launch_server"; exit 0
fi

PY_ARGS=(
  analysis/ablations/run_autometrics_ablation.py
  "$DATASET_NAME"
  "$TARGET_MEASURE"
  "$SEED"
  "$OUT_DIR"
  --model-name "$model"
  --api-base "$API_BASE"
  --metricbank "$METRICBANK_MODE"
)

if [ -n "$K" ]; then PY_ARGS+=( --k "$K" ); fi
if [ -n "$N" ]; then PY_ARGS+=( --n "$N" ); fi
if [ "$NO_METRIC_CARDS" = "true" ]; then PY_ARGS+=( --no-metric-cards ); fi
if [ "$FORCE_REINDEX" = "true" ]; then PY_ARGS+=( --force-reindex ); fi
if [ "$RESIZED" = "true" ]; then PY_ARGS+=( --resized ); fi

python "${PY_ARGS[@]}"
STATUS=$?

pkill -f "sglang.launch_server"

if [ $STATUS -eq 0 ]; then
  echo "✅ Ablation completed"
else
  echo "❌ Ablation failed with status $STATUS"
fi


