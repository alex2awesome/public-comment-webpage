#!/bin/bash

#SBATCH --job-name=gpt5mini_autometrics
#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=100G
#SBATCH --partition=john
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=logs/gpt5mini_autometrics_%j.out
#SBATCH --error=logs/gpt5mini_autometrics_%j.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Usage:
#   export DATASET_NAME="EvalGenProduct"
#   export TARGET_MEASURE="quality"
#   export SEEDS="42 43 44 45 46"
#   export API_BASE="https://api.openai.com/v1"   # Or your proxy base
#   sbatch scripts/main_autometrics/gpt5mini/run_autometrics_gpt5mini.sh

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

cd /nlp/scr2/nlp/personal-rm/autometrics

set -euo pipefail

# Validate required env vars
if [ -z "${DATASET_NAME:-}" ]; then
  echo "ERROR: DATASET_NAME is required"; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then
  echo "ERROR: TARGET_MEASURE is required"; exit 1; fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is required in environment"; exit 1; fi

# Defaults
SEEDS=${SEEDS:-"42 43 44 45 46"}
API_BASE=${API_BASE:-"https://api.openai.com/v1"}
MODEL_NAME=${MODEL_NAME:-"openai/gpt-5-mini"}

echo "Launching Autometrics (GPT-5-mini)"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Seeds:   $SEEDS"
echo "Model:   $MODEL_NAME"
echo "API:     $API_BASE"

# Ensure logs dir exists and distinct output dirs from qwen
mkdir -p logs
BASE_RESULTS_DIR="results/main_runs/autometrics/gpt5mini/${DATASET_NAME}_${TARGET_MEASURE}"
mkdir -p "$BASE_RESULTS_DIR"

# GPU visibility: allow all 4 GPUs to the job; internal code will shard/allocate
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Iterate seeds
SUCCESSFUL_SEEDS=()
FAILED_SEEDS=()

for seed in $SEEDS; do
  echo "\n==================== Seed $seed ===================="

  # Seed-specific caches (distinct namespace)
  export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}_seed${seed}"
  export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}_seed${seed}"

  # Run
  python analysis/main_experiments/run_main_autometrics.py \
    "$DATASET_NAME" \
    "$TARGET_MEASURE" \
    "$seed" \
    "$BASE_RESULTS_DIR" \
    --model-name "$MODEL_NAME" \
    --api-base "$API_BASE" \
    --skip-mipro

  STATUS=$?
  if [ $STATUS -eq 0 ]; then
    echo "✅ Seed $seed completed"
    SUCCESSFUL_SEEDS+=("$seed")
  else
    echo "❌ Seed $seed failed with status $STATUS"
    FAILED_SEEDS+=("$seed")
  fi
done

echo "\n======= SUMMARY (GPT-5-mini) ======="
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Results: $BASE_RESULTS_DIR"
echo "Successful seeds: ${#SUCCESSFUL_SEEDS[@]} -> ${SUCCESSFUL_SEEDS[*]:-none}"
echo "Failed seeds:     ${#FAILED_SEEDS[@]} -> ${FAILED_SEEDS[*]:-none}"

for corr in pearson spearman kendall; do
  echo "\n$corr correlations:"
  for seed in ${SUCCESSFUL_SEEDS[@]:-}; do
    f="$BASE_RESULTS_DIR/score_${corr}_${seed}.txt"
    if [ -f "$f" ]; then
      printf "  Seed %2s: %s\n" "$seed" "$(cat "$f")"
    fi
  done
done


