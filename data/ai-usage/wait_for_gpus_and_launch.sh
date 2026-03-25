#!/usr/bin/env bash
# wait_for_gpus_and_launch.sh
#
# Polls every 60 seconds until N GPUs have < 100 MB memory used,
# then launches the generate → estimate → infer pipeline.
#
# Usage:
#   bash wait_for_gpus_and_launch.sh [N_GPUS] [GPU_IDS]
#
# Examples:
#   bash wait_for_gpus_and_launch.sh 1 2          # wait for GPU 2 to be free
#   bash wait_for_gpus_and_launch.sh 2 "0,1"      # wait for GPUs 0 and 1 to be free
#   bash wait_for_gpus_and_launch.sh               # default: 1 GPU, any GPU

set -euo pipefail

N_GPUS_NEEDED="${1:-1}"
GPU_IDS="${2:-}"  # empty = auto-detect all GPUs
POLL_INTERVAL=60
MEM_THRESHOLD_MB=100

PYTHON=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python3
WORKDIR=/lfs/skampere3/0/alexspan/regulations-demo
SCRIPT=data/ai-usage/corpus_ai_usage.py
OUTDIR=data/ai-usage-generations
LOG="$WORKDIR/generate.log"

# If no GPU IDs specified, discover all available GPUs
if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    echo "$(date): Auto-detected GPUs: [$GPU_IDS]"
fi

echo "$(date): Waiting for $N_GPUS_NEEDED GPU(s) from [$GPU_IDS] to have < ${MEM_THRESHOLD_MB}MB used..."
echo "$(date): Checking every ${POLL_INTERVAL}s. Log will be written to $LOG"

find_free_gpus() {
    # Returns comma-separated list of free GPU IDs
    local free_ids=""
    for gpu_id in $(echo "$GPU_IDS" | tr ',' ' '); do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
        if [ -z "$mem_used" ]; then
            continue
        fi
        if [ "$mem_used" -lt "$MEM_THRESHOLD_MB" ]; then
            if [ -z "$free_ids" ]; then
                free_ids="$gpu_id"
            else
                free_ids="$free_ids,$gpu_id"
            fi
        else
            echo "  GPU $gpu_id: ${mem_used}MB used (busy)" >&2
        fi
    done
    echo "$free_ids"
}

SELECTED_GPUS=""
while true; do
    free_ids=$(find_free_gpus)
    n_free=$(echo "$free_ids" | tr ',' '\n' | grep -c '[0-9]' || true)

    if [ "$n_free" -ge "$N_GPUS_NEEDED" ]; then
        # Take only the first N_GPUS_NEEDED
        SELECTED_GPUS=$(echo "$free_ids" | tr ',' '\n' | head -n "$N_GPUS_NEEDED" | tr '\n' ',' | sed 's/,$//')
        echo "$(date): $n_free GPU(s) free. Using GPU(s): [$SELECTED_GPUS]. Launching pipeline!"
        break
    fi

    echo "$(date): Only $n_free/$N_GPUS_NEEDED GPU(s) free. Waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

cd "$WORKDIR"
export HOME=/lfs/skampere3/0/alexspan
export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"

echo "$(date): === Starting pipeline (CUDA_VISIBLE_DEVICES=$SELECTED_GPUS) ===" | tee "$LOG"

# Step 1: Generate (GPU)
echo "$(date): Starting generate..." | tee -a "$LOG"
$PYTHON "$SCRIPT" generate \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --offline --dedup --sample-per-agency 200 \
    --doc-types public_submission notice rule proposed_rule \
    --batch-size 128 --max-tokens 2048 --max-model-len 8192 \
    --output-dir "$OUTDIR" \
    >> "$LOG" 2>&1

echo "$(date): Generate complete." | tee -a "$LOG"

# Step 2: Estimate (CPU)
echo "$(date): Starting estimate..." | tee -a "$LOG"
$PYTHON "$SCRIPT" estimate \
    --dedup --doc-types public_submission notice rule proposed_rule \
    --ai-corpus-dir "$OUTDIR" \
    --output-dir "$OUTDIR/ai_usage_distributions" \
    >> "$LOG" 2>&1

echo "$(date): Estimate complete." | tee -a "$LOG"

# Step 3: Infer (CPU)
echo "$(date): Starting infer..." | tee -a "$LOG"
$PYTHON "$SCRIPT" infer \
    --doc-types public_submission notice rule proposed_rule \
    --distribution-dir "$OUTDIR/ai_usage_distributions" \
    --stratify-by agency quarter \
    --output "$OUTDIR/ai_usage_results.csv.gz" \
    >> "$LOG" 2>&1

echo "$(date): === Pipeline complete ===" | tee -a "$LOG"
echo "$(date): Results at $OUTDIR/ai_usage_results.csv.gz"
