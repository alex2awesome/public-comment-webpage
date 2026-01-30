#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=340GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=qwen_autometrics
#SBATCH --output=logs/qwen_autometrics_%j.out
#SBATCH --error=logs/qwen_autometrics_%j.err
#SBATCH --constraint=141G
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Unified script for autometrics experiments with Qwen3-32B
# GPU Strategy: GPU 0 for Qwen server, GPU 1 for autometrics pipeline
#
# Usage:
#   export DATASET_NAME="CoGymTravelOutcome"
#   export TARGET_MEASURE="agentRating"
#   export SEEDS="42 43 44 45 46"
#   sbatch run_autometrics_qwen.sh

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics



# Server configuration
model="Qwen/Qwen3-32B"
model_nickname="qwen3_32b"

# Port assignment function
assign_port() {
    local dataset_name="$1"
    local target_measure="$2"
    
    # Base port range for autometrics experiments (8000-8999)
    local base_port=8000
    
    # Create a hash from dataset and target measure for consistent assignment
    local hash_input="${dataset_name}_${target_measure}"
    local hash_value=$(echo "$hash_input" | md5sum | cut -d' ' -f1 | head -c 8)
    local hash_decimal=$((16#$hash_value))
    
    # Assign port based on hash only (same experiment = same port)
    # Use modulo to keep within 8000-8999 range
    local port_offset=$((hash_decimal % 100))  # 0-99 offset
    local final_port=$((base_port + port_offset))
    
    echo $final_port
}

# Assign port for this experiment (same for all seeds)
port=$(assign_port "$DATASET_NAME" "$TARGET_MEASURE")

# Validate required environment variables
if [ -z "$DATASET_NAME" ]; then
    echo "❌ Error: DATASET_NAME environment variable is required"
    echo "Usage: export DATASET_NAME='CoGymTravelOutcome' && sbatch run_autometrics_qwen.sh"
    exit 1
fi

if [ -z "$TARGET_MEASURE" ]; then
    echo "❌ Error: TARGET_MEASURE environment variable is required"
    echo "Usage: export TARGET_MEASURE='agentRating' && sbatch run_autometrics_qwen.sh"
    exit 1
fi

# Set default seeds if not provided
if [ -z "$SEEDS" ]; then
    SEEDS="42 43 44 45 46"
    echo "Using default seeds: $SEEDS"
fi

# Output directory
OUTPUT_DIR="results/main_runs/autometrics/qwen_run2/${DATASET_NAME}_${TARGET_MEASURE}"

echo "Starting Autometrics Experiments with Qwen3-32B..."
echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"
echo "Port: $port"

# Start Qwen server on GPU 0
echo "Starting Qwen server on port $port..."
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path ${model} --port ${port} --host 0.0.0.0 --tp 1 --dtype bfloat16 --mem-fraction-static 0.8 --trust-remote-code > /dev/null 2>&1 &

# Wait for server to be ready
TIMEOUT=90
START_TIME=$(date +%s)

while ! curl -s http://localhost:${port}/v1/get_model_info > /dev/null; do
    echo "Waiting for server to start..."
    sleep 20
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED_TIME -gt $((TIMEOUT * 60)) ]; then
        echo "Timeout reached after 90 minutes. Killing job."
        exit 1
    fi
done

echo "Server is up and running!"

# Switch to autometrics environment
conda activate autometrics

# Set API base URL
API_BASE=http://localhost:${port}/v1

# Set environment for autometrics (use GPU 1)
export CUDA_VISIBLE_DEVICES=1
export OPENAI_API_KEY="None"
export OPENAI_API_BASE="$API_BASE"
export AUTOMETRICS_LM_GENERATOR="Qwen/Qwen3-32B"
export AUTOMETRICS_LM_JUDGE="Qwen/Qwen3-32B"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Track results
SUCCESSFUL_SEEDS=()
FAILED_SEEDS=()

# Run experiments for each seed
for seed in $SEEDS; do
    echo ""
    echo "=============================================================================="
    echo "Running seed $seed..."
    echo "=============================================================================="
    
    # Mirror ablation cache naming; default to K=30, n=5 cache
    ABLA_TAG="${MAIN_ABLATION_TAG:-full_k30_n5}"
    export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${seed}"
    export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${seed}"
    
    echo "Using DSPY cache: $DSPY_CACHEDIR"
    echo "Using Autometrics cache: $AUTOMETRICS_CACHE_DIR"
    # Skip if already completed
    if [ -f "$OUTPUT_DIR/score_pearson_${seed}.txt" ] && [ -f "$OUTPUT_DIR/log_${seed}.json" ]; then
        echo "✅ Seed $seed already completed. Skipping."
        continue
    fi

    # Run the autometrics experiment
    python analysis/main_experiments/run_main_autometrics.py \
        "$DATASET_NAME" \
        "$TARGET_MEASURE" \
        "$seed" \
        "$OUTPUT_DIR" \
        --model-name "$model" \
        --api-base "$API_BASE"
    
    EXPERIMENT_EXIT_CODE=$?
    
    if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
        echo "✅ Seed $seed completed successfully!"
        SUCCESSFUL_SEEDS+=("$seed")
        
        # Display results if available
        if [ -f "$OUTPUT_DIR/score_pearson_$seed.txt" ]; then
            echo "📈 Correlation Scores for seed $seed:"
            echo "   Pearson:  $(cat "$OUTPUT_DIR/score_pearson_$seed.txt")"
            echo "   Spearman: $(cat "$OUTPUT_DIR/score_spearman_$seed.txt")"
            echo "   Kendall:  $(cat "$OUTPUT_DIR/score_kendall_$seed.txt")"
        fi
    else
        echo "❌ Seed $seed failed with exit code: $EXPERIMENT_EXIT_CODE"
        FAILED_SEEDS+=("$seed")
    fi
done

echo ""
echo "Autometrics experiments completed!"

# Cleanup: Kill the server
pkill -f "sglang.launch_server"

echo ""
echo "=============================================================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Total seeds attempted: $(echo $SEEDS | wc -w)"
echo "Successful seeds: ${#SUCCESSFUL_SEEDS[@]}"
echo "Failed seeds: ${#FAILED_SEEDS[@]}"

if [ ${#SUCCESSFUL_SEEDS[@]} -gt 0 ]; then
    echo ""
    echo "✅ Successful seeds: ${SUCCESSFUL_SEEDS[*]}"
fi

if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed seeds: ${FAILED_SEEDS[*]}"
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"

# Display summary of all successful results
if [ ${#SUCCESSFUL_SEEDS[@]} -gt 0 ]; then
    echo ""
    echo "📊 SUMMARY OF ALL SUCCESSFUL RESULTS:"
    echo "====================================="
    
    for corr_type in pearson spearman kendall; do
        echo ""
        echo "$(echo $corr_type | tr '[:lower:]' '[:upper:]') Correlation:"
        for seed in "${SUCCESSFUL_SEEDS[@]}"; do
            if [ -f "$OUTPUT_DIR/score_${corr_type}_$seed.txt" ]; then
                score=$(cat "$OUTPUT_DIR/score_${corr_type}_$seed.txt")
                printf "  Seed %2s: %8.4f\n" "$seed" "$score"
            fi
        done
    done
fi

echo ""
echo "=============================================================================="
