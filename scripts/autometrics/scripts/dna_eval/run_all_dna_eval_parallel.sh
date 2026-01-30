#!/bin/bash

# Master DNAEval Correlation Analysis - Parallel Execution
# Usage:
#   bash scripts/dna_eval/run_all_dna_eval_parallel.sh [MODEL] [EXECUTION_MODE]
#   MODEL: gpt4o_mini (default), qwen3_32b
#   EXECUTION_MODE: local (default), slurm

set -e

MODEL=${1:-gpt4o_mini}
EXECUTION_MODE=${2:-local}

echo "Starting DNAEval correlation analysis..."
echo "Model: $MODEL"
echo "Execution mode: $EXECUTION_MODE"

# Ensure directories
mkdir -p results/main_runs/baselines/dna_eval_sub_results
mkdir -p logs/dna_eval

run_script() {
    local script_name=$1
    local log_file="logs/dna_eval/${script_name}_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting $script_name (logging to $log_file)..."
    if bash "scripts/dna_eval/$script_name" > "$log_file" 2>&1; then
        echo "✓ $script_name completed successfully"
        return 0
    else
        echo "✗ $script_name failed (check $log_file for details)"
        return 1
    fi
}

start_time=$(date +%s)

pids=()
scripts=()

if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    echo "Submitting SLURM jobs..."
    if [[ "$MODEL" == "gpt4o_mini" ]]; then
        job1=$(sbatch --parsable scripts/dna_eval/run_dna_eval_gpt4o_helpsteer.sh); echo "HelpSteer: $job1"
        job2=$(sbatch --parsable scripts/dna_eval/run_dna_eval_gpt4o_simplification.sh); echo "Simplification: $job2"
        job3=$(sbatch --parsable scripts/dna_eval/run_dna_eval_gpt4o_cogym.sh); echo "CoGym: $job3"
        job4=$(sbatch --parsable scripts/dna_eval/run_dna_eval_gpt4o_remaining.sh); echo "Remaining: $job4"
        echo "Monitor with: sacct -j $job1,$job2,$job3,$job4"
        exit 0
    elif [[ "$MODEL" == "qwen3_32b" ]]; then
        job1=$(sbatch --parsable scripts/dna_eval/run_dna_eval_qwen_helpsteer.sh); echo "Qwen HelpSteer: $job1"
        job2=$(sbatch --parsable scripts/dna_eval/run_dna_eval_qwen_simplification.sh); echo "Qwen Simplification: $job2"
        job3=$(sbatch --parsable scripts/dna_eval/run_dna_eval_qwen_cogym.sh); echo "Qwen CoGym: $job3"
        job4=$(sbatch --parsable scripts/dna_eval/run_dna_eval_qwen_remaining.sh); echo "Qwen Remaining: $job4"
        echo "Monitor with: sacct -j $job1,$job2,$job3,$job4"
        exit 0
    else
        echo "Error: SLURM submission only configured for gpt4o_mini and qwen3_32b"
        exit 1
    fi
else
    if [[ "$MODEL" == "gpt4o_mini" ]]; then
        run_script "run_dna_eval_gpt4o_helpsteer.sh" & pids+=($!); scripts+=("GPT4o_HelpSteer")
        run_script "run_dna_eval_gpt4o_simplification.sh" & pids+=($!); scripts+=("GPT4o_Simplification")
        run_script "run_dna_eval_gpt4o_cogym.sh" & pids+=($!); scripts+=("GPT4o_CoGym")
        run_script "run_dna_eval_gpt4o_remaining.sh" & pids+=($!); scripts+=("GPT4o_Remaining")
    elif [[ "$MODEL" == "qwen3_32b" ]]; then
        run_script "run_dna_eval_qwen_helpsteer.sh" & pids+=($!); scripts+=("Qwen_HelpSteer")
        run_script "run_dna_eval_qwen_simplification.sh" & pids+=($!); scripts+=("Qwen_Simplification")
        run_script "run_dna_eval_qwen_cogym.sh" & pids+=($!); scripts+=("Qwen_CoGym")
        run_script "run_dna_eval_qwen_remaining.sh" & pids+=($!); scripts+=("Qwen_Remaining")
    else
        echo "Error: Model '$MODEL' not supported for local execution"
        exit 1
    fi
fi

echo ""; echo "Waiting for all dataset analyses to complete..."; echo "Running scripts: ${scripts[@]}"

failed_scripts=(); completed_scripts=()
for i in "${!pids[@]}"; do
    pid=${pids[$i]}; script=${scripts[$i]}
    echo "Waiting for $script (PID: $pid)..."
    if wait $pid; then
        echo "✓ $script analysis completed successfully"; completed_scripts+=("$script")
    else
        echo "✗ $script analysis failed"; failed_scripts+=("$script")
    fi
done

end_time=$(date +%s); elapsed=$((end_time - start_time)); minutes=$((elapsed / 60)); seconds=$((elapsed % 60))

echo ""; echo "=== PARALLEL EXECUTION SUMMARY ==="
echo "Total execution time: ${minutes}m ${seconds}s"
echo "Completed successfully: ${#completed_scripts[@]} scripts"
echo "Failed: ${#failed_scripts[@]} scripts"
if [ ${#completed_scripts[@]} -gt 0 ]; then echo "Successful scripts: ${completed_scripts[@]}"; fi
if [ ${#failed_scripts[@]} -gt 0 ]; then echo "Failed scripts: ${failed_scripts[@]}"; echo "Check logs/dna_eval/ for details"; fi

echo ""; echo "🎉 Parallel DNAEval correlation analysis completed!"


