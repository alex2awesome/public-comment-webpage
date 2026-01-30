#!/bin/bash

# Master LLM Judge Correlation Analysis - Parallel Execution
# This script launches all dataset-specific scripts in parallel for maximum throughput
# Usage: 
#   bash scripts/llm_judge/run_all_llm_judge_parallel.sh [MODEL] [EXECUTION_MODE]
#   MODEL: gpt4o_mini (default), qwen3_32b, llama3_70b
#   EXECUTION_MODE: local (default), slurm

set -e

# Model to run (can be overridden)
MODEL=${1:-gpt4o_mini}
EXECUTION_MODE=${2:-local}

echo "Starting LLM Judge correlation analysis..."
echo "Model: $MODEL"
echo "Execution mode: $EXECUTION_MODE"

if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    echo "Will submit SLURM jobs for cluster execution"
else
    echo "Will run locally with parallel execution"
fi

# Ensure all directories exist
mkdir -p results/main_runs/baselines/llm_judge_sub_results
mkdir -p logs/llm_judge

# Function to run a script and capture its output
run_script() {
    local script_name=$1
    local log_file="logs/llm_judge/${script_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting $script_name (logging to $log_file)..."
    
    # Run the script and capture output
    if bash "scripts/llm_judge/$script_name" > "$log_file" 2>&1; then
        echo "✓ $script_name completed successfully"
        return 0
    else
        echo "✗ $script_name failed (check $log_file for details)"
        return 1
    fi
}

# Start timestamp
start_time=$(date +%s)

# Launch all dataset-specific scripts in parallel
echo ""
echo "Launching dataset-specific analysis scripts in parallel..."

# Array to store background process PIDs
pids=()
scripts=()

# Choose execution method
if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    echo "Submitting SLURM jobs..."
    
    # Submit jobs to SLURM and capture job IDs
    if [[ "$MODEL" == "gpt4o_mini" ]]; then
        echo "Submitting HelpSteer job..."
        job1=$(sbatch --parsable scripts/llm_judge/run_llm_judge_gpt4o_helpsteer.sh)
        echo "  Job ID: $job1"
        
        echo "Submitting Simplification job..."
        job2=$(sbatch --parsable scripts/llm_judge/run_llm_judge_gpt4o_simplification.sh)
        echo "  Job ID: $job2"
        
        echo "Submitting CoGym job..."
        job3=$(sbatch --parsable scripts/llm_judge/run_llm_judge_gpt4o_cogym.sh)
        echo "  Job ID: $job3"
        
        echo "Submitting Remaining datasets job..."
        job4=$(sbatch --parsable scripts/llm_judge/run_llm_judge_gpt4o_remaining.sh)
        echo "  Job ID: $job4"
        
        echo ""
        echo "All jobs submitted. Monitor with:"
        echo "  squeue -u $USER"
        echo "  sacct -j $job1,$job2,$job3,$job4"
        echo ""
        echo "Jobs will email results to mryan0@stanford.edu when complete."
        echo "Results will be in results/main_runs/baselines/ when finished."
        exit 0
    elif [[ "$MODEL" == "qwen3_32b" ]]; then
        echo "Submitting Qwen HelpSteer job..."
        job1=$(sbatch --parsable scripts/llm_judge/run_llm_judge_qwen_helpsteer.sh)
        echo "  Job ID: $job1"
        
        echo "Submitting Qwen Simplification job..."
        job2=$(sbatch --parsable scripts/llm_judge/run_llm_judge_qwen_simplification.sh)
        echo "  Job ID: $job2"
        
        echo "Submitting Qwen CoGym job..."
        job3=$(sbatch --parsable scripts/llm_judge/run_llm_judge_qwen_cogym.sh)
        echo "  Job ID: $job3"
        
        echo "Submitting Qwen Remaining datasets job..."
        job4=$(sbatch --parsable scripts/llm_judge/run_llm_judge_qwen_remaining.sh)
        echo "  Job ID: $job4"
        
        echo ""
        echo "All jobs submitted. Monitor with:"
        echo "  squeue -u $USER"
        echo "  sacct -j $job1,$job2,$job3,$job4"
        echo ""
        echo "Jobs will email results to mryan0@stanford.edu when complete."
        echo "Results will be in results/main_runs/baselines/ when finished."
        exit 0
    else
        echo "Error: SLURM submission only configured for gpt4o_mini and qwen3_32b currently"
        echo "Available models: gpt4o_mini, qwen3_32b"
        exit 1
    fi
else
    # Local parallel execution
    if [[ "$MODEL" == "gpt4o_mini" ]]; then
        run_script "run_llm_judge_gpt4o_helpsteer.sh" &
        pids+=($!)
        scripts+=("GPT4o_HelpSteer")
        
        # Launch Simplification datasets
        run_script "run_llm_judge_gpt4o_simplification.sh" &
        pids+=($!)
        scripts+=("GPT4o_Simplification")
        
        # Launch CoGym datasets
        run_script "run_llm_judge_gpt4o_cogym.sh" &
        pids+=($!)
        scripts+=("GPT4o_CoGym")
        
        # Launch remaining datasets
        run_script "run_llm_judge_gpt4o_remaining.sh" &
        pids+=($!)
        scripts+=("GPT4o_Remaining")
    elif [[ "$MODEL" == "qwen3_32b" ]]; then
        run_script "run_llm_judge_qwen_helpsteer.sh" &
        pids+=($!)
        scripts+=("Qwen_HelpSteer")
        
        # Launch Simplification datasets
        run_script "run_llm_judge_qwen_simplification.sh" &
        pids+=($!)
        scripts+=("Qwen_Simplification")
        
        # Launch CoGym datasets
        run_script "run_llm_judge_qwen_cogym.sh" &
        pids+=($!)
        scripts+=("Qwen_CoGym")
        
        # Launch remaining datasets
        run_script "run_llm_judge_qwen_remaining.sh" &
        pids+=($!)
        scripts+=("Qwen_Remaining")
    else
        echo "Error: Model '$MODEL' not supported for local execution"
        echo "Available models: gpt4o_mini, qwen3_32b"
        exit 1
    fi
fi

# Wait for all background processes to complete
echo ""
echo "Waiting for all dataset analyses to complete..."
echo "Running scripts: ${scripts[@]}"

failed_scripts=()
completed_scripts=()

for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    script=${scripts[$i]}
    
    echo "Waiting for $script (PID: $pid)..."
    
    if wait $pid; then
        echo "✓ $script analysis completed successfully"
        completed_scripts+=("$script")
    else
        echo "✗ $script analysis failed"
        failed_scripts+=("$script")
    fi
done

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo ""
echo "=== PARALLEL EXECUTION SUMMARY ==="
echo "Total execution time: ${minutes}m ${seconds}s"
echo "Completed successfully: ${#completed_scripts[@]} scripts"
echo "Failed: ${#failed_scripts[@]} scripts"

if [ ${#completed_scripts[@]} -gt 0 ]; then
    echo "Successful scripts: ${completed_scripts[@]}"
fi

if [ ${#failed_scripts[@]} -gt 0 ]; then
    echo "Failed scripts: ${failed_scripts[@]}"
    echo "Check log files in logs/llm_judge/ for details"
fi

# Merge results if we have any successful runs
if [ ${#completed_scripts[@]} -gt 0 ]; then
    echo ""
    echo "=== MERGING RESULTS ==="
    echo "Combining all dataset-specific results into final files..."
    
    # Run the merge script
    if python scripts/llm_judge/merge_llm_judge_results.py --model "$MODEL" --correlation all --verbose; then
        echo "✓ Results merged successfully!"
        
        echo ""
        echo "=== FINAL RESULTS LOCATION ==="
        echo "Merged results:"
        for corr in kendall pearson spearman; do
            final_file="results/main_runs/baselines/llm_judge_${MODEL}_${corr}.csv"
            if [ -f "$final_file" ]; then
                echo "  $final_file"
            fi
        done
        
        echo ""
        echo "Individual dataset results:"
        echo "  results/main_runs/baselines/llm_judge_sub_results/"
        
        echo ""
        echo "Log files:"
        echo "  logs/llm_judge/"
        
    else
        echo "✗ Failed to merge results"
        exit 1
    fi
else
    echo ""
    echo "✗ No successful runs to merge"
    exit 1
fi

echo ""
echo "🎉 Parallel LLM Judge correlation analysis completed!"
echo ""
echo "Next steps:"
echo "1. Review the merged CSV files in results/main_runs/baselines/"
echo "2. Check individual dataset results in results/main_runs/baselines/llm_judge_sub_results/"
echo "3. Examine log files in logs/llm_judge/ if needed"
echo ""
echo "To run additional models, use:"
echo "  bash scripts/llm_judge/run_all_llm_judge_parallel.sh qwen3_32b local"
echo "  bash scripts/llm_judge/run_all_llm_judge_parallel.sh qwen3_32b slurm"
echo ""
echo "Available models: gpt4o_mini, qwen3_32b"
echo "Available modes: local, slurm" 