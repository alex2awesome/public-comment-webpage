#!/bin/bash

# Parallel Fine-tuned Metric Correlation Analysis
# This script runs fine-tuned metric correlation analysis across all datasets in parallel
# using SLURM job submission or local parallel execution.

set -e

# Default settings
MODE="slurm"  # Can be "slurm" or "local"

# Parse command line arguments
if [[ $# -gt 0 ]]; then
    MODE="$1"
fi

# Validate mode
if [[ "$MODE" != "slurm" && "$MODE" != "local" ]]; then
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [slurm|local]"
    echo ""
    echo "Modes:"
    echo "  slurm - Submit jobs to SLURM scheduler (recommended for cluster)"
    echo "  local - Run jobs locally in parallel (for testing/development)"
    exit 1
fi

# Record start time
start_time=$(date +%s)

echo "=================================================================="
echo "     Parallel Fine-tuned Metric Correlation Analysis"
echo "=================================================================="
echo ""
echo "Mode: $MODE"
echo "Start time: $(date)"
echo ""

# Helper function for local execution
run_script() {
    script_name="$1"
    echo "Running $script_name..."
    bash "scripts/finetune/$script_name"
}

# Initialize arrays for tracking parallel jobs
pids=()
scripts=()

# Create necessary directories
mkdir -p results/main_runs/baselines/finetune_sub_results
mkdir -p scripts/finetune/logs

if [[ "$MODE" == "slurm" ]]; then
    # SLURM submission mode
    echo "=== SLURM Job Submission ==="
    echo ""
    echo "Submitting fine-tuning correlation analysis jobs..."
    echo "Each job will process different dataset groups:"
    echo "  - HelpSteer: 5 measures (24h runtime)"
    echo "  - HelpSteer2: 5 measures (24h runtime)"
    echo "  - Simplification: SimpDA, SimpEval (16h runtime)"
    echo "  - CoGym: 6 datasets, 9 measures total (16h runtime)"
    echo "  - Remaining: 5 datasets, 11 measures total (18h runtime)"
    echo ""
    
    # Submit jobs to SLURM and capture job IDs
    echo "Submitting HelpSteer job..."
    job1=$(sbatch --parsable scripts/finetune/run_finetune_helpsteer.sh)
    echo "  Job ID: $job1"
    
    echo "Submitting HelpSteer2 job..."
    job2=$(sbatch --parsable scripts/finetune/run_finetune_helpsteer2.sh)
    echo "  Job ID: $job2"
    
    echo "Submitting Simplification job..."
    job3=$(sbatch --parsable scripts/finetune/run_finetune_simplification.sh)
    echo "  Job ID: $job3"
    
    echo "Submitting CoGym job..."
    job4=$(sbatch --parsable scripts/finetune/run_finetune_cogym.sh)
    echo "  Job ID: $job4"
    
    echo "Submitting Remaining datasets job..."
    job5=$(sbatch --parsable scripts/finetune/run_finetune_remaining.sh)
    echo "  Job ID: $job5"
    
    echo ""
    echo "All jobs submitted. Monitor with:"
    echo "  squeue -u $USER"
    echo "  sacct -j $job1,$job2,$job3,$job4,$job5"
    echo ""
    echo "Jobs will email results to mryan0@stanford.edu when complete."
    echo "Results will be in results/main_runs/baselines/ when finished."
    echo ""
    echo "Expected total training runs: ~170 (34 dataset-measure combinations × 5 seeds)"
    echo "Each job will save dataset-specific results to results/main_runs/baselines/finetune_sub_results/"
    echo ""
    echo "To merge final results after all jobs complete, run:"
    echo "  python scripts/finetune/merge_finetune_results.py --correlation all --verbose"
    exit 0
else
    # Local parallel execution
    echo "=== Local Parallel Execution ==="
    echo ""
    echo "WARNING: Local execution will run all fine-tuning jobs in parallel"
    echo "This will require significant computational resources:"
    echo "  - 5 parallel processes"
    echo "  - ~170 total ModernBERT fine-tuning runs"
    echo "  - Multiple GPUs recommended"
    echo "  - High memory usage (each process uses ~16-100GB GPU memory)"
    echo ""
    echo "Launching dataset analyses in parallel..."
    
    # Launch HelpSteer
    run_script "run_finetune_helpsteer.sh" &
    pids+=($!)
    scripts+=("HelpSteer")
    
    # Launch HelpSteer2
    run_script "run_finetune_helpsteer2.sh" &
    pids+=($!)
    scripts+=("HelpSteer2")
    
    # Launch Simplification datasets
    run_script "run_finetune_simplification.sh" &
    pids+=($!)
    scripts+=("Simplification")
    
    # Launch CoGym datasets
    run_script "run_finetune_cogym.sh" &
    pids+=($!)
    scripts+=("CoGym")
    
    # Launch remaining datasets
    run_script "run_finetune_remaining.sh" &
    pids+=($!)
    scripts+=("Remaining")
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
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo ""
echo "=== PARALLEL EXECUTION SUMMARY ==="
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"
echo "Completed successfully: ${#completed_scripts[@]} scripts"
echo "Failed: ${#failed_scripts[@]} scripts"

if [ ${#completed_scripts[@]} -gt 0 ]; then
    echo "Successful scripts: ${completed_scripts[@]}"
fi

if [ ${#failed_scripts[@]} -gt 0 ]; then
    echo "Failed scripts: ${failed_scripts[@]}"
    echo "Check log files in scripts/finetune/logs/ for details"
fi

# Merge results if we have any successful runs
if [ ${#completed_scripts[@]} -gt 0 ]; then
    echo ""
    echo "=== MERGING RESULTS ==="
    echo "Combining all dataset-specific results into final files..."
    
    # Run the merge script
    if python scripts/finetune/merge_finetune_results.py --correlation all --verbose; then
        echo "✓ Results merged successfully!"
        
        echo ""
        echo "=== FINAL RESULTS LOCATION ==="
        echo "Merged results:"
        for corr in kendall pearson spearman; do
            final_file="results/main_runs/baselines/finetune_${corr}.csv"
            if [ -f "$final_file" ]; then
                echo "  $final_file"
            fi
        done
        
        echo ""
        echo "Individual dataset results:"
        echo "  results/main_runs/baselines/finetune_sub_results/"
        
        echo ""
        echo "Log files:"
        echo "  scripts/finetune/logs/"
        
        echo ""
        echo "Trained models saved to:"
        echo "  $AUTOMETRICS_MODEL_DIR (or user data directory if not set)"
        
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
echo "🎉 Parallel Fine-tuned Metric correlation analysis completed!"
echo ""
echo "Summary of datasets processed:"
echo "  - HelpSteer: 5 measures"
echo "  - HelpSteer2: 5 measures"
echo "  - SimpDA/SimpEval: 4 measures total"
echo "  - CoGym (6 variants): 9 measures total"
echo "  - Remaining (5 datasets): 11 measures total"
echo "  - TOTAL: 34 dataset-measure combinations"
echo "  - TOTAL TRAINING RUNS: 170 (34 combinations × 5 seeds)"
echo ""
echo "Next steps:"
echo "1. Review the merged CSV files in results/main_runs/baselines/"
echo "2. Check individual dataset results in results/main_runs/baselines/finetune_sub_results/"
echo "3. Examine log files in scripts/finetune/logs/ if needed"
echo "4. Compare results with static metrics and LLM judges"
echo ""
echo "To run test version with limited datasets/seeds:"
echo "  bash scripts/finetune/test_finetune.sh"
echo ""
echo "Available modes: local, slurm" 