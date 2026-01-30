#!/bin/bash

echo "============================================"
echo "Master Metric Generation Benchmark Runner"
echo "============================================"
echo "This script will submit all metric generation benchmark jobs to SLURM"
echo "for both GPT-4o-mini and Qwen3-32B models with separate cache directories."
echo ""

# Check if we're running from the right directory
if [ ! -f "analysis/ablations/run_metric_generation_benchmark.py" ]; then
    echo "ERROR: Please run this script from the autometrics root directory"
    exit 1
fi

# Create logs directory
mkdir -p scripts/metric_generation/logs
mkdir -p results/ablations/metric_generation/sub_results

echo "Submitting metric generation benchmark jobs for both models..."
echo ""

echo "Dataset coverage overview:"
echo "  - HelpSteer: 5 dataset-measure combinations (separate job for speed)"
echo "  - HelpSteer2: 5 dataset-measure combinations (separate job for speed)"
echo "  - Simplification group: SimpDA, SimpEval (4 dataset-measure combinations)"
echo "  - CoGym group: 6 CoGym variants (9 dataset-measure combinations)"
echo "  - Remaining group: 5 datasets (8 dataset-measure combinations)"
echo "  - TOTAL: 15 datasets, 31 dataset-measure combinations"
echo "  - TOTAL EXPERIMENTS PER MODEL: 248 (31 combinations × 8 generators)"
echo "  - TOTAL ACROSS BOTH MODELS: 496 experiments"
echo ""

echo "Models tested:"
echo "  - GPT-4o-mini (OpenAI API)"
echo "  - Qwen3-32B (local server)"
echo ""

echo "Generator types tested:"
echo "  - Multi-metric generators (10 metrics per trial):"
echo "    • Basic LLM Judge"
echo "    • Rubric Generator (Prometheus)"
echo "    • Rubric Generator (DSPy)"
echo "    • G-Eval"
echo "    • Code Generation"
echo "  - Single-metric generators (1 metric per trial):"
echo "    • LLM Judge (MIPROv2-Optimized)"
echo "    • Fine-tuned ModernBERT"
echo "    • LLM Judge (Example-Based)"
echo ""

# Function to submit a job and capture job ID
submit_job() {
    local script_name=$1
    local job_name=$2
    
    echo "Submitting $job_name..."
    job_id=$(sbatch --parsable $script_name)
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Job submitted: $job_id"
        echo "$job_id" >> .temp_job_ids
    else
        echo "  ✗ Failed to submit $job_name"
        return 1
    fi
}

# Clear any existing temp job IDs file
rm -f .temp_job_ids

echo "Submitting jobs..."
echo "=================="

echo ""
echo "GPT-4o-mini jobs (using OpenAI API):"
echo "------------------------------------"
# Submit GPT-4o-mini jobs
submit_job "scripts/metric_generation/run_metric_gen_gpt4o_helpsteer.sh" "GPT-4o HelpSteer"
submit_job "scripts/metric_generation/run_metric_gen_gpt4o_helpsteer2.sh" "GPT-4o HelpSteer2"
submit_job "scripts/metric_generation/run_metric_gen_gpt4o_simplification.sh" "GPT-4o Simplification"
submit_job "scripts/metric_generation/run_metric_gen_gpt4o_cogym.sh" "GPT-4o CoGym"
submit_job "scripts/metric_generation/run_metric_gen_gpt4o_remaining.sh" "GPT-4o Remaining"

echo ""
echo "Qwen3-32B jobs (using local server):"
echo "------------------------------------"
# Submit Qwen3-32B jobs
submit_job "scripts/metric_generation/run_metric_gen_qwen_helpsteer.sh" "Qwen HelpSteer"
submit_job "scripts/metric_generation/run_metric_gen_qwen_helpsteer2.sh" "Qwen HelpSteer2"
submit_job "scripts/metric_generation/run_metric_gen_qwen_simplification.sh" "Qwen Simplification"
submit_job "scripts/metric_generation/run_metric_gen_qwen_cogym.sh" "Qwen CoGym"
submit_job "scripts/metric_generation/run_metric_gen_qwen_remaining.sh" "Qwen Remaining"

echo ""
echo "Job submission summary:"
echo "======================"

if [ -f .temp_job_ids ]; then
    job_count=$(wc -l < .temp_job_ids)
    echo "Successfully submitted $job_count jobs:"
    
    job_ids=$(cat .temp_job_ids | tr '\n' ',' | sed 's/,$//')
    echo "Job IDs: $job_ids"
    
    echo ""
    echo "Monitor job status with:"
    echo "  squeue -j $job_ids"
    echo ""
    echo "Monitor logs with:"
    echo "  tail -f scripts/metric_generation/logs/*gpt4o*.out"
    echo "  tail -f scripts/metric_generation/logs/*qwen*.out"
    echo ""
    
    echo "Expected execution timeline:"
    echo "  GPT-4o-mini jobs:"
    echo "    - HelpSteer: ~48h (5 combinations × 8 generators × 5 seeds)"
    echo "    - HelpSteer2: ~48h (5 combinations × 8 generators × 5 seeds)"
    echo "    - Simplification: ~36h (4 combinations × 8 generators × 5 seeds)"
    echo "    - CoGym: ~48h (9 combinations × 8 generators × 5 seeds)"
    echo "    - Remaining: ~48h (8 combinations × 8 generators × 5 seeds)"
    echo "  Qwen3-32B jobs:"
    echo "    - HelpSteer: ~48h (5 combinations × 8 generators × 5 seeds)"
    echo "    - HelpSteer2: ~48h (5 combinations × 8 generators × 5 seeds)"
    echo "    - Simplification: ~36h (4 combinations × 8 generators × 5 seeds)"
    echo "    - CoGym: ~48h (9 combinations × 8 generators × 5 seeds)"
    echo "    - Remaining: ~48h (8 combinations × 8 generators × 5 seeds)"
    echo "  - TOTAL: ~432h compute-hours across parallel jobs for both models"
    echo ""
    
    echo "Results will be saved to:"
    echo "  Main results: results/ablations/metric_generation/"
    echo "  Sub-results: results/ablations/metric_generation/sub_results/"
    echo "  Generated metrics: results/ablations/metric_generation/generated_metrics/"
    echo ""
    echo "Cache directories used (each job has unique cache):"
    echo "  GPT-4o-mini: /nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_gpt4o_[dataset_group]"
    echo "  Qwen3-32B: /nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_qwen_[dataset_group]"
    echo ""
    echo "Qwen server ports used:"
    echo "  HelpSteer: 7410, HelpSteer2: 7420, Simplification: 7430, CoGym: 7440, Remaining: 7450"
    echo ""
    
    # Submit a final merge job that depends on all the others
    echo "Submitting merge job to run after all benchmarks complete..."
    merge_job_id=$(sbatch --parsable --dependency=afterok:$job_ids scripts/metric_generation/merge_metric_gen_results.py)
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Merge job submitted: $merge_job_id"
        echo ""
        echo "All jobs submitted successfully!"
        echo "The merge job ($merge_job_id) will automatically run when all benchmark jobs complete."
    else
        echo "  ✗ Failed to submit merge job"
        echo "You'll need to run the merge script manually after all jobs complete."
    fi
    
    # Clean up temp file
    rm -f .temp_job_ids
    
else
    echo "No jobs were submitted successfully."
    exit 1
fi

echo ""
echo "============================================"
echo "Metric Generation Benchmark Setup Complete"
echo "============================================" 