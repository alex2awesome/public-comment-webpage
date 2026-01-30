#!/bin/bash
#SBATCH --job-name=test_metric_gen
#SBATCH --output=scripts/metric_generation/logs/test_metric_gen.out
#SBATCH --error=scripts/metric_generation/logs/test_metric_gen.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --constraint=[141G|80G]
#SBATCH --partition=sphinx

# Environment setup
echo "============================================"
echo "Metric Generation Benchmark - TEST RUN"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================"

# Set up environment
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# Set up conda environment
source /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

# Set up model directory  
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"

# Set up DSPy cache directory
export DSPY_CACHE_DIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_test"
mkdir -p "$DSPY_CACHE_DIR"

echo ""
echo "Environment setup:"
echo "  Python: $(which python)"
echo "  Conda environment: $CONDA_DEFAULT_ENV"
echo "  Model directory: $AUTOMETRICS_MODEL_DIR"
echo "  DSPy cache directory: $DSPY_CACHE_DIR"
echo "  CUDA devices: $CUDA_VISIBLE_DEVICES"
echo ""

echo "Starting QUICK TEST of metric generation benchmark..."
echo "Testing on limited scope:"
echo "  - Datasets: HelpSteer only"
echo "  - Measures: helpfulness only"
echo "  - Seeds: 42, 43 only (instead of 5 seeds)"
echo "  - Generators: llm_judge, codegen only (instead of all 8)"
echo ""

# Test with very limited scope for validation
python analysis/ablations/run_metric_generation_benchmark.py \
    --dataset HelpSteer \
    --measure helpfulness \
    --seeds 42 43 \
    --generator llm_judge codegen \
    --correlation kendall \
    --generator-model gpt4o_mini \
    --judge-model gpt4o_mini \
    --output-dir results/ablations/metric_generation_test \
    --model-save-dir "$AUTOMETRICS_MODEL_DIR" \
    --verbose

echo ""
echo "Test run configuration:"
echo "  - Total combinations: 1 dataset × 1 measure × 2 generators = 2 experiments"
echo "  - Total seeds per experiment: 2"
echo "  - Expected total runs: 4"
echo ""

# Check if results were generated
if [ -f "results/ablations/metric_generation_test/metric_generation_benchmark_gpt4o_mini_gpt4o_mini_kendall.csv" ]; then
    echo "✓ Test results file created successfully"
    
    # Show basic statistics
    python -c "
import pandas as pd
import os

results_file = 'results/ablations/metric_generation_test/metric_generation_benchmark_gpt4o_mini_gpt4o_mini_kendall.csv'
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    print(f'Results summary:')
    print(f'  Records: {len(df)}')
    print(f'  Generators: {df[\"generator_type\"].unique().tolist()}')
    print(f'  Datasets: {df[\"dataset\"].unique().tolist()}')
    print(f'  Measures: {df[\"measure\"].unique().tolist()}')
    print()
    
    # Check for any successful runs
    successful = df['num_successful_runs'].sum()
    total_possible = len(df) * 2  # 2 seeds
    print(f'Success rate: {successful}/{total_possible} runs ({successful/total_possible*100:.1f}%)')
    
    if successful > 0:
        print('✓ Some metric generation runs succeeded')
        
        # Show sample results
        valid_results = df[df['num_successful_runs'] > 0]
        if not valid_results.empty:
            print()
            print('Sample results:')
            for _, row in valid_results.iterrows():
                print(f'  {row[\"generator_description\"]} on {row[\"dataset\"]}.{row[\"measure\"]}: {row[\"mean_±_ci\"]}')
    else:
        print('✗ No successful runs - check configuration')
else:
    echo "✗ Test results file not created - check logs for errors"
fi

echo ""
echo "Generated metrics saved to:"
echo "  results/ablations/metric_generation_test/generated_metrics/"

# List generated metrics if any
if [ -d "results/ablations/metric_generation_test/generated_metrics" ]; then
    metric_count=$(find results/ablations/metric_generation_test/generated_metrics -name "*.py" | wc -l)
    echo "  Total generated metric files: $metric_count"
    
    if [ $metric_count -gt 0 ]; then
        echo "  Sample generated metrics:"
        find results/ablations/metric_generation_test/generated_metrics -name "*.py" | head -5 | while read file; do
            echo "    $file"
        done
    fi
else
    echo "  No generated metrics directory found"
fi

echo ""
echo "Test run completed at: $(date)"
echo ""
echo "If this test succeeds, you can run the full benchmark with:"
echo "  ./scripts/metric_generation/run_all_metric_gen_parallel.sh" 