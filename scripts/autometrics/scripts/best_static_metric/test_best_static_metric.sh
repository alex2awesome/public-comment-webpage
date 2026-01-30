#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --job-name=best_static_metric_test
#SBATCH --output=scripts/best_static_metric/logs/test_best_static_metric.out
#SBATCH --error=scripts/best_static_metric/logs/test_best_static_metric.err
#SBATCH --constraint=141G
#SBATCH --requeue

# Load environment
. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

# Run a quick test with specific datasets and measures to verify the script works
# This is useful for debugging and ensuring everything is set up correctly
# Testing Primock57 with just inc_plus_omi measure using dataset-specific cache

echo "Running test with Primock57 dataset, inc_plus_omi measure for validation..."

python analysis/main_experiments/run_best_static_metric.py \
    --best-metrics-file results/best_metrics.csv \
    --output-file results/stability/best_metrics_stability_test.csv \
    --cache-dir ./.cache/best_static_metric \
    --seeds 42 43 44 \
    --correlation kendall \
    --dataset Primock57 \
    --measure inc_plus_omi \
    --verbose

echo "Test completed! Check results/stability/best_metrics_stability_test.csv"
echo "Cache will be in ./.cache/best_static_metric/primock57/seed_*/" 