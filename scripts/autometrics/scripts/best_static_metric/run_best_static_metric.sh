#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=best_static_metric_summeval
#SBATCH --output=logs/best_static_metric_summeval.out
#SBATCH --error=logs/best_static_metric_summeval.err
#SBATCH --constraint=141G
#SBATCH --requeue

# Load environment
. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

# Run the best static metric stability analysis
# This tests correlation stability across multiple seeds for best-performing metrics
# CRITICAL: Uses persistent test sets ONLY for reproducible paper results

CORRELATION_FUNCS="all"

python analysis/main_experiments/run_best_static_metric.py \
    --best-metrics-file results/best_metrics.csv \
    --output-file results/main_runs/baselines/best_static_metric.csv \
    --seeds 42 43 44 45 46 \
    --correlation $CORRELATION_FUNCS \
    --dataset SummEval \
    --verbose

if [ "$CORRELATION_FUNCS" == "all" ]; then
    echo "Analysis completed! Check results/main_runs/baselines/best_static_metric_kendall.csv, results/main_runs/baselines/best_static_metric_pearson.csv, results/main_runs/baselines/best_static_metric_spearman.csv" 
else
    echo "Analysis completed! Check results/main_runs/baselines/best_static_metric_${CORRELATION_FUNCS}.csv" 
fi