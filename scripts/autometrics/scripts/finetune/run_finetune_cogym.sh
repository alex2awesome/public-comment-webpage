#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --job-name=finetune_cogym
#SBATCH --output=scripts/finetune/logs/finetune_cogym.out
#SBATCH --error=scripts/finetune/logs/finetune_cogym.err
#SBATCH --constraint=[141G|80G]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Script for CoGym datasets with fine-tuned ModernBERT metrics

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

echo "Starting Fine-tuned Metric analysis for CoGym datasets..."
echo "Processing datasets: CoGymTabularOutcome, CoGymTabularProcess, CoGymTravelOutcome, CoGymTravelProcess, CoGymLessonOutcome, CoGymLessonProcess"
echo "Seeds: 42 43 44 45 46"
echo "Correlations: kendall, pearson, spearman"
echo "Model: ModernBERT-Large with PEFT/LoRA"

# Create output directories
mkdir -p results/main_runs/baselines/finetune_sub_results
mkdir -p scripts/finetune/logs

# Set environment variables for fine-tuning
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_finetune_cogym"

echo "Environment setup:"
echo "  AUTOMETRICS_MODEL_DIR: $AUTOMETRICS_MODEL_DIR"
echo "  DSPY_CACHEDIR: $DSPY_CACHEDIR"

# Check available GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# Run the analysis for CoGym datasets
python analysis/main_experiments/run_finetune_correlation.py \
    --dataset CoGymTabularOutcome CoGymTabularProcess CoGymTravelOutcome CoGymTravelProcess CoGymLessonOutcome CoGymLessonProcess \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --model-save-dir "$AUTOMETRICS_MODEL_DIR" \
    --verbose

echo "CoGym datasets analysis completed!"
echo "Individual dataset files saved in: results/main_runs/baselines/finetune_sub_results/"

echo ""
echo "Summary of processed datasets:"
echo "  - CoGymTabularOutcome: 1 measure (outcomeRating)"
echo "  - CoGymTabularProcess: 2 measures (agentRating, communicationRating)"
echo "  - CoGymTravelOutcome: 1 measure (outcomeRating)"
echo "  - CoGymTravelProcess: 2 measures (agentRating, communicationRating)"
echo "  - CoGymLessonOutcome: 1 measure (outcomeRating)"
echo "  - CoGymLessonProcess: 2 measures (agentRating, communicationRating)"
echo "  - Total: 9 dataset-measure combinations"
echo "  - Total training runs: 45 (9 combinations × 5 seeds)"
echo ""
echo "Expected output files:"
for dataset in CoGymTabularOutcome CoGymTabularProcess CoGymTravelOutcome CoGymTravelProcess CoGymLessonOutcome CoGymLessonProcess; do
    for corr in kendall pearson spearman; do
        echo "  - results/main_runs/baselines/finetune_sub_results/finetune_${corr}_${dataset}.csv"
    done
done
echo ""
echo "Results will be merged with other datasets using the merge script."

# Generate summary statistics
echo ""
echo "Generating summary statistics..."
python -c "
import pandas as pd
import numpy as np
import os

print('=== Fine-tuned Metric CoGym Results Summary ===')

correlation_types = ['kendall', 'pearson', 'spearman']
datasets = ['CoGymTabularOutcome', 'CoGymTabularProcess', 'CoGymTravelOutcome', 'CoGymTravelProcess', 'CoGymLessonOutcome', 'CoGymLessonProcess']

for corr_type in correlation_types:
    print(f'\n--- {corr_type.upper()} Correlation ---')
    
    total_metrics = 0
    all_correlations = []
    all_successful_runs = 0
    
    for dataset in datasets:
        filepath = f'results/main_runs/baselines/finetune_sub_results/finetune_{corr_type}_{dataset}.csv'
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            total_metrics += len(df)
            
            if 'mean_correlation' in df.columns:
                valid_corrs = df['mean_correlation'].dropna()
                all_correlations.extend(valid_corrs.abs().tolist())
            
            if 'num_successful_runs' in df.columns:
                all_successful_runs += df['num_successful_runs'].sum()
            
            print(f'  {dataset}: {len(df)} metrics processed')
        else:
            print(f'  {dataset}: No results file found')
    
    if all_correlations:
        print(f'  Total metrics: {total_metrics}')
        print(f'  Mean |correlation|: {np.mean(all_correlations):.4f}')
        print(f'  Median |correlation|: {np.median(all_correlations):.4f}')
        print(f'  Successful runs: {all_successful_runs}/{total_metrics * 5}')
"

echo ""
echo "Fine-tuned metric CoGym analysis completed!" 