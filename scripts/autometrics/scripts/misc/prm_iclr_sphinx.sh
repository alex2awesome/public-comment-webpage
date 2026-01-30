#!/bin/bash

#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --job-name=prm_iclr
#SBATCH --output=logs/prm_iclr_%j.out
#SBATCH --error=logs/prm_iclr_%j.err

# Activate conda environment
. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

# Move to repo root
cd /nlp/scr2/nlp/personal-rm/autometrics

python analysis/misc/prm_iclr.py \
  --input autometrics/dataset/datasets/iclr/train.csv \
  --output analysis/misc/prm_iclr_results.csv \
  --full-text-col full_text \
  --abstract-col abstract \
  --min-section-chars 50 \
  --include_preface \
  --keep-subheadings


