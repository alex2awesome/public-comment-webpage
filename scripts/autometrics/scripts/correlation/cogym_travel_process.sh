#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=200GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=16:00:00 
#SBATCH --nodes=1
#SBATCH --job-name=cogym_travel_process
#SBATCH --output=logs/cogym_travel_process.out
#SBATCH --error=logs/cogym_travel_process.err
#SBATCH --constraint=141G
#SBATCH --requeue

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics  

python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelProcess --correlation all --top-k 5 --cache-dir ./.cache/cogym_travel_process