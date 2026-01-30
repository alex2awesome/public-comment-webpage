#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=240GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=16:00:00  
#SBATCH --nodes=1
#SBATCH --job-name=cogym_correlation
#SBATCH --output=logs/cogym_correlation.out
#SBATCH --error=logs/cogym_correlation.err
#SBATCH --constraint=141G
#SBATCH --requeue

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelOutcome --correlation all --top-k 5 --cache-dir ./.cache/cogym_travel_outcome
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelProcess --correlation all --top-k 5 --cache-dir ./.cache/cogym_travel_process
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymLessonOutcome --correlation all --top-k 5 --cache-dir ./.cache/cogym_lesson_outcome
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymLessonProcess --correlation all --top-k 5 --cache-dir ./.cache/cogym_lesson_process
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTabularOutcome --correlation all --top-k 5 --cache-dir ./.cache/cogym_tabular_outcome
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTabularProcess --correlation all --top-k 5 --cache-dir ./.cache/cogym_tabular_process
