#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=200GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --job-name=all_correlation
#SBATCH --output=logs/all_correlation.out
#SBATCH --error=logs/all_correlation.err
#SBATCH --constraint=141G
#SBATCH --requeue

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

python autometrics/experiments/correlation/benchmark_correlation.py --dataset SimpDA --correlation all --top-k 5 --cache-dir ./.cache/simpda
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymLessonOutcome --correlation all --top-k 5 --cache-dir ./.cache/cogym_lesson_outcome
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymLessonProcess --correlation all --top-k 5 --cache-dir ./.cache/cogym_lesson_process
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTabularOutcome --correlation all --top-k 5 --cache-dir ./.cache/cogym_tabular_outcome
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTabularProcess --correlation all --top-k 5 --cache-dir ./.cache/cogym_tabular_process
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelOutcome --correlation all --top-k 5 --cache-dir ./.cache/cogym_travel_outcome
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelProcess --correlation all --top-k 5 --cache-dir ./.cache/cogym_travel_process
python autometrics/experiments/correlation/benchmark_correlation.py --dataset Design2Code --correlation all --top-k 5 --cache-dir ./.cache/design2code
python autometrics/experiments/correlation/benchmark_correlation.py --dataset EvalGenMedical --correlation all --top-k 5 --cache-dir ./.cache/evalgen_medical
python autometrics/experiments/correlation/benchmark_correlation.py --dataset EvalGenProduct --correlation all --top-k 5 --cache-dir ./.cache/evalgen_product
python autometrics/experiments/correlation/benchmark_correlation.py --dataset HelpSteer --correlation all --top-k 5 --cache-dir ./.cache/helpsteer
python autometrics/experiments/correlation/benchmark_correlation.py --dataset HelpSteer2 --correlation all --top-k 5 --cache-dir ./.cache/helpsteer2
python autometrics/experiments/correlation/benchmark_correlation.py --dataset Primock57 --correlation all --top-k 5 --cache-dir ./.cache/primock57
python autometrics/experiments/correlation/benchmark_correlation.py --dataset RealHumanEval --correlation all --top-k 5 --cache-dir ./.cache/real_human_eval
python autometrics/experiments/correlation/benchmark_correlation.py --dataset SimpEval --correlation all --top-k 5 --cache-dir ./.cache/simeval
python autometrics/experiments/correlation/benchmark_correlation.py --dataset SummEval --correlation all --top-k 5 --cache-dir ./.cache/summeval