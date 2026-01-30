#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --open-mode=append
#SBATCH --partition=jag-standard
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --job-name=rf_benchmark_utilization
#SBATCH --output=logs/rf_benchmark_utilization.out
#SBATCH --error=logs/rf_benchmark_utilization.err
#SBATCH --exclude=jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh ; conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

# model="meta-llama/Llama-3.3-70B-Instruct"
# port=7410
# model_nickname="llama70b"

# export DSPY_CACHEDIR=/scratch/m000076/mryan0/autometrics/dspy_cache
# export CUDA_VISIBLE_DEVICES=1,2,3,4  # Use 4 GPUs

# # TODO: Uncomment the following line to launch the server (ADD BACK IN 4 GPUs INSTEAD OF 1)
# python -m sglang.launch_server --model-path $model --port $port --host 0.0.0.0 --tp 4 --dtype bfloat16 --mem-fraction-static 0.9 --trust-remote-code --allow-auto-truncate &

conda activate autometrics

# Takes up to 7 minutes to load
# sleep 420
# export CUDA_VISIBLE_DEVICES=0 # Use only one GPU for the evaluation

python autometrics/experiments/utilization/benchmark_utilization.py --skip-reference-based