#!/bin/bash

#SBATCH --cpus-per-task=8           # Drop CPU count
#SBATCH --mem=64GB                  # Reduce memory
#SBATCH --gres=gpu:1                # Use explicit GRES
#SBATCH --partition=preempt
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --account=marlowe-m000076
#SBATCH --job-name=realhumaneval
#SBATCH --output=logs/realhumaneval.out
#SBATCH --error=logs/realhumaneval.err

. /cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh ; conda activate sglang

cd /projects/m000076/mryan0/autometrics

model="meta-llama/Llama-3.3-70B-Instruct"
port=7410
model_nickname="llama70b"

export DSPY_CACHEDIR=/scratch/m000076/mryan0/autometrics/dspy_cache

# TODO: Uncomment the following line to launch the server (ADD BACK IN 4 GPUs INSTEAD OF 1)
# python -m sglang.launch_server --model-path $model --port $port --host 0.0.0.0 --tp 4 --dtype bfloat16 --mem-fraction-static 0.9 --trust-remote-code --allow-auto-truncate &

conda activate autometrics

# Takes up to 7 minutes to load
# sleep 420

python realhumaneval.py > realhumaneval.txt