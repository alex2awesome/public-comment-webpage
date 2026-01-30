#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --open-mode=append
#SBATCH --partition=john-lo
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --job-name=all_recommendation
#SBATCH --output=logs/all_recommendation.out
#SBATCH --error=logs/all_recommendation.err
#SBATCH --requeue

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

API_KEY=$OPENAI_API_KEY
export DSPY_CACHEDIR=/nlp/scr3/nlp/20questions/dspy_cache/autometrics

python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset SimpDA --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymLessonOutcome --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymLessonProcess --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTabularOutcome --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTabularProcess --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTravelOutcome --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTravelProcess --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset Design2Code --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset EvalGenMedical --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset EvalGenProduct --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset HelpSteer --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset HelpSteer2 --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset Primock57 --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset RealHumanEval --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset SimpEval --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset SummEval --top-k 20 --llm openai/gpt-4o-mini --llm-api-key $API_KEY