import pickle
import os
import dspy
from optimize.dspy_optimize import AirlineAgent
import pandas as pd
import argparse
from StaticRegression_TauBench_reward_seed42 import Autometrics_Regression_reward_StaticRegression
from StaticRegression_TauBenchBigger_reward_seed42_metric2 import Autometrics_Regression_reward_StaticRegression_Metric2
from StaticRegression_TauBenchBigger_reward_seed42_metric2_human import Autometrics_Regression_reward_StaticRegression_Metric2_Human
from StaticRegression_TauBenchBigger_reward_seed42_metric2 import Membership_Benefit_Application_Rubric_LLMJudge
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

######################## METRICS ########################

def basic_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    return pred.reward

autometric = Autometrics_Regression_reward_StaticRegression()
autometric2 = Autometrics_Regression_reward_StaticRegression_Metric2()
autometric2_human = Autometrics_Regression_reward_StaticRegression_Metric2_Human()
autometric2_best = Membership_Benefit_Application_Rubric_LLMJudge()

def autometrics_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    metric_input = pred.messages[1]['content']
    metric_output = pred.messages[1:]

    result = autometric.calculate_with_feedback(metric_input, metric_output)
    return ScoreWithFeedback(score=result.score, feedback=result.feedback)

def autometrics_metric2(example, pred, trace=None, pred_name=None, pred_trace=None):
    metric_input = pred.messages[1]['content']
    metric_output = pred.messages[1:]

    result = autometric2.calculate_with_feedback(metric_input, metric_output)
    return ScoreWithFeedback(score=result.score, feedback=result.feedback)

def autometrics_metric2_human(example, pred, trace=None, pred_name=None, pred_trace=None):
    metric_input = pred.messages[1]['content']
    metric_output = pred.messages[1:]

    result = autometric2_human.calculate_with_feedback(metric_input, metric_output)
    return ScoreWithFeedback(score=result.score, feedback=result.feedback)

def autometrics_metric2_best(example, pred, trace=None, pred_name=None, pred_trace=None):
    metric_input = pred.messages[1]['content']
    metric_output = pred.messages[1:]

    result = autometric2_best.calculate_with_feedback(metric_input, metric_output)
    return ScoreWithFeedback(score=result.score, feedback=result.feedback)

##########################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="gepa_logs")
    parser.add_argument("--mode", choices=["standard", "autometrics", "synthetic_autometrics", "autometrics_metric2", "synthetic_autometrics_metric2", "autometrics_metric2_human", "synthetic_autometrics_metric2_human", "autometrics_metric2_best", "synthetic_autometrics_metric2_best"], default="standard")
    parser.add_argument("--model", type=str, default="litellm_proxy/Qwen/Qwen3-32B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-url", type=str, default="http://sphinx3.stanford.edu:8544/v1")
    parser.add_argument("--filter", type=int, nargs="+", default=range(25))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ["BASE_URL"] = args.base_url
    log_dir = os.path.join(args.log_dir, args.mode, args.model, str(args.temperature), str(args.seed))

    # Setup LLM
    llm = dspy.LM(args.model, base_url=os.environ["BASE_URL"], api_key=None, temperature=args.temperature)
    dspy.configure(lm=llm)

    # Get Dataset
    print('args.filter', args.filter)
    _filter = args.filter if args.filter != [-1] else None

    # load dspy files
    folder = "dspy_examples_synthesized" if args.mode == "synthetic_autometrics" or args.mode == "synthetic_autometrics_metric2" or args.mode == "synthetic_autometrics_metric2_human" or args.mode == "synthetic_autometrics_metric2_best" else "dspy_examples"
    dataset = []
    for file in os.listdir(folder):
        
        if file.endswith(".pkl") and "dspy_example" in file:
            task_index = int(file.split("dspy_example")[-1].split("_")[0])
            if _filter is None or task_index in _filter:
                dataset.append(pickle.load(open(f"{folder}/{file}", "rb")))

    dataset = [example.with_inputs('env', 'task_index') for example in dataset]

    metric = None
    if args.mode == "standard":
        metric = basic_metric
    elif args.mode == "autometrics":
        metric = autometrics_metric
    elif args.mode == "synthetic_autometrics":
        metric = autometrics_metric
    elif args.mode == "autometrics_metric2":
        metric = autometrics_metric2
    elif args.mode == "synthetic_autometrics_metric2":
        metric = autometrics_metric2
    elif args.mode == "autometrics_metric2_human":
        metric = autometrics_metric2_human
    elif args.mode == "synthetic_autometrics_metric2_human":
        metric = autometrics_metric2_human
    elif args.mode == "autometrics_metric2_best":
        metric = autometrics_metric2_best
    elif args.mode == "synthetic_autometrics_metric2_best":
        metric = autometrics_metric2_best

    # Initialize Program
    program = AirlineAgent(dataset[0].env.wiki, max_iters=30)

    # Initialize Optimizer
    optimizer = dspy.GEPA(metric=metric, track_stats=True, auto="heavy", reflection_lm=llm, seed=args.seed, use_wandb=True, log_dir=log_dir, num_threads=64)

    # Run Optimizer
    optimized_program = optimizer.compile(program, trainset=dataset)

    if not os.path.exists(os.path.join(args.log_dir, "optimized_programs")):
        os.makedirs(os.path.join(args.log_dir, "optimized_programs"))

    model_name = args.model.split("/")[-1]

    optimized_program.save(f"{args.log_dir}/optimized_programs/{args.mode}_{model_name}_{args.temperature}_{args.seed}.json")


if __name__ == "__main__":
    main()

# Example Usage:
# python optimize/gepa_optimize.py --mode standard --base-url http://sphinx3.stanford.edu:8956/v1
# python optimize/gepa_optimize.py --mode autometrics --base-url http://sphinx3.stanford.edu:8956/v1
# python optimize/gepa_optimize.py --mode synthetic_autometrics --base-url http://sphinx3.stanford.edu:8956/v1