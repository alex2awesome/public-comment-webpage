#!/usr/bin/env python3
"""
Super Simple Autometrics Tutorial
=================================

This tutorial shows the absolute basics of using autometrics.
Just load a dataset, run the pipeline, and get your metrics!
"""

## Cell 1: Setup and Imports
import os
import dspy
from autometrics.autometrics import Autometrics
from autometrics.dataset.datasets.simplification.simplification import SimpDA
from autometrics.aggregator.regression.ElasticNet import ElasticNet

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

## Cell 2: Load Dataset
# Load the SimpDA dataset (text simplification)
dataset = SimpDA()
target_measure = "simplicity"  # The human score column we want to predict

print(f"Dataset: {dataset.get_name()}")
print(f"Size: {len(dataset.get_dataframe())} examples")
print(f"Target measure: {target_measure}")

## Cell 3: Configure LLMs
# Use GPT-4o-mini for both generation and judging
generator_llm = dspy.LM("openai/gpt-4o-mini")
judge_llm = dspy.LM("openai/gpt-4o-mini")

print("LLMs configured!")

## Cell 4: Create Simple Autometrics Pipeline
# Super simple configuration:
# - Generate 1 metric using LLM judge
# - Retrieve 10 metrics from the bank
# - Select top 5 using ElasticNet regression
autometrics = Autometrics(
    metric_generation_configs={
        "llm_judge": {"metrics_per_trial": 1}  # Just generate 1 metric
    },
    regression_strategy=ElasticNet,  # Use ElasticNet instead of default Lasso
    seed=42,  # For reproducibility
    generated_metrics_dir="tutorial_metrics"  # Unique directory for this tutorial
)

print("Autometrics pipeline created!")

## Cell 5: Run the Pipeline
print("Running autometrics pipeline...")
print("This will:")
print("1. Generate 1 LLM judge metric")
print("2. Retrieve 10 relevant metrics from the bank")
print("3. Evaluate all metrics on your dataset")
print("4. Select top 5 using ElasticNet regression")
print("5. Create a final aggregated metric")

results = autometrics.run(
    dataset=dataset,
    target_measure=target_measure,
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_to_retrieve=10,  # Retrieve 10 metrics
    num_to_regress=5     # Select top 5
)

print("Pipeline complete! 🎉")

## Cell 6: View Results
print("\n" + "="*50)
print("RESULTS")
print("="*50)

print(f"\nGenerated metrics: {len(results['all_generated_metrics'])}")
for i, metric in enumerate(results['all_generated_metrics']):
    print(f"  {i+1}. {metric.__name__}")

print(f"\nRetrieved metrics: {len(results['retrieved_metrics'])}")
for i, metric in enumerate(results['retrieved_metrics'][:3]):  # Show first 3
    print(f"  {i+1}. {metric.__name__}")

print(f"\nTop selected metrics: {len(results['top_metrics'])}")
for i, metric in enumerate(results['top_metrics']):
    print(f"  {i+1}. {metric.get_name()}")

print(f"\nFinal regression metric: {results['regression_metric'].get_name()}")
print(f"Description: {results['regression_metric'].get_description()}")

## Cell 7: Use Your Metrics
print("\n" + "="*50)
print("USING YOUR METRICS")
print("="*50)

# Get predictions from your final metric
final_scores = results['regression_metric'].predict(dataset)
human_scores = dataset.get_dataframe()[target_measure]

print(f"\nPredicted vs Human scores for first 5 examples:")
print("Example | Predicted | Human | Pred Rank | Human Rank")
print("-" * 55)

# Get first 5 examples
first_5_pred = final_scores[:5]
first_5_human = human_scores.iloc[:5]

for i in range(min(5, len(final_scores))):
    predicted = first_5_pred[i]
    human = first_5_human.iloc[i]
    
    # Calculate ranks within these 5 examples (higher score = higher rank)
    pred_rank = (first_5_pred > predicted).sum() + 1
    human_rank = (first_5_human > human).sum() + 1
    
    print(f"  {i+1}     | {predicted:.3f}    | {human:.3f} | {pred_rank:>9} | {human_rank:>10}")

# Check correlation with human scores
import numpy as np
from scipy.stats import pearsonr

correlation, p_value = pearsonr(human_scores, final_scores)
print(f"\nCorrelation with human scores: {correlation:.3f} (p={p_value:.3f})")

## Cell 8: View Report Card
print("\n" + "="*50)
print("REPORT CARD")
print("="*50)

print(results['report_card'])

print("\n" + "="*50)
print("TUTORIAL COMPLETE!")
print("="*50)
print("You now have:")
print("✅ A custom metric for your task")
print("✅ Top 5 most relevant metrics")
print("✅ A final aggregated metric")
print("✅ Correlation with human judgments")
print("\nYou can use these metrics on new data!")
