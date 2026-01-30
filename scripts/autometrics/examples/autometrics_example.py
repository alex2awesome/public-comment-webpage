#!/usr/bin/env python3
"""
Power user example usage of the Autometrics pipeline.

This script demonstrates how to customize all aspects of the Autometrics pipeline
for advanced users who want full control over the configuration.

For a simple example with all defaults, see autometrics_simple_example.py

Usage:
    python autometrics_example.py
"""

import os
import sys
import dspy

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autometrics.autometrics import Autometrics
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer
from autometrics.recommend.PipelinedRec import PipelinedRec
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.LLMRec import LLMRec

def main():
    """Run the Autometrics pipeline with full customization."""
    
    # 1. Load a dataset
    print("Loading HelpSteer dataset...")
    dataset = HelpSteer()
    target_measure = "helpfulness"
    
    # 2. Configure LLMs
    print("Configuring LLMs...")
    
    # For OpenAI (requires OPENAI_API_KEY environment variable)
    generator_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    judge_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Alternative: For local Qwen server
    # generator_llm = dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base="http://localhost:7410/v1", api_key="None")
    # judge_llm = dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base="http://localhost:7410/v1", api_key="None")
    
    # 3. Customize metric generation configs
    custom_generator_configs = {
        "llm_judge": {
            "metrics_per_trial": 3,
            "description": "Custom LLM Judge"
        },
        "codegen": {
            "metrics_per_trial": 2,
            "description": "Custom Code Generation"
        },
        "rubric_prometheus": {
            "metrics_per_trial": 2,
            "description": "Custom Prometheus Rubric"
        }
    }
    
    # 4. Create Autometrics with custom configuration (overriding meaningful defaults)
    print("Creating Autometrics pipeline...")
    autometrics = Autometrics(
        # Override specific defaults while keeping others:
        # - metric_generation_configs: DEFAULT_GENERATOR_CONFIGS → custom_generator_configs
        metric_generation_configs=custom_generator_configs,
        
        # - retriever: PipelinedRec → PipelinedRec (same, but with custom kwargs)
        # - retriever_kwargs: DEFAULT_RETRIEVER_KWARGS → custom kwargs
        retriever_kwargs={
            "recommenders": [ColBERT, LLMRec],
            "top_ks": [30, 15],  # Custom top-k values (overrides [60, 30])
            "index_paths": [None, None],
            "force_reindex": False
        },
        
        # - regression_strategy: Lasso → Lasso (same as default)
        # - regression_kwargs: DEFAULT_REGRESSION_KWARGS → custom kwargs (if needed)
        # - metric_bank: all_metric_classes → all_metric_classes (same as default)
        # - seed: 42 → 123 (custom seed)
        seed=123,
        
        # - allowed_failed_metrics: 0 → 2 (allow some failures)
        allowed_failed_metrics=2,
        
        # Custom output directory (no default in signature)
        generated_metrics_dir="custom_example_metrics",
        merge_generated_with_bank=False
    )
    
    # 5. Run the pipeline with custom parameters
    print("Running Autometrics pipeline...")
    results = autometrics.run(
        dataset=dataset,
        target_measure=target_measure,
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        
        # Custom retrieval and regression parameters
        num_to_retrieve=15,  # Retrieve fewer metrics
        num_to_regress=3,    # Select fewer final metrics
        
        # Force regeneration
        regenerate_metrics=False,
        
        # Custom API base for Prometheus (if using rubric_prometheus)
        prometheus_api_base="http://your-prometheus-server:7420/v1",
        
        # Custom model save directory for fine-tuning
        model_save_dir="/custom/finetune/models"
    )
    
    # 6. Display results
    print("\n" + "="*60)
    print("AUTOMETRICS PIPELINE RESULTS (CUSTOM CONFIG)")
    print("="*60)
    
    print(f"\nDataset: {results['dataset'].get_name()}")
    print(f"Target Measure: {results['target_measure']}")
    
    print(f"\nGenerated Metrics: {len(results['all_generated_metrics'])}")
    for i, metric in enumerate(results['all_generated_metrics']):
        print(f"  {i+1}. {metric.__name__}")
    
    print(f"\nRetrieved Metrics: {len(results['retrieved_metrics'])}")
    for i, metric in enumerate(results['retrieved_metrics']):
        print(f"  {i+1}. {metric.__name__}")
    
    print(f"\nTop Selected Metrics: {len(results['top_metrics'])}")
    for i, metric in enumerate(results['top_metrics']):
        print(f"  {i+1}. {metric.get_name()}")
    
    if results['regression_metric']:
        print(f"\nRegression Metric:")
        print(f"  Name: {results['regression_metric'].get_name()}")
        print(f"  Description: {results['regression_metric'].get_description()}")
    
    print(f"\nImportance Scores:")
    for i, (score, metric_name) in enumerate(results['importance_scores'][:5]):
        print(f"  {i+1}. {metric_name}: {score:.4f}")
    
    print(f"\nReport Card:")
    print(results['report_card'])
    
    print("\nPipeline completed successfully!")
    print("This example shows full customization capabilities.")

if __name__ == "__main__":
    main() 