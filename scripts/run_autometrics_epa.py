"""
CLI script that mirrors the “Run on EPA Data” section of notebooks/run_autometrics.ipynb.

The workflow:
  1. Load EPA comments, derive the binary `c` label, and filter down to long-form comments.
  2. Downsample so the label distribution is balanced and take a random sample for evaluation.
  3. Build an Autometrics Dataset and run the autometrics pipeline using ElasticNet regression.
  4. Print summaries, compute correlations, and optionally emit a metric distribution boxplot.

Example:
    python scripts/run_autometrics_epa.py \\
        --data-path notebooks/full_matched_comment_df__epa.csv \\
        --sample-size 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import dspy
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from autometrics.aggregator.regression.ElasticNet import ElasticNet
from autometrics.autometrics import Autometrics
from autometrics.dataset.Dataset import Dataset


EPA_DEFAULT_METRIC_COLUMNS = [
    "Clarity_and_Professionalism_of_Submission_gpt-5-mini",
    "Actionable_Specificity_of_Requests_gpt-5-mini",
    "Evidence_Quality_and_Citation_Rigor_gpt-5-mini",
    "Policy_Impact_and_Urgency_gpt-5-mini",
    "Implementation_Feasibility_and_Cost_Detail_gpt-5-mini",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Autometrics EPA tutorial pipeline outside of the notebook."
    )
    parser.add_argument(
        "--data-path",
        default=str(Path("notebooks") / "full_matched_comment_df__epa.csv"),
        help="Path to the EPA CSV (default: notebooks/full_matched_comment_df__epa.csv).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=500,
        help="Minimum number of words a comment must have to be included.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=5000,
        help="Maximum number of words a comment may have to be included.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of rows to sample for the Autometrics Dataset.",
    )
    parser.add_argument(
        "--downsample-random-state",
        type=int,
        default=42,
        help="Random state for downsampling balanced labels.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Random state for the final dataset sample.",
    )
    parser.add_argument(
        "--metrics-per-trial",
        type=int,
        default=10,
        help="Number of LLM judge metrics to generate.",
    )
    parser.add_argument(
        "--num-to-retrieve",
        type=int,
        default=5,
        help="Number of metrics to retrieve from the bank.",
    )
    parser.add_argument(
        "--num-to-regress",
        type=int,
        default=5,
        help="Number of metrics to keep after regression.",
    )
    parser.add_argument(
        "--generator-model",
        default="openai/gpt-5",
        help="Model ID used for metric generation prompts.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-5-mini",
        help="Model ID used for metric execution / judging.",
    )
    parser.add_argument(
        "--generated-metrics-dir",
        default="tutorial_metrics_epa",
        help="Directory where generated metrics should be stored.",
    )
    parser.add_argument(
        "--boxplot-path",
        type=Path,
        default=Path("artifacts") / "epa_metric_boxplot.png",
        help="Optional path to save the seaborn boxplot (directories created automatically).",
    )
    return parser.parse_args()


def downsample_df(
    df: pd.DataFrame, label_col: str = "c", random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Downsample so that each label bucket contains the same number of rows.
    Mirrors the helper defined in notebooks/run_autometrics.ipynb.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")
    label_counts = df[label_col].value_counts(dropna=False)
    if label_counts.empty:
        return df.copy()
    min_count = int(label_counts.min())
    return (
        df.groupby(label_col, group_keys=False)
        .apply(
            lambda group: group.sample(n=min_count, random_state=random_state)
            if len(group) >= min_count
            else group
        )
        .reset_index(drop=True)
    )


def load_epa_dataframe(
    data_path: Path,
    *,
    min_words: int,
    max_words: int,
    downsample_seed: Optional[int],
) -> pd.DataFrame:
    """Recreate the EPA filtering logic from the notebook."""
    df = pd.read_csv(data_path, index_col=0)
    df = df.assign(
        c=lambda frame: frame["comment_in_response"].notnull().astype(int),
    )
    filtered = (
        df.loc[lambda frame: frame["call"].notnull()]
        .loc[lambda frame: frame["actual_comment"].notnull()]
        .loc[
            lambda frame: frame["actual_comment"].str.split().str.len().between(
                min_words + 1, max_words - 1
            )
        ]
    )
    balanced = downsample_df(filtered, random_state=downsample_seed)
    return balanced


def create_dataset(
    dataframe: pd.DataFrame, *, sample_size: int, sample_seed: Optional[int]
) -> Dataset:
    if sample_size > len(dataframe):
        raise ValueError(
            f"Requested sample_size={sample_size} but only {len(dataframe)} rows available."
        )
    sampled = dataframe.sample(sample_size, random_state=sample_seed).reset_index(
        drop=True
    )
    return Dataset(
        dataframe=sampled,
        target_columns=["c"],
        ignore_columns=["index", "comment_in_response", "responses_to_comments"],
        metric_columns=[],
        name="MyCustomDataset_EPA",
        data_id_column="index",
        input_column="call",
        output_column="actual_comment",
        task_description=(
            "Rank candidate policy feedback drafts for follow-up: given a citizen’s "
            "submission, determine which responses merit escalation to agency officials."
        ),
    )


def run_autometrics_pipeline(
    dataset: Dataset,
    *,
    generator_model: str,
    judge_model: str,
    metrics_per_trial: int,
    num_to_retrieve: int,
    num_to_regress: int,
    generated_metrics_dir: str,
) -> dict:
    generator_llm = dspy.LM(generator_model)
    judge_llm = dspy.LM(judge_model)

    print("Running autometrics pipeline...")
    print("This will:")
    print(f"1. Generate {metrics_per_trial} LLM judge metric(s)")
    print(f"2. Retrieve {num_to_retrieve} relevant metrics from the bank")
    print("3. Evaluate all metrics on your dataset")
    print(f"4. Select top {num_to_regress} using ElasticNet regression")
    print("5. Create a final aggregated metric")

    autometrics = Autometrics(
        metric_generation_configs={
            "llm_judge": {"metrics_per_trial": metrics_per_trial}
        },
        regression_strategy=ElasticNet,
        seed=42,
        generated_metrics_dir=generated_metrics_dir,
    )

    results = autometrics.run(
        dataset=dataset,
        target_measure="c",
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        num_to_retrieve=num_to_retrieve,
        num_to_regress=num_to_regress,
        regenerate_metrics=True,
    )
    print("Pipeline complete! 🎉")
    return results


def summarize_results(results: dict, dataset: Dataset) -> None:
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    all_generated = results.get("all_generated_metrics", [])
    retrieved = results.get("retrieved_metrics", [])
    top_metrics = results.get("top_metrics", [])

    print(f"\nGenerated metrics: {len(all_generated)}")
    for i, metric in enumerate(all_generated, 1):
        print(f"  {i}. {metric.__name__}")

    print(f"\nRetrieved metrics: {len(retrieved)}")
    for i, metric in enumerate(retrieved, 1):
        print(f"  {i}. {metric.__name__}")

    print(f"\nTop selected metrics: {len(top_metrics)}")
    for i, metric in enumerate(top_metrics, 1):
        print(f"  {i}. {metric.get_name()}")

    regression_metric = results["regression_metric"]
    print(f"\nFinal regression metric: {regression_metric.get_name()}")
    print(f"Description: {regression_metric.get_description()}")

    final_scores = regression_metric.predict(dataset)
    human_scores = dataset.get_dataframe()["c"]

    print(f"\nPredicted vs Human scores for first 5 examples:")
    print("Example | Predicted | Human | Pred Rank | Human Rank")
    print("-" * 55)

    first_5_pred = final_scores[:5]
    first_5_human = human_scores.iloc[:5]
    for i in range(min(5, len(final_scores))):
        predicted = first_5_pred[i]
        human = first_5_human.iloc[i]
        pred_rank = int((first_5_pred > predicted).sum() + 1)
        human_rank = int((first_5_human > human).sum() + 1)
        print(
            f"  {i+1}     | {predicted:.3f}    | {human:.3f} | "
            f"{pred_rank:>9} | {human_rank:>10}"
        )

    correlation, p_value = pearsonr(human_scores, final_scores)
    print(f"\nCorrelation with human scores: {correlation:.3f} (p={p_value:.3f})")

    print("\n" + "=" * 50)
    print("REPORT CARD")
    print("=" * 50)
    print(results.get("report_card", "No report card available."))

    report_path = results.get("report_card_path")
    if report_path:
        print(f"\nHTML report written to: {report_path}")

    print("\n" + "=" * 50)
    print("TUTORIAL COMPLETE!")
    print("=" * 50)
    print("You now have:")
    print("✅ A custom metric for your task")
    print("✅ Top metrics selected via regression")
    print("✅ A final aggregated metric")
    print("✅ Correlation with human judgments")


def maybe_plot_metric_boxplot(
    dataset: Dataset,
    metric_columns: Iterable[str],
    *,
    output_path: Optional[Path],
) -> None:
    if output_path is None:
        return
    df = dataset.get_dataframe()
    missing = [col for col in metric_columns if col not in df.columns]
    if missing:
        print(
            "Skipping boxplot because the following metric columns are missing:\n  "
            + ", ".join(missing)
        )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax = sns.boxplot(data=df[metric_columns])
    ax.set_xticklabels(
        [col.replace("gpt-5-mini", "").replace("_", " ").strip() for col in metric_columns],
        rotation=10,
        horizontalalignment="right",
    )
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path)
    fig.clf()
    print(f"Saved metric distribution boxplot to {output_path}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set before running this script.")

    print(f"Loading EPA data from {data_path} ...")
    epa_df = load_epa_dataframe(
        data_path,
        min_words=args.min_words,
        max_words=args.max_words,
        downsample_seed=args.downsample_random_state,
    )
    print(f"Filtered EPA dataframe has {len(epa_df)} rows after balancing.")

    dataset = create_dataset(
        epa_df,
        sample_size=args.sample_size,
        sample_seed=args.sample_random_state,
    )

    results = run_autometrics_pipeline(
        dataset,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        metrics_per_trial=args.metrics_per_trial,
        num_to_retrieve=args.num_to_retrieve,
        num_to_regress=args.num_to_regress,
        generated_metrics_dir=args.generated_metrics_dir,
    )

    maybe_plot_metric_boxplot(
        dataset,
        EPA_DEFAULT_METRIC_COLUMNS,
        output_path=args.boxplot_path,
    )
    summarize_results(results, dataset)


if __name__ == "__main__":
    main()
