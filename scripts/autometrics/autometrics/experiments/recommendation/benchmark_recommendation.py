#!/usr/bin/env python3
"""Benchmark Recommendation Script
=================================

Run metric recommendation experiments over a chosen dataset using a suite of
recommenders (BM25, ColBERT, Faiss, LLMRec, and the three pipeline variants).

Example usage
-------------
python benchmark_recommendation.py --dataset summeval --top-k 20 --output-dir outputs/recommendation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

from platformdirs import user_data_dir

# ---------------------------------------------------------------------------
# Light-weight imports first – avoid heavy model loads on --help
# ---------------------------------------------------------------------------

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from autometrics.experiments.correlation.benchmark_correlation import (
    get_dataset_suggestions,
    load_dataset,
)

from autometrics.metrics.MetricBank import (
    reference_free_metric_classes,
    all_metric_classes,
)
from autometrics.recommend.BM25 import BM25
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.Faiss import Faiss
from autometrics.recommend.LLMRec import LLMRec
from autometrics.recommend.PipelinedRec import PipelinedRec
from autometrics.experiments.recommendation import RecommendationExperiment

logger = logging.getLogger("benchmark_recommendation")

# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    suggestions = get_dataset_suggestions()
    parser = argparse.ArgumentParser(
        description="Benchmark metric recommendation systems on a dataset.\n"
        "Available datasets: " + ", ".join(suggestions)
    )

    parser.add_argument(
        "--dataset",
        default="summeval",
        help="Dataset identifier to evaluate (see list above or use --list-datasets).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs. Defaults to outputs/recommendation/<dataset>.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of metrics to recommend (final list length).",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated list of recommenders to run (bm25,colbert,faiss,llmrec,bm25+llmrec,colbert+llmrec,faiss+llmrec) or 'all'.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force rebuilding of retrieval indexes (BM25/ColBERT/Faiss).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (currently only used for dataset splitting if needed).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available dataset identifiers and exit.",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="Name of the LLM to use for LLM-based recommenders (e.g. 'openai/gpt-4o-mini').  If omitted, we rely on an existing global dspy configuration.",
    )
    parser.add_argument("--llm-api-base", dest="llm_api_base", default=None, help="Custom API base URL for self-hosted models (optional).")
    parser.add_argument("--llm-api-key", dest="llm_api_key", default=None, help="API key for the LLM provider (optional – can also come from env vars).")
    parser.add_argument("--llm-model-type", dest="llm_model_type", default=None, help="Model type for the LLM (e.g. 'chat') if required by dspy.")
    parser.add_argument(
        "--llmrec-only",
        action="store_true",
        help="Run only LLMRec (skip other recommenders) for focused debugging.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Helper to decide which metric classes to index
# ---------------------------------------------------------------------------

def determine_metric_classes(dataset) -> tuple[List[type], str]:
    """Return (metric_classes, variant_name)."""
    variant = "all_metrics"
    metric_classes: List[type]
    try:
        ref_cols = dataset.get_reference_columns()
        if ref_cols is None or len(ref_cols) == 0:
            variant = "reference_free"
            metric_classes = reference_free_metric_classes
        else:
            metric_classes = all_metric_classes
    except Exception:
        # Conservatively assume references present → include all metrics
        metric_classes = all_metric_classes
    return metric_classes, variant

# ---------------------------------------------------------------------------
# Recommender builders
# ---------------------------------------------------------------------------

def build_recommenders(metric_classes: List[type], variant: str, top_k: int, force_reindex: bool, selected: List[str], model=None) -> Dict[str, "MetricRecommender"]:
    """Instantiate recommender objects according to *selected* list."""
    recs: Dict[str, "MetricRecommender"] = {}

    def idx(prefix: str) -> str:
        """Return a stable directory under the user data dir.

        We want paths like:
            ~/.local/share/autometrics/<prefix>_<variant>

        The *platformdirs* API expects the first positional argument to be
        *appname* – this becomes the leaf directory on Unix – while the second
        argument (``appauthor``) introduces an intermediate directory that we
        do not want.  Therefore we only pass *appname* and join the remainder
        manually.
        """
        base = user_data_dir("autometrics")  # ~/.local/share/autometrics
        return os.path.join(base, f"{prefix}_{variant}")

    wanted = set(s.lower() for s in selected)

    if "all" in wanted:
        wanted = {
            "bm25",
            "colbert",
            "faiss",
            "llmrec",
            "bm25+llmrec",
            "colbert+llmrec",
            "faiss+llmrec",
        }

    # ------------------------------------------------------------------
    # Individual recommenders
    # ------------------------------------------------------------------
    if "bm25" in wanted:
        recs["BM25"] = BM25(
            metric_classes=metric_classes,
            index_path=idx("bm25"),
            force_reindex=force_reindex,
        )
    if "colbert" in wanted:
        recs["ColBERT"] = ColBERT(
            metric_classes=metric_classes,
            index_path=idx("colbert"),
            force_reindex=force_reindex,
        )
    if "faiss" in wanted:
        recs["Faiss"] = Faiss(
            metric_classes=metric_classes,
            index_path=idx("faiss"),
            force_reindex=force_reindex,
        )
    if "llmrec" in wanted:
        recs["LLMRec"] = LLMRec(
            metric_classes=metric_classes,
            index_path=idx("llm"),
            force_reindex=force_reindex,
            model=model,
        )

    # ------------------------------------------------------------------
    # Pipeline recommenders
    # ------------------------------------------------------------------
    if "bm25+llmrec" in wanted:
        recs["PipelinedRec(BM25→LLMRec)"] = PipelinedRec(
            metric_classes=metric_classes,
            recommenders=[BM25, LLMRec],
            top_ks=[30, top_k],
            index_paths=[idx("bm25"), None],
            force_reindex=force_reindex,
            model=model,
        )
    if "colbert+llmrec" in wanted:
        recs["PipelinedRec(ColBERT→LLMRec)"] = PipelinedRec(
            metric_classes=metric_classes,
            recommenders=[ColBERT, LLMRec],
            top_ks=[30, top_k],
            index_paths=[idx("colbert"), None],
            force_reindex=force_reindex,
            model=model,
        )
    if "faiss+llmrec" in wanted:
        recs["PipelinedRec(Faiss→LLMRec)"] = PipelinedRec(
            metric_classes=metric_classes,
            recommenders=[Faiss, LLMRec],
            top_ks=[30, top_k],
            index_paths=[idx("faiss"), None],
            force_reindex=force_reindex,
            model=model,
        )

    return recs

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.list_datasets:
        print("Available datasets:\n" + "\n".join(get_dataset_suggestions()))
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    metric_classes, variant = determine_metric_classes(dataset)
    logger.info(f"Using {len(metric_classes)} metric classes ({variant.replace('_', ' ')})")

    methods_list = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else ["all"]
    
    # Override methods if --llmrec-only is specified
    if args.llmrec_only:
        methods_list = ["llmrec"]
        logger.info("🎯 Running ONLY LLMRec for focused debugging")

    # ---------------------------------------------------------------
    # LLM configuration (if requested)
    # ---------------------------------------------------------------
    lm = None
    if args.llm:
        import dspy

        lm_kwargs = {}
        if args.llm_api_base:
            lm_kwargs["api_base"] = args.llm_api_base
        if args.llm_api_key:
            lm_kwargs["api_key"] = args.llm_api_key
        if args.llm_model_type:
            lm_kwargs["model_type"] = args.llm_model_type

        lm = dspy.LM(args.llm, **lm_kwargs)
        dspy.configure(lm=lm)  # Set as global default for convenience

    recommenders = build_recommenders(
        metric_classes=metric_classes,
        variant=variant,
        top_k=args.top_k,
        force_reindex=args.force_reindex,
        selected=methods_list,
        model=lm,
    )

    if not recommenders:
        logger.error("No recommenders selected – exiting.")
        return 1

    base_output_dir = args.output_dir or os.path.join("outputs", "recommendation", args.dataset)
    os.makedirs(base_output_dir, exist_ok=True)

    # If we have an LLM, append its sanitized name to output dir for clarity
    if lm is not None:
        def _sanitize(name: str) -> str:
            return name.replace("/", "_").replace(":", "_")

        base_output_dir = os.path.join(base_output_dir, f"llm_{_sanitize(args.llm)}")

    experiment = RecommendationExperiment(
        name=f"Recommendation Benchmark – {args.dataset}",
        description="Benchmarking metric recommendation systems",
        recommenders=recommenders,
        output_dir=base_output_dir,
        dataset=dataset,
        top_k=args.top_k,
        seed=args.seed,
    )

    experiment.run(print_results=args.verbose)
    experiment.save_results()

    # Also store basic LLM info for provenance
    if lm is not None:
        from autometrics.experiments.results import JSONResult

        llm_info = {
            "llm_name": args.llm,
            "api_base": args.llm_api_base,
            "model_type": args.llm_model_type,
        }
        JSONResult(llm_info).save(base_output_dir, "llm_info")

    logger.info("Recommendation benchmarking complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
