from __future__ import annotations

from typing import Dict, List, Callable
import pandas as pd

from autometrics.experiments.experiment import Experiment
from autometrics.experiments.results import TabularResult
from autometrics.recommend.MetricRecommender import MetricRecommender

class RecommendationExperiment(Experiment):
    """Benchmark experiment for metric recommendation systems.

    For every target column of a dataset, each recommender is asked to return an
    ordered list of *k* metric classes.  The ordered recommendations are stored
    as :class:`TabularResult` objects where each row corresponds to one
    recommender and the columns denote the ranked metric names.
    """

    def __init__(
        self,
        name: str,
        description: str,
        recommenders: Dict[str, MetricRecommender],
        output_dir: str,
        dataset: 'Dataset',
        top_k: int = 20,
        seed: int = 42,
        **kwargs,
    ) -> None:
        # We pass an *empty* metrics list to the base Experiment as the
        # recommenders themselves do not need instantiated Metric objects.
        super().__init__(
            name=name,
            description=description,
            metrics=[],
            output_dir=output_dir,
            dataset=dataset,
            seed=seed,
            should_split=False,  # Recommendation works on full dataset
            **kwargs,
        )
        self.recommenders = recommenders  # mapping name -> recommender instance
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def _collect_target_columns(self) -> List[str]:
        """Return the list of target columns from *train_dataset*."""
        if hasattr(self.train_dataset, "get_target_columns"):
            return list(self.train_dataset.get_target_columns())
        if hasattr(self.train_dataset, "target_columns"):
            return list(self.train_dataset.target_columns)
        raise AttributeError(
            "Dataset must expose `get_target_columns()` or `target_columns` attribute."
        )

    def run(self, print_results: bool = False):
        target_cols = self._collect_target_columns()
        if print_results:
            print(f"Running recommendation experiment on {len(target_cols)} target columns …")

        for tgt in target_cols:
            if print_results:
                print(f"\nTarget column: {tgt}")
            rows: Dict[str, List[str]] = {}

            for rec_name, recommender in self.recommenders.items():
                if print_results:
                    print(f"  • {rec_name} …", end="", flush=True)
                metrics = recommender.recommend(
                    dataset=self.train_dataset,
                    target_measurement=tgt,
                    k=self.top_k,
                )
                metric_names = [m.__name__ if m else "ERR" for m in metrics]
                # Ensure fixed width by padding with empty strings
                if len(metric_names) < self.top_k:
                    metric_names += [""] * (self.top_k - len(metric_names))
                if print_results:
                    print("done")
                rows[rec_name] = metric_names

            df = pd.DataFrame.from_dict(
                rows, orient="index", columns=[f"Rank_{i+1}" for i in range(self.top_k)]
            )
            # Key includes target column for separate files
            key = f"{tgt}/recommendations"
            self.results[key] = TabularResult(df)

            if print_results:
                print(df) 