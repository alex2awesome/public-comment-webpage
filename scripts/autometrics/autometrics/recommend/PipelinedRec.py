from typing import List, Type

from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
import dspy
from autometrics.recommend.LLMRec import LLMRec

class PipelinedRec(MetricRecommender):
    """
    A metric recommender that uses a pipeline of recommenders to recommend metrics.
    """
    def __init__(self, metric_classes: List[Type[Metric]], recommenders: List[Type[MetricRecommender]], top_ks: List[int], index_paths: List[str], force_reindex: bool = False, model: dspy.LM = None, use_description_only: bool = False):
        super().__init__(metric_classes, index_paths[0], force_reindex)
        self.recommenders = recommenders
        self.top_ks = top_ks
        self.index_paths = index_paths
        self.force_reindex = force_reindex
        self.model = model
        self.use_description_only = use_description_only

        if len(self.recommenders) != len(self.top_ks):
            if len(self.top_ks) == 1:
                self.top_ks = self.top_ks * len(self.recommenders)
            elif not (len(self.top_ks) == len(self.recommenders) - 1):
                raise ValueError("Must provide the same number of top_ks as recommenders")
        elif len(self.top_ks) > len(self.recommenders):
            raise ValueError("Must provide the same number of top_ks as recommenders")

        # Check that top k is monotonically decreasing
        if not all(top_ks[i] >= top_ks[i+1] for i in range(len(top_ks) - 1)):
            raise ValueError("Top k must be monotonically decreasing")
        
        if len(self.index_paths) != len(self.recommenders):
            # pad with None
            self.index_paths = self.index_paths + [None] * (len(self.recommenders) - len(self.index_paths))

        super().__init__(metric_classes, index_paths[0], force_reindex)

    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        results = self.metric_classes
        top_ks = self.top_ks
        if len(self.top_ks) < len(self.recommenders):
            if len(self.top_ks) == 1:
                top_ks = [self.top_ks[0]] * len(self.recommenders)
            elif len(self.top_ks) == len(self.recommenders) - 1:
                top_ks = self.top_ks + [k]
            else:
                raise ValueError("Must provide the same number of top_ks as recommenders")
        elif len(self.top_ks) > len(self.recommenders):
            raise ValueError("Must provide the same number of top_ks as recommenders")
        
        if k != top_ks[-1]:
            raise ValueError("Must provide the same k for the last recommender")
        
        # Check that top k is monotonically decreasing
        if not all(top_ks[i] >= top_ks[i+1] for i in range(len(top_ks) - 1)):
            raise ValueError("Top k must be monotonically decreasing")

        for recommender_cls, top_k, index_path in zip(self.recommenders, top_ks, self.index_paths):
            if recommender_cls is LLMRec:
                if index_path is None:
                    recommender = recommender_cls(metric_classes=results, force_reindex=self.force_reindex, model=self.model, use_description_only=self.use_description_only)
                else:
                    recommender = recommender_cls(metric_classes=results, index_path=index_path, force_reindex=self.force_reindex, model=self.model, use_description_only=self.use_description_only)
            else:
                if index_path is None:
                    recommender = recommender_cls(metric_classes=results, force_reindex=self.force_reindex, use_description_only=self.use_description_only)
                else:
                    recommender = recommender_cls(metric_classes=results, index_path=index_path, force_reindex=self.force_reindex, use_description_only=self.use_description_only)
            results = recommender.recommend(dataset, target_measurement, top_k)
            # drop any "None"s from results
            results = [r for r in results if r is not None]
        return results