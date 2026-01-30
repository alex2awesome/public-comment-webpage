import dspy
import os
from typing import Callable, Optional

# Heavy imports moved to lazy loading to avoid 90s startup delay
from autometrics.recommend.MetricRecommender import MetricRecommender
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric
from autometrics.dataset.Dataset import DummyDataset
from platformdirs import user_data_dir


colbert_index_path = os.path.join(user_data_dir("autometrics"), "colbert_all_metrics")
bm25_index_path = os.path.join(user_data_dir("autometrics"), "bm25_all_metrics")

recommender_global = None

def generate_metric_details(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
    """Generate the details of a given metric."""
    return f"""- **Name:** {metric.name}
- **Description:** {metric.description}"""

def generate_intended_use(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
    """Generate the intended use of a given metric."""
    return """- None (The intended use should be overridden in the generated metric class.)"""

def generate_metric_implementation(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
    """Generate the implementation of a given metric."""
    # METRIC CARD MUST BE FALSE, otherwise this will trigger a recursive call to this function
    return f"```python\n{metric._generate_python_code(include_metric_card=False)}\n```"

def generate_known_limitations(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
    """Generate known limitations for a given metric."""
    return """- **Known Limitations:**
  - None (The known limitations should be overridden in the generated metric class.)"""

def generate_related_metrics(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric, recommender: Optional[MetricRecommender] = None, force_reindex=False):
    """Generate related metrics for a given metric."""
    global recommender_global

    if recommender is None:
        if recommender_global is None:
            # Lazy import heavy modules only when first needed (avoids 90s startup delay)
            from autometrics.metrics.MetricBank import all_metric_classes
            from autometrics.recommend.BM25 import BM25
            from autometrics.recommend.ColBERT import ColBERT
            # Try to initialize BM25 recommender first
            try:
                print(f"Initializing BM25 recommender with index path: {bm25_index_path}")
                recommender_global = BM25(metric_classes=all_metric_classes, index_path=bm25_index_path)
                print("BM25 recommender loaded successfully")
            except Exception as bm25_error:
                print(f"Error initializing BM25 recommender: {bm25_error}")
                print("Falling back to ColBERT recommender...")
                try:
                    recommender_global = ColBERT(metric_classes=all_metric_classes, index_path=colbert_index_path, force_reindex=force_reindex)
                    # Check if the searcher is properly initialized
                    if hasattr(recommender_global, 'index') and hasattr(recommender_global.index, 'searcher'):
                        if recommender_global.index.searcher is None:
                            print("Warning: ColBERT index searcher is None, rebuilding index...")
                            recommender_global = ColBERT(metric_classes=all_metric_classes, index_path=colbert_index_path, force_reindex=True)
                            print("ColBERT index rebuilt successfully")
                        else:
                            print("ColBERT index loaded successfully")
                    else:
                        print("Warning: ColBERT index doesn't have searcher attribute, rebuilding...")
                        recommender_global = ColBERT(metric_classes=all_metric_classes, index_path=colbert_index_path, force_reindex=True)
                        print("ColBERT index rebuilt successfully")
                except Exception as colbert_error:
                    print(f"Failed to initialize ColBERT recommender: {colbert_error}")
                    print("No recommender could be initialized.")
                    recommender_global = None
        recommender = recommender_global

    if recommender is None:
        print("No recommender available for related metrics.")
        return """- **Related Metrics:**\n  - None"""

    try:
        dataset = DummyDataset(task_description=f"NA; Please recommend a related metric to {metric.name}.")
        results = recommender.recommend(dataset=dataset, target_measurement=metric.name, k=3)

        results_str = ""
        for result in results:
            results_str += f"\n  - **{result.__name__}:** {result.description.split('.')[0] + '.'}"

        return f"""- **Related Metrics:**{results_str}"""
    
    except Exception as e:
        print(f"Error generating related metrics: {e}")
        return """- **Related Metrics:**\n  - None"""

def generate_further_reading(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
    """Generate further reading for a given metric."""
    return """- **Papers:**
  - [Autometrics](https://github.com/XenonMolecule/autometrics)"""

def generate_citation(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
    """Generate a citation for a given metric."""
    return """```
@software{Ryan_Autometrics_2025,
    author = {Ryan, Michael J. and Zhang, Yanzhe and Salunkhe, Amol and Chu, Yi and Rahman, Emily and Xu, Di and Yang, Diyi},
    license = {MIT},
    title = {{Autometrics}},
    url = {https://github.com/XenonMolecule/autometrics},
    version = {1.0.0},
    year = {2025}
}
```"""

def generate_author(metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
    """Generate the author of the metric."""

    author = getattr(metric, "metric_card_author_model", None)
    if author is not None:
        author = getattr(author, "model", author)
    else:
        author = getattr(metric, "model_str", None)

    author = str(author)
    author = author.split("/")[-1]

    return f"""- **Authors:** This metric card was automatically generated by {author}.
- **Acknowledgement of AI Assistance:** This metric card was entirely automatically generated by {author} using the Autometrics library. No human intervention was involved. User discretion is advised.
- **Contact:** For questions about the autometrics library, please contact [Michael J Ryan](mailto:mryan0@stanford.edu)."""

class MetricCardBuilder:
    """Simple builder with instance methods that can be easily overridden."""
    
    def __init__(self, metric: GeneratedRefFreeMetric | GeneratedRefBasedMetric):
        self.metric = metric
    
    def metric_details(self) -> str:
        return generate_metric_details(self.metric)
    
    def intended_use(self) -> str:
        return generate_intended_use(self.metric)
    
    def metric_implementation(self) -> str:
        return generate_metric_implementation(self.metric)
    
    def known_limitations(self) -> str:
        return generate_known_limitations(self.metric)
    
    def related_metrics(self) -> str:
        return generate_related_metrics(self.metric)
    
    def further_reading(self) -> str:
        return generate_further_reading(self.metric)
    
    def citation(self) -> str:
        return generate_citation(self.metric)
    
    def author(self) -> str:
        return generate_author(self.metric)
    
    def build(self) -> str:
        """Render the full metric-card markdown."""
        return (
            f"---\n# Metric Card for {self.metric.name}\n\n"
            f"{self.metric.description}\n\n"
            "## Metric Details\n\n" + self.metric_details() + "\n\n"
            "## Intended Use\n\n" + self.intended_use() + "\n\n"
            "## Metric Implementation\n\n" + self.metric_implementation() + "\n\n"
            "## Known Limitations\n\n" + self.known_limitations() + "\n\n"
            "## Related Metrics\n\n" + self.related_metrics() + "\n\n"
            "## Further Reading\n\n" + self.further_reading() + "\n\n"
            "## Citation\n\n" + self.citation() + "\n\n"
            "## Metric Card Authors\n\n" + self.author()
        )