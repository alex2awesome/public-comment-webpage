from typing import Type
from autometrics.metrics.MetricBank import all_metric_classes
from autometrics.metrics.Metric import Metric

metric_map = { metric_class.__name__.upper().strip(): metric_class for metric_class in all_metric_classes }

def metric_name_to_class(metric_name: str) -> Type[Metric]:
    return metric_map.get(metric_name.upper().strip(), None)