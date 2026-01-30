from abc import ABC, abstractmethod
import dspy
from typing import Callable, List, Optional
from autometrics.metrics.Metric import Metric
from autometrics.util.format import get_default_formatter

class Generator(ABC):

    def __init__(
        self,
        name,
        description,
        generator_llm: dspy.LM = None,
        executor_class: type = None,
        executor_kwargs: dict = None,
        *,
        truncate_chars: Optional[int] = None,
    ):
        self.name = name
        self.description = description
        self.generator_llm = generator_llm
        self.executor_class = executor_class
        self.executor_kwargs = executor_kwargs
        self.formatter_truncate_chars = truncate_chars

    @abstractmethod
    def generate(self, dataset, target_measure: str, n_metrics: int = 5, **kwargs) -> List[Metric]:
        """
        Generate new metrics based on the dataset and task description
        """
        pass

    def _get_formatter(self, dataset):
        if not dataset:
            return lambda x: str(x)
        return get_default_formatter(dataset, truncate_text=self.formatter_truncate_chars)

    def _resolve_formatter(self, dataset, formatter: Optional[Callable] = None):
        if formatter is not None:
            return formatter
        return self._get_formatter(dataset)

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return f"{self.name}: {self.description}"
