from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from abc import ABC, abstractmethod
import os
class Experiment(ABC):

    def __init__(self, name: str, description: str, metrics: list[Metric], output_dir: str, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, seed: int = 42, **kwargs):
        self.name = name
        self.description = description
        self.metrics = metrics
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.seed = seed
        self.kwargs = kwargs
        self.results = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __init__(self, name: str, description: str, metrics: list[Metric], output_dir: str, dataset: Dataset, seed: int = 42, should_split: bool = True, **kwargs):
        self.name = name
        self.description = description
        self.metrics = metrics
        self.output_dir = output_dir
        if should_split:
            self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_splits(seed=seed)
        else:
            self.train_dataset = dataset
            self.val_dataset = dataset
            self.test_dataset = dataset
        self.seed = seed
        self.kwargs = kwargs
        self.results = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @abstractmethod
    def run(self, print_results: bool = False):
        pass

    def save_results(self):
        for key, value in self.results.items():
            full = os.path.join(self.output_dir, key)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            value.save(self.output_dir, key)

class ExperimentRunner:
    def __init__(self, experiments: list[Experiment]):
        self.experiments = experiments

    def add_experiment(self, experiment: Experiment):
        self.experiments.append(experiment)

    def run(self, print_results: bool = False):
        for experiment in self.experiments:
            if print_results:
                print(f"Running experiment {experiment.name}")
                print(f"Description: {experiment.description}")
            experiment.run(print_results)
            experiment.save_results()