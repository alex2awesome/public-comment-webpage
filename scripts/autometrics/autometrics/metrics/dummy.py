from autometrics.metrics.Metric import Metric

class DummyMetric(Metric):
    """
    Dummy metric when the metric is precomputed.  This will not generalize, but can be used for testing and development.
    """
    def __init__(self, name=None, description=None):
        super().__init__(name, description)

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        pass

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        pass

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate the metric
        """
        pass

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate the metric
        """
        pass

    def predict(self, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        pass