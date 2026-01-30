from autometrics.metrics.Metric import Metric

class ReferenceBasedMetric(Metric):
    """
    Abstract class for reference-based metrics
    """
    def __init__(self, name, description, **kwargs):
        super().__init__(name, description, **kwargs)

    def calculate_row(self, row, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric
        """
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()
        reference_columns = dataset.get_reference_columns()

        if not input_column:
            raise ValueError("Input column not found in dataset.  When constructing your Dataset please provide input_column.")
        if not output_column:
            raise ValueError("Output column not found in dataset.  When constructing your Dataset please provide output_column.")
        if not reference_columns:
            raise ValueError("Reference columns not found in dataset.  This is required for reference based metrics.  When constructing your Dataset please provide reference_columns.")

        input = row[input_column]
        output = row[output_column]
        references = row[reference_columns]

        result = self.calculate(input, output, references, **kwargs)

        if update_dataset:
            row[self.name] = result

        return result

    def predict(self, dataset, update_dataset=True, with_feedback: bool = True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        df = dataset.get_dataframe()
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()
        reference_columns = dataset.get_reference_columns()

        if not input_column:
            raise ValueError("Input column not found in dataset.  When constructing your Dataset please provide input_column.")
        if not output_column:
            raise ValueError("Output column not found in dataset.  When constructing your Dataset please provide output_column.")
        if not reference_columns:
            raise ValueError("Reference columns not found in dataset.  This is required for reference based metrics.  When constructing your Dataset please provide reference_columns.")

        inputs = df[input_column].values.tolist()
        outputs = df[output_column].values.tolist()
        references = df[reference_columns].values.tolist()

        if with_feedback and getattr(self, 'has_feedback', False):
            results_wrapped = self.calculate_batched_with_feedback(inputs, outputs, references)
            results = [r.score for r in results_wrapped]
            feedback_vals = [r.feedback for r in results_wrapped]
        else:
            results = self.calculate_batched(inputs, outputs, references)

        if update_dataset:
            # Use assign to create a new DataFrame column without chained assignment issues
            df = df.assign(**{self.name: results})
            if with_feedback and getattr(self, 'has_feedback', False):
                df = df.assign(**{f"{self.name}__feedback": feedback_vals})
            dataset.set_dataframe(df)

            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)

        return results

