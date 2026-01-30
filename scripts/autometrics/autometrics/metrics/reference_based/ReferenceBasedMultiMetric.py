from autometrics.metrics.MultiMetric import MultiMetric

class ReferenceBasedMultiMetric(MultiMetric):
    """
    Abstract class for reference-based metrics
    """
    def __init__(self, name, description, submetric_names=[], **kwargs) -> None:
        super().__init__(name, description, submetric_names, **kwargs)

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

        results = self.calculate(input, output, references, **kwargs)

        if update_dataset:
            for idx, name in enumerate(self.submetric_names):
                row[name] = results[idx]

        return results

    def predict(self, dataset, update_dataset=True, **kwargs):
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

        results = self.calculate_batched(inputs, outputs, references)

        if update_dataset:
            for idx, name in enumerate(self.submetric_names):
                # Use assign to avoid chained assignment issues by creating a new DataFrame
                df = df.assign(**{name: results[idx]})
                dataset.set_dataframe(df)

                if name not in dataset.get_metric_columns():
                    dataset.get_metric_columns().append(name)

        return results

