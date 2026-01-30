from autometrics.metrics.Metric import Metric
from autometrics.metrics.MultiMetric import MultiMetric
from typing import List, Optional, Dict, Any, Union, Tuple

class PairwiseMetric:
    """
    A metric that computes a pairwise comparison between two model outputs.
    
    It can either wrap a regular Metric and calculates the difference between the metric value
    for each output, or implement direct pairwise comparison.
    
    By default for wrapped metrics, it computes:
    score = metric.calculate(input, output1, references) - metric.calculate(input, output2, references)
    
    This can be overridden in subclasses for different comparison behaviors.
    """
    def __init__(self, scalar_metric: Optional[Metric] = None, name=None, description=None, **kwargs):

        if scalar_metric is not None and isinstance(scalar_metric, MultiMetric):
            raise ValueError("PairwiseMetric cannot wrap a MultiMetric.  Instead, use PairwiseMultiMetric.")

        self.metric = scalar_metric
        
        if scalar_metric:
            # When wrapping a metric, use its name/description if not provided
            self.name = name or f"pairwise_{scalar_metric.get_name()}"
            self.description = description or f"Pairwise comparison using {scalar_metric.get_description()}"
        else:
            # When creating a direct pairwise metric
            if not name or not description:
                raise ValueError("For direct pairwise metrics (no scalar_metric), name and description must be provided")
            self.name = name
            self.description = description
            
        self.kwargs = kwargs
    
    def calculate(self, input, output_1, output_2, references=None, **kwargs):
        """
        Calculate the pairwise metric between two outputs.
        
        Args:
            input: The input text or data point
            output_1: The first output to compare
            output_2: The second output to compare
            references: Optional reference text(s) for reference-based metrics
            **kwargs: Additional arguments passed to the underlying metric
            
        Returns:
            The pairwise metric value
        """
        if self.metric:
            # When wrapping a scalar metric
            val1 = self.metric.calculate(input, output_1, references, **{**self.kwargs, **kwargs})
            val2 = self.metric.calculate(input, output_2, references, **{**self.kwargs, **kwargs})
            return self._combine_results(val1, val2)
        else:
            # Direct pairwise computation - must be implemented by subclasses
            return self._calculate_pairwise_impl(input, output_1, output_2, references, **kwargs)
    
    def calculate_batched(self, inputs, outputs_1, outputs_2, references=None, **kwargs):
        """
        Calculate metrics for batches of inputs and pairs of outputs.
        
        Args:
            inputs: List of input texts or data points
            outputs_1: List of output texts or data points for the first model
            outputs_2: List of output texts or data points for the second model
            references: Optional list of reference texts for reference-based metrics
            **kwargs: Additional arguments passed to the underlying metric
            
        Returns:
            List of pairwise metric values for each input-output pair
        """
        if len(outputs_1) != len(inputs) or len(outputs_2) != len(inputs):
            raise ValueError(f"Inputs ({len(inputs)}), outputs_1 ({len(outputs_1)}), and outputs_2 ({len(outputs_2)}) must have the same length")
            
        if self.metric:
            # Use the underlying metric's batched calculation
            vals1 = self.metric.calculate_batched(inputs, outputs_1, references, **{**self.kwargs, **kwargs})
            vals2 = self.metric.calculate_batched(inputs, outputs_2, references, **{**self.kwargs, **kwargs})
            
            # Calculate differences for each pair
            return [self._combine_results(v1, v2) for v1, v2 in zip(vals1, vals2)]
        else:
            # Direct pairwise batch computation
            return self._calculate_batched_pairwise_impl(inputs, outputs_1, outputs_2, references, **kwargs)
    
    def calculate_row(self, row, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for a row in a PairwiseDataset.
        
        Args:
            row: The dataframe row to process
            dataset: A PairwiseDataset instance
            update_dataset: Whether to update the dataset with the results
            **kwargs: Additional arguments passed to calculate
            
        Returns:
            The pairwise metric value for this row
        """
        input_column = dataset.get_input_column()
        
        if not input_column:
            raise ValueError("Input column not found in dataset. When constructing your Dataset please provide input_column.")
            
        # Get the input from the dataset row
        input = row[input_column]
        
        # Get both outputs
        output_1 = row[dataset.get_output_column_1()]
        output_2 = row[dataset.get_output_column_2()]
        
        # Get references if available
        references = None
        if dataset.get_reference_columns():
            reference_columns = dataset.get_reference_columns()
            references = row[reference_columns]
        
        # Calculate the pairwise metric
        result = self.calculate(input, output_1, output_2, references, **kwargs)
        
        # Update the dataset if requested
        if update_dataset:
            row[self.name] = result
            
        return result
    
    def predict(self, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for an entire dataset.
        
        Args:
            dataset: The dataset to process
            update_dataset: Whether to update the dataset with the results
            **kwargs: Additional arguments passed to calculate
            
        Returns:
            List of pairwise metric values for each row in the dataset
        """
        df = dataset.get_dataframe()
        output_column_1 = dataset.get_output_column_1()
        output_column_2 = dataset.get_output_column_2()
        input_column = dataset.get_input_column()
        
        if not input_column:
            raise ValueError("Input column not found in dataset")
        if not output_column_1 or not output_column_2:
            raise ValueError("Output columns for both models must be specified")
        
        # Get inputs and outputs
        inputs = df[input_column].tolist()
        outputs_1 = df[output_column_1].tolist()
        outputs_2 = df[output_column_2].tolist()
        
        # Get references if available
        references = None
        reference_columns = dataset.get_reference_columns()
        if reference_columns:
            references = [row[reference_columns] for _, row in df.iterrows()]
        
        # Calculate metrics in batch
        results = self.calculate_batched(inputs, outputs_1, outputs_2, references, **kwargs)
        
        # Update the dataset if requested
        if update_dataset:
            df[self.name] = results
            dataset.set_dataframe(df)
            
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)
                
        return results
    
    def _combine_results(self, val1, val2):
        """
        Combine the two metric values into a final result.
        By default, this returns val1 - val2.
        
        This method can be overridden in subclasses to implement different
        comparison behaviors.
        
        Args:
            val1: The metric value for the first output
            val2: The metric value for the second output
            
        Returns:
            The combined/compared result
        """
        return val1 - val2
    
    def _calculate_pairwise_impl(self, input, output_1, output_2, references=None, **kwargs):
        """
        Implement direct pairwise calculation between two outputs.
        
        This method must be implemented by subclasses that don't wrap a scalar metric.
        The default implementation raises NotImplementedError.
        
        Args:
            input: The input text or data
            output_1: The first output to compare
            output_2: The second output to compare
            references: Optional reference data
            **kwargs: Additional parameters
            
        Returns:
            A metric value representing the comparison between outputs
        """
        if self.metric:
            # When wrapping a scalar metric, use the scalar metrics for each output and combine
            val1 = self.metric.calculate(input, output_1, references, **kwargs)
            val2 = self.metric.calculate(input, output_2, references, **kwargs)
            return self._combine_results(val1, val2)
        else:
            raise NotImplementedError("Direct pairwise metrics must implement _calculate_pairwise_impl")
    
    def _calculate_batched_pairwise_impl(self, inputs, outputs_1, outputs_2, references=None, **kwargs):
        """
        Implement direct pairwise calculation between two outputs.
        
        Default implementation is to loop through the inputs and call the pairwise calculation for each input.
        
        Args:
            inputs: List of input texts or data
            outputs_1: List of output texts from first model
            outputs_2: List of output texts from second model
            references: Optional list of reference data
            **kwargs: Additional parameters
            
        Returns:
            List of metric values representing the comparisons
        """
        results = []
        
        # Handle references properly
        if references is None:
            # If no references, use None for each input
            for i, (input, output_1, output_2) in enumerate(zip(inputs, outputs_1, outputs_2)):
                results.append(self.calculate(input, output_1, output_2, None, **kwargs))
        else:
            # If references provided, pair each with its input
            for i, (input, output_1, output_2) in enumerate(zip(inputs, outputs_1, outputs_2)):
                ref = references[i] if i < len(references) else None
                results.append(self.calculate(input, output_1, output_2, ref, **kwargs))
                
        return results
    
    def get_name(self):
        """Return the name of this pairwise metric"""
        return self.name
    
    def get_description(self):
        """Return the description of this pairwise metric"""
        return self.description
    
    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__()
        
