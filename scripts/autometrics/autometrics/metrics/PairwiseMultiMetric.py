from autometrics.metrics.PairwiseMetric import PairwiseMetric
from autometrics.metrics.MultiMetric import MultiMetric
from typing import List, Optional, Dict, Any, Union, Tuple

class PairwiseMultiMetric(PairwiseMetric):
    """
    A specialized PairwiseMetric for handling MultiMetrics that return multiple scores.
    
    This class wraps a MultiMetric and applies pairwise comparison for each submetric,
    preserving the structure of the multiple scores.
    """
    
    def __init__(self, multi_metric: MultiMetric, name=None, description=None, **kwargs):
        """
        Initialize a PairwiseMultiMetric.
        
        Args:
            multi_metric: The MultiMetric to wrap for pairwise comparison
            name: Custom name for the metric (defaults to pairwise_{multi_metric.get_name()})
            description: Custom description (defaults to Pairwise comparison using {multi_metric.get_description()})
            **kwargs: Additional keyword arguments for the metric
        """
        if not isinstance(multi_metric, MultiMetric):
            raise ValueError("PairwiseMultiMetric must be initialized with a MultiMetric")
        
        # Initialize name and description if not provided
        if name is None:
            name = f"pairwise_{multi_metric.get_name()}"
        if description is None:
            description = f"Pairwise comparison using {multi_metric.get_description()}"
        
        # Initialize directly without calling parent constructor with multi_metric
        # Instead we'll bypass the MultiMetric check by calling PairwiseMetric.__init__ with scalar_metric=None
        PairwiseMetric.__init__(self, scalar_metric=None, name=name, description=description, **kwargs)
        
        # Store the multi_metric separately
        self.metric = multi_metric
        
        # Get submetric names from the MultiMetric and create pairwise versions
        self.submetric_names = [f"pairwise_{name}" for name in multi_metric.get_submetric_names()]
        
    def get_submetric_names(self) -> List[str]:
        """
        Get the names of submetrics.
        
        Returns:
            List of submetric names
        """
        return self.submetric_names
        
    def _combine_results(self, val1, val2):
        """
        Combine the two sets of metric values into final results.
        For MultiMetrics, this means comparing each submetric individually.
        
        Args:
            val1: The metric values for the first output (tuple/list for MultiMetric)
            val2: The metric values for the second output (tuple/list for MultiMetric)
            
        Returns:
            List or tuple of combined results, one for each submetric
        """
        # Both val1, val2 should have the same structure (list/tuple)
        if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                raise ValueError(f"Mismatched submetric counts: {len(val1)} vs {len(val2)}")
                
            # Apply difference to each component separately
            return [v1 - v2 for v1, v2 in zip(val1, val2)]
        else:
            # If not list/tuple, use the standard difference
            return super()._combine_results(val1, val2)
            
    def calculate_batched(self, inputs, outputs_1, outputs_2, references=None, **kwargs):
        """
        Calculate batched pairwise metrics for MultiMetric.
        
        Args:
            inputs: List of inputs
            outputs_1: List of first outputs
            outputs_2: List of second outputs
            references: Optional list of references
            **kwargs: Additional arguments
            
        Returns:
            List of lists, with each inner list containing the differences for one input
        """
        if len(outputs_1) != len(inputs) or len(outputs_2) != len(inputs):
            raise ValueError(f"Inputs ({len(inputs)}), outputs_1 ({len(outputs_1)}), and outputs_2 ({len(outputs_2)}) must have the same length")
        
        # Calculate metrics for each set of outputs
        # MultiMetric.calculate_batched returns a list where the outer list is for each submetric, 
        # and the inner list is for each input
        vals1 = self.metric.calculate_batched(inputs, outputs_1, references, **{**self.kwargs, **kwargs})
        vals2 = self.metric.calculate_batched(inputs, outputs_2, references, **{**self.kwargs, **kwargs})
        
        # Transpose the results to get one list per input
        # This makes vals1_by_input[i] = [submetric1_val, submetric2_val, ...] for input i
        vals1_by_input = list(zip(*vals1))
        vals2_by_input = list(zip(*vals2))
        
        # For each input, calculate the difference between output1 and output2 metrics
        result_by_input = []
        for v1_by_input, v2_by_input in zip(vals1_by_input, vals2_by_input):
            result_by_input.append(self._combine_results(v1_by_input, v2_by_input))
        
        # Return results where result_by_input[i] is the pairwise metrics for input i
        return result_by_input
    
    def predict(self, dataset, update_dataset: bool = True, **kwargs):
        """
        Calculate the metric for an entire dataset.
        For MultiMetrics, this updates multiple columns in the dataset.
        
        Args:
            dataset: The dataset to process
            update_dataset: Whether to update the dataset with the results
            **kwargs: Additional arguments passed to calculate
            
        Returns:
            List of lists of pairwise metric values
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
        
        # Calculate metrics for each output separately
        vals1 = self.metric.calculate_batched(inputs, outputs_1, references, **{**self.kwargs, **kwargs})
        vals2 = self.metric.calculate_batched(inputs, outputs_2, references, **{**self.kwargs, **kwargs})
        
        # Calculate differences for each submetric
        result_by_submetric = []
        for submetric_idx in range(len(self.get_submetric_names())):
            if submetric_idx < len(vals1) and submetric_idx < len(vals2):
                # Calculate differences for this submetric
                submetric_diffs = [v1 - v2 for v1, v2 in zip(vals1[submetric_idx], vals2[submetric_idx])]
                result_by_submetric.append(submetric_diffs)
        
        # Update the dataset if requested
        if update_dataset:
            for i, submetric_name in enumerate(self.get_submetric_names()):
                if i < len(result_by_submetric):
                    df[submetric_name] = result_by_submetric[i]
                    
                    if submetric_name not in dataset.get_metric_columns():
                        dataset.get_metric_columns().append(submetric_name)
            
            dataset.set_dataframe(df)
                
        # For consistency with the interface, transpose the results to match the expected format
        # from calculate_batched
        results_by_input = list(zip(*result_by_submetric)) if result_by_submetric else []
        return [list(r) for r in results_by_input]

    def calculate(self, input, output_1, output_2, references=None, **kwargs):
        """
        Calculate pairwise comparison for MultiMetric.
        
        Args:
            input: The input text or data
            output_1: The first output to compare
            output_2: The second output to compare
            references: Optional reference data
            **kwargs: Additional arguments
            
        Returns:
            List of differences, one per submetric
        """
        # Use the wrapped MultiMetric to calculate values for each output
        val1 = self.metric.calculate(input, output_1, references, **{**self.kwargs, **kwargs})
        val2 = self.metric.calculate(input, output_2, references, **{**self.kwargs, **kwargs})
        
        # Calculate differences for each submetric value
        return [v1 - v2 for v1, v2 in zip(val1, val2)]
    
    def calculate_batched(self, inputs, outputs_1, outputs_2, references=None, **kwargs):
        """
        Calculate batched pairwise metrics for MultiMetric.
        
        Args:
            inputs: List of inputs
            outputs_1: List of first outputs
            outputs_2: List of second outputs
            references: Optional list of references
            **kwargs: Additional arguments
            
        Returns:
            List of lists, with each inner list containing the differences for one input
        """
        if len(outputs_1) != len(inputs) or len(outputs_2) != len(inputs):
            raise ValueError(f"Inputs ({len(inputs)}), outputs_1 ({len(outputs_1)}), and outputs_2 ({len(outputs_2)}) must have the same length")
        
        # Calculate metrics for each output separately
        # MultiMetric.calculate_batched returns a list where the outer list is for each submetric
        # and the inner list is for each input
        vals1 = self.metric.calculate_batched(inputs, outputs_1, references, **{**self.kwargs, **kwargs})
        vals2 = self.metric.calculate_batched(inputs, outputs_2, references, **{**self.kwargs, **kwargs})
        
        # Calculate differences for each submetric and each input
        result_by_submetric = []
        for submetric_idx in range(len(self.get_submetric_names())):
            if submetric_idx < len(vals1) and submetric_idx < len(vals2):
                # Calculate differences for this submetric across all inputs
                submetric_diffs = [v1 - v2 for v1, v2 in zip(vals1[submetric_idx], vals2[submetric_idx])]
                result_by_submetric.append(submetric_diffs)
        
        # Transpose to get results by input rather than by submetric
        results_by_input = list(zip(*result_by_submetric)) if result_by_submetric else []
        return [list(r) for r in results_by_input] 