import dspy
from autometrics.metrics.Metric import Metric
from autometrics.util.format import get_default_formatter
from concurrent.futures import ThreadPoolExecutor, as_completed
from autometrics.metrics.Metric import MetricResult
from tqdm import tqdm
import math

class LLMAsAJudgeSignature(dspy.Signature):
    """Given an input text, the task description that the model was trying to follow, and a metric to rate the text on, return a score from 1-5 on this metric."""
    text = dspy.InputField(desc="The input text that we want to rate.")
    task_description = dspy.InputField(desc="A description of the task that the model was trying to solve when it generated the text.  Could be left blank if not available.")
    metric = dspy.InputField(desc="The metric that we want to rate the text on.")
    score = dspy.OutputField(desc="The score that the text should recieve on this metric (1=low, 5=high).")

class LLMAsAJudge(dspy.Module):
    def __init__(self):
        super(LLMAsAJudge, self).__init__()
        self.generate_score = dspy.ChainOfThought(LLMAsAJudgeSignature)

    def forward(self, text, metric, task_description=None, lm=None):
        if task_description is None:
            task_description = "None"
        result = self.generate_score(task_description=task_description, text=text, metric=metric, lm=lm)
        score = result.score
        # Convert the string score to a float by stripping any additional text and converting to a float
        if '\n' in score:
            score = score.split('\n')[0]

        if '.' in score:
            score = score.split('.')[0]
            
        try:
            score = float(score.strip())
        except:
            score = 0.0

        pred = dspy.Prediction(text=text, metric=metric, score=score)
        # Attach reasoning if provided by DSPy
        try:
            setattr(pred, 'reasoning', getattr(result, 'reasoning', ''))
        except Exception:
            pass
        return pred
    
def grade_row(row, axis, llm, formatter, task_description):
    '''Helper function to grade a single row and return MetricResult(score, feedback).'''
    with dspy.settings.context(lm=llm):
        pred = LLMAsAJudge()(formatter(row), axis, task_description, lm=llm)
    # Parse score to float defensively
    try:
        score_val = float(pred.score)
    except Exception:
        score_val = float(str(pred.score).strip())
    feedback = getattr(pred, 'reasoning', '')
    return MetricResult(score=score_val, feedback=feedback)
    
class LLMJudge(Metric):
    has_feedback: bool = True
    def __init__(self, name, description, model, dataset, evaluation_axis, formatter=None, task_description=None):
        # Convert model to string representation for caching if needed
        if hasattr(model, 'model'):
            model_str = str(model.model)
        else:
            model_str = str(model)
            
        # Pass all parameters explicitly to parent constructor for caching
        super().__init__(
            name=name, 
            description=description,
            model=model,  # The actual model object
            model_str=model_str,  # String representation for caching
            dataset=dataset,
            evaluation_axis=evaluation_axis,
            task_description=task_description
        )
        
        self.model = model
        if formatter is None:
            self.formatter = get_default_formatter(dataset)
        else:
            self.formatter = formatter
        self.dataset = dataset
        self.task_description = task_description
        self.evaluation_axis = evaluation_axis
        
        # Exclude dataset from cache key as it doesn't affect results directly
        self.exclude_from_cache_key('dataset')

    def _calculate_with_feedback_impl(self, input, output, references=None, **kwargs):
        # Minimal: leverage grade_row for a single synthesized row
        row = {self.dataset.get_input_column(): input, self.dataset.get_output_column(): output}
        if references is not None:
            for i, ref in enumerate(references):
                row[self.dataset.get_reference_columns()[i]] = ref
        return grade_row(row, self.evaluation_axis, self.model, self.formatter, self.task_description)

    def _calculate_batched_with_feedback_impl(self, inputs, outputs, references=None, num_workers=64, **kwargs):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Prepare synthesized rows to reuse grade_row
        df_cols = {
            'input': self.dataset.get_input_column(),
            'output': self.dataset.get_output_column(),
            'refs': self.dataset.get_reference_columns() or []
        }
        if references is None:
            refs_list = [None] * len(inputs)
        else:
            refs_list = references
        results = [None] * len(inputs)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            def build_row(i):
                row = {df_cols['input']: inputs[i], df_cols['output']: outputs[i]}
                if refs_list and refs_list[i] is not None and df_cols['refs']:
                    for j, ref in enumerate(refs_list[i]):
                        if j < len(df_cols['refs']):
                            row[df_cols['refs'][j]] = ref
                return row
            futures = {
                executor.submit(grade_row, build_row(i), self.evaluation_axis, self.model, self.formatter, self.task_description): i
                for i in range(len(inputs))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    # Maintain score-only implementations for compatibility and abstract base
    def _calculate_impl(self, input, output, references=None, **kwargs):
        mr = self._calculate_with_feedback_impl(input, output, references, **kwargs)
        return mr.score

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        wrapped = self._calculate_batched_with_feedback_impl(inputs, outputs, references, **kwargs)
        return [w.score for w in wrapped]

    def calculate_row(self, row, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric
        """
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()

        if not input_column:
            raise ValueError("Input column not found in dataset.  When constructing your Dataset please provide input_column.")
        if not output_column:
            raise ValueError("Output column not found in dataset.  When constructing your Dataset please provide output_column.")
        
        references = None
        if dataset.get_reference_columns():
            reference_columns = dataset.get_reference_columns()
            references = row[reference_columns]

        input = row[input_column]
        output = row[output_column]

        result = self.calculate(input, output, references, **kwargs)

        if update_dataset:
            row[self.name] = result

        return result
    
    def predict(self, dataset, update_dataset=True, max_workers=64, metric_name=None, with_feedback: bool = True, **kwargs):
            '''
                Grade the dataframe using the LLM judge in parallel with progress bar
            '''
            if metric_name is None:
                metric_name = self.evaluation_axis.split(":")[0].replace("*", "") + "_" + self.model.model.split("/")[-1]

            df = dataset.get_dataframe()

            results = [None] * len(df)

            # Create a ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks to the executor
                futures = {executor.submit(grade_row, row, self.evaluation_axis, self.model, self.formatter, self.task_description): index for index, row in df.iterrows()}

                # Collect the results with tqdm progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Grading rows", unit="row"):
                    index = futures[future]
                    results[index] = future.result()

            scores = [r.score if r and not math.isnan(r.score) else 0.0 for r in results]
            feedback_vals = [r.feedback for r in results]

            if update_dataset:
                df[metric_name] = scores
                if with_feedback and getattr(self, 'has_feedback', False) and feedback_vals is not None:
                    df[f"{metric_name}__feedback"] = feedback_vals
                dataset.set_dataframe(df)
                if metric_name not in dataset.get_metric_columns():
                    dataset.get_metric_columns().append(metric_name)

            return scores