from autometrics.metrics.Metric import Metric

from prometheus_eval.litellm import LiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import math
import pandas as pd
from IPython.display import display, HTML
import dspy
import re

class JudgeByRubricSignature(dspy.Signature):
    """Given an input and output of a model, score the model's output based on a rubric."""
    metric_title = dspy.InputField(desc="The title of the metric.")
    metric_description = dspy.InputField(desc="A description of the metric.")
    rubric = dspy.InputField(desc="A rubric for the metric scoring from 1 to 5.")

    input = dspy.InputField(desc="The input to the model.")
    reference = dspy.InputField(desc="(Optional) An example of a good output for the model.")
    output = dspy.InputField(desc="The output of the model which is to be scored.")

    score = dspy.OutputField(desc="The score of the metric.")

class JudgeByRubric(dspy.Module):
    def __init__(self):
        super(JudgeByRubric, self).__init__()
        self.score_by_rubric = dspy.ChainOfThought(JudgeByRubricSignature)

    def forward(self, task_description, metric_title, metric_description, rubric, input, reference, output, lm=None):
        score = self.score_by_rubric(task_description=task_description, metric_title=metric_title, metric_description=metric_description, rubric=rubric, input=input, reference=reference, output=output, lm=lm).score

        return dspy.Prediction(score=score)


class LLMJudgeRubricDSPy(Metric):
    has_feedback: bool = True
    def __init__(self, name, description, dataset, rubric, judge=None, task_description=None, judge_api_base="http://future-hgx-1:7410/v1"):
        # Initialize default judge if not provided
        if judge is None:
            judge = dspy.LM("openai/meta-llama/Meta-Llama-3.3-70b-Instruct", api_base=judge_api_base, api_key="None")
        
        # Pass all important parameters to parent constructor for caching
        super().__init__(
            name=name, 
            description=description,
            rubric=rubric,
            judge=judge,
            task_description=task_description,
            judge_api_base=judge_api_base,
            dataset=dataset  # We'll exclude this later
        )
        
        self.dataset = dataset
        self.rubric = rubric
        self.judge = judge
        self.task_description = task_description
        
        # Exclude non-affecting parameters from cache key
        self.exclude_from_cache_key('dataset', 'judge_api_base')

    def _calculate_with_feedback_impl(self, input, output, references=None, **kwargs):
        if self.task_description:
            input = self.task_description + "\n\n" + input

        reference = None
        if references is not None:
            reference = references[0] # NOTE: Only one reference is supported

        rubric = SCORE_RUBRIC_TEMPLATE.format(**self.rubric)
        
        with dspy.context(lm=self.judge):
            pred = JudgeByRubric()(task_description=input, metric_title=self.name, metric_description=self.description, rubric=rubric, input=input, reference=reference, output=output, lm=self.judge)

        # score.score is a string containing a number (with possible extra text)
        # Convert it to an integer using regex
        score = re.search(r'\d+', pred.score)
        if score is None:
            numeric = 0
        else:
            numeric = int(score.group())

        # Extract DSPy ChainOfThought reasoning (spec mandates .reasoning only)
        feedback = getattr(pred, 'reasoning', '')
        return MetricResult(score=float(numeric), feedback=feedback)

    def _calculate_impl(self, input, output, references=None, **kwargs):
        res = self._calculate_with_feedback_impl(input, output, references, **kwargs)
        return res.score

    def _calculate_batched_impl(self, inputs, outputs, references=None, num_workers=64, **kwargs):
        """
        Calculate scores using multi-threading, ensuring results are in the correct order.
        Each input/output pair is processed individually in parallel, and results are ordered correctly.
        Includes a tqdm progress bar for overall tracking.
        """

        # Use ThreadPoolExecutor to process each item in parallel
        results = [None] * len(inputs)  # Placeholder for results to maintain order
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.calculate, inputs[i], outputs[i], [references[i]] if references else None, **kwargs): i
                for i in range(len(inputs))
            }

            # Initialize tqdm progress bar
            with tqdm(total=len(futures), desc="Processing Items") as pbar:
                for future in as_completed(futures):
                    index = futures[future]  # Retrieve the original index
                    try:
                        results[index] = future.result()  # Store result in correct order
                    except Exception as e:
                        print(f"Error processing item {index}: {e}")
                    pbar.update(1)  # Update progress bar as each future completes

        return results

    def predict(self, dataset, update_dataset=True, num_workers=64, **kwargs):
        """
        Calculate the metric for the dataset
        """
        df = dataset.get_dataframe()
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()

        if not input_column:
            raise ValueError("Input column not found in dataset.  When constructing your Dataset please provide input_column.")
        if not output_column:
            raise ValueError("Output column not found in dataset.  When constructing your Dataset please provide output_column.")

        inputs = df[input_column].values.tolist()
        outputs = df[output_column].values.tolist()

        results_wrapped = self.calculate_batched_with_feedback(inputs, outputs, num_workers=num_workers)

        # If any of the results are NaN, replace them with 0
        results = [r.score if r and not math.isnan(r.score) else 0 for r in results_wrapped]

        if update_dataset:
            df[self.name] = results
            # Persist feedback column
            df[f"{self.name}__feedback"] = [r.feedback for r in results_wrapped]
            dataset.set_dataframe(df)

            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)

        return results
    
    def display_rubric(self, rubric, metric_title=None):
        """
        Display the rubric in a tabular format for Jupyter Notebooks with enhanced text visibility.
        
        Parameters:
        rubric (dict): A dictionary containing the rubric criteria and score descriptions.
                    Expected keys: 'criteria', 'score1_description', 'score2_description',
                    'score3_description', 'score4_description', 'score5_description'.
        metric_title (str, optional): Title of the metric to display above the rubric table.
        
        Returns:
        None: Displays the rubric as a table in Jupyter Notebook.
        """
        pd.set_option('display.max_colwidth', None)

        # Create a pandas DataFrame to hold the rubric
        rubric_df = pd.DataFrame({
            "Criteria": [rubric.get("criteria", "N/A")],
            "Score 1": [rubric.get("score1_description", "N/A")],
            "Score 2": [rubric.get("score2_description", "N/A")],
            "Score 3": [rubric.get("score3_description", "N/A")],
            "Score 4": [rubric.get("score4_description", "N/A")],
            "Score 5": [rubric.get("score5_description", "N/A")]
        })

        # Apply custom CSS to ensure proper text wrapping and visibility
        styled_rubric = rubric_df.style.set_table_styles(
            [{
                'selector': 'td',
                'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]
            }]
        )

        # Display the title if provided
        if metric_title:
            display(HTML(f"<h3 style='text-align: left;'>{metric_title}</h3>"))

        # Display the styled DataFrame
        display(styled_rubric)

    def display(self):
        """
        Display the metric rubric in a tabular format for Jupyter Notebooks.
        """
        self.display_rubric(self.rubric, metric_title=self.name)