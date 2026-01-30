from autometrics.metrics.Metric import Metric

from prometheus_eval.litellm import LiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import math
import pandas as pd
from IPython.display import display, HTML

class LLMJudgeRubric(Metric):
    def __init__(self, name, description, dataset, rubric, model=None, judge=None, task_description=None, judge_api_base="http://jagupard37:8000/v1"):
        # Initialize default model if not provided
        if model is None:
            model = LiteLLM('openai/prometheus-eval/prometheus-7b-v2.0', api_base=judge_api_base, api_key='None')
        
        # Pass all important parameters to parent constructor for caching
        super().__init__(
            name=name, 
            description=description,
            rubric=rubric,
            model=model if model else None,
            task_description=task_description,
            judge_api_base=judge_api_base, # We'll exclude this later
            dataset=dataset  # We'll exclude this later
        )
        
        self.dataset = dataset
        self.rubric = rubric
        self.model = model
        self.judge = judge if judge else PrometheusEval(model=self.model, absolute_grade_template=ABSOLUTE_PROMPT)
        self.task_description = task_description
        
        # Exclude non-affecting parameters from cache key
        self.exclude_from_cache_key('dataset', 'judge_api_base')

    def _calculate_impl(self, input, output, references=None, **kwargs):
        if self.task_description:
            input = self.task_description + "\n\n" + input

        reference = None
        if references is not None:
            reference = references[0] # NOTE: Only one reference is supported

        rubric = SCORE_RUBRIC_TEMPLATE.format(**self.rubric)
        
        feedback, score = self.judge.single_absolute_grade(
            instruction=input,
            response=output,
            rubric=rubric,
            reference_answer=reference,
            params={"use_tqdm": False}
        )

        return score

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

        results = self.calculate_batched(inputs, outputs, num_workers=num_workers)

        # If any of the results are NaN, replace them with 0
        results = [result if result and not math.isnan(result) else 0 for result in results]

        if update_dataset:
            df[self.name] = results
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