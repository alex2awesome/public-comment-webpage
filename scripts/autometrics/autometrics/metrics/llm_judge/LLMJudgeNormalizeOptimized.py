from autometrics.metrics.Metric import Metric
import dspy
from autometrics.util.format import get_default_formatter
from dspy.teleprompt import MIPROv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math
import pandas as pd
from autometrics.util.normalize import find_distinct_quintiles_with_min_max, map_to_bucket

def exact_match_rounded(x, y):
    return int(round(x) == round(y))

def inverse_distance(x, y):
    if x == y:
        return 1

    return 1 / (abs(x - y) + 1)

def get_wrapped_metric(metric_func):
    def wrapped_metric(example, pred, trace=None):
        return metric_func(example.score, pred.score)
    return wrapped_metric

def prepare_dataset(dataset, target_column, task_description, metric_name, formatter):
    dspy_dataset = []

    # Find the quintile ranges for the target column
    quintiles = find_distinct_quintiles_with_min_max(dataset, target_column)

    for i, row in dataset.get_dataframe().iterrows():
        dspy_dataset.append(
            dspy.Example(
                text=formatter(row),
                task_description=task_description,
                metric=metric_name,
                score=map_to_bucket(row[target_column], quintiles) + 1
            ).with_inputs('text', 'task_description', 'metric')
        )

    return dspy_dataset

def grade_row(row, axis, llm, formatter, task_description, program):
    '''Helper function to grade a single row'''
    with dspy.settings.context(lm=llm):
        return program(formatter(row), axis,task_description=task_description).score

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

    def forward(self, text, metric, task_description=None):
        if task_description is None:
            task_description = "None"
        score = self.generate_score(task_description=task_description, text=text, metric=metric).score
        # Convert the string score to a float by stripping any additional text and converting to a float
        if '\n' in score:
            score = score.split('\n')[0]
        try:
            score = float(score.strip())
        except:
            score = 0.0

        return dspy.Prediction(text=text, metric=metric, score=score)

class LLMJudgeNormalizeOptimized(Metric):
    # TODO: Better output prompt path
    def __init__(self, name, description, model, train_dataset, formatter=None, task_description=None, target_column=None, eval_function_name='inverse_distance', custom_eval_function=None, load_prompt=None, optimize=True, output_prompt_path='output_prompt.dspy', metric_name=None):
        self.eval_function = None
        if custom_eval_function is not None:
            self.eval_function = custom_eval_function
        elif eval_function_name == 'exact_match_rounded':
            self.eval_function = exact_match_rounded
        elif eval_function_name == 'inverse_distance':
            self.eval_function = inverse_distance

        self.model = model
        if formatter is None:
            self.formatter = get_default_formatter(train_dataset)
        else:
            self.formatter = formatter
        self.task_description = task_description
        self.dataset = train_dataset
        self.target_column = target_column
        self.metric_name = metric_name if metric_name is not None else target_column

        self.program = LLMAsAJudge()

        if load_prompt is not None:
            self.program.load(load_prompt)

        if optimize:

            train_set = prepare_dataset(
                train_dataset,
                target_column,
                task_description,
                self.metric_name,
                self.formatter,
            )

            teleprompter = MIPROv2(
                metric=get_wrapped_metric(self.eval_function),
                auto="medium",
                num_threads=64,
            )

            optimized_program = teleprompter.compile(
                self.program.deepcopy(),
                trainset=train_set,
                max_bootstrapped_demos=8,
                max_labeled_demos=8,
                requires_permission_to_run=False,
            )

            if output_prompt_path is not None:
                optimized_program.save(output_prompt_path)

            self.program = optimized_program


        super().__init__(name, description, model=model, train_dataset=train_dataset, formatter=formatter, task_description=task_description, target_column=target_column, eval_function_name=eval_function_name, custom_eval_function=custom_eval_function, load_prompt=load_prompt, optimize=optimize, output_prompt_path=output_prompt_path, metric_name=metric_name, attempts=attempts, examples_per_range=examples_per_range, seed=seed)

    def _calculate_impl(self, input, output, references=None, **kwargs):
        row = {self.dataset.get_input_column(): input, self.dataset.get_output_column(): output}
        if references is not None:
            for i, ref in enumerate(references):
                row[self.dataset.get_reference_columns()[i]] = ref

        grade_row(row, self.metric_name, self.model, self.formatter, self.task_description, self.program)

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
    
    def predict(self, dataset, update_dataset=True, max_workers=64, metric_name=None, **kwargs):
            '''
                Grade the dataframe using the LLM judge in parallel with progress bar
            '''
            if metric_name is None:
                metric_name = self.metric_name + "_" + self.model.model.split("/")[-1]

            df = dataset.get_dataframe()

            results = []

            # Create a ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks to the executor
                futures = {executor.submit(grade_row, row, self.metric_name, self.model, self.formatter, self.task_description, self.program): index for index, row in df.iterrows()}

                # Collect the results with tqdm progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Grading rows", unit="row"):
                    index = futures[future]
                    try:
                        score = future.result()
                        if update_dataset:
                            df.at[index, metric_name] = score
                        results.append((index, score))
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        score = 0.0
                        if update_dataset:
                            df.at[index, metric_name] = score
                        results.append((index, score))

            if update_dataset:
                dataset.set_dataframe(df)
                if metric_name not in dataset.get_metric_columns():
                    dataset.get_metric_columns().append(metric_name)

            results.sort(key=lambda x: x[0])

            return [score if score is not None and not math.isnan(score) else 0 for _, score in results]