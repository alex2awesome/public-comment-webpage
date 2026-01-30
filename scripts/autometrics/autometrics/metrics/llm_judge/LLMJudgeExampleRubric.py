from autometrics.metrics.Metric import Metric
import dspy
from autometrics.util.format import get_default_formatter
from dspy.teleprompt import MIPROv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math
from autometrics.util.normalize import find_distinct_quintiles_with_min_max, map_to_bucket
from dspy.evaluate import Evaluate
import random

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

def prepare_dataset_bucketted(dataset, target_column, task_description, metric_name, formatter, suggested_range=(1,5)):
    buckets = [[] for _ in range(5)]
    dspy_dataset = []

    # Find the quintile ranges for the target column
    quintiles = find_distinct_quintiles_with_min_max(dataset, target_column)

    for i, row in dataset.get_dataframe().iterrows():
        bucket_idx = map_to_bucket(row[target_column], quintiles)
        buckets[bucket_idx].append(
            {
                'text': formatter(row),
                'task_description': task_description,
                'metric': metric_name,
                'suggested_range': suggested_range,
                'score': row[target_column]
            }
        )

        dspy_dataset.append(
            dspy.Example(
                text=formatter(row),
                task_description=task_description,
                metric=metric_name,
                suggested_range=suggested_range,
                score=row[target_column]
            ).with_inputs('text', 'task_description', 'metric', 'suggested_range')
        )

    return buckets, dspy_dataset

def grade_row(row, axis, llm, formatter, task_description, program, suggested_range=(1,5)):
    '''Helper function to grade a single row'''
    with dspy.settings.context(lm=llm):
        return program(formatter(row), axis, suggested_range=suggested_range,task_description=task_description).score

class LLMAsAJudgeSignature(dspy.Signature):
    """Given an input text, the task description that the model was trying to follow, and a metric to rate the text on, return a score on this metric."""
    text = dspy.InputField(desc="The input text that we want to rate.")
    task_description = dspy.InputField(desc="A description of the task that the model was trying to solve when it generated the text.  Could be left blank if not available.")
    metric = dspy.InputField(desc="The metric that we want to rate the text on.")
    suggested_range = dspy.InputField(desc="The suggested range of possible values for the metric.")
    score = dspy.OutputField(desc="The score that the text should recieve on this metric.")

class LLMAsAJudge(dspy.Module):
    def __init__(self):
        super(LLMAsAJudge, self).__init__()
        self.generate_score = dspy.ChainOfThought(LLMAsAJudgeSignature)

    def forward(self, text, metric, suggested_range=(1,5), task_description=None):
        if task_description is None:
            task_description = "None"
        suggested_range_str = f"{suggested_range[0]} to {suggested_range[1]}"
        score = self.generate_score(task_description=task_description, text=text, metric=metric, suggested_range=suggested_range_str).score
        # Convert the string score to a float by stripping any additional text and converting to a float
        if '\n' in score:
            score = score.split('\n')[0]
        try:
            score = float(score.strip())
        except:
            score = 0.0

        return dspy.Prediction(text=text, metric=metric, score=score)

class LLMJudgeExampleRubric(Metric):
    # TODO: Better output prompt path
    def __init__(self, name, description, model, train_dataset, formatter=None, task_description=None, target_column=None, eval_function_name='inverse_distance', custom_eval_function=None, load_prompt=None, optimize=True, output_prompt_path='output_prompt.dspy', metric_name=None, attempts=5, examples_per_range=2, seed=42):
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
        self.suggested_range = (self.dataset.get_dataframe()[self.target_column].min().item(), self.dataset.get_dataframe()[self.target_column].max().item())
        self.attempts = attempts
        self.examples_per_range = examples_per_range

        self.program = LLMAsAJudge()

        random.seed(seed)

        if load_prompt is not None:
            self.program.load(load_prompt)

        if optimize:

            train_buckets, trainset = prepare_dataset_bucketted(
                train_dataset,
                target_column,
                task_description,
                self.metric_name,
                self.formatter,
                self.suggested_range
            )

            evaluate = Evaluate(devset=trainset, metric=get_wrapped_metric(self.eval_function), num_threads=64, display_progress=True, display_table=False)

            demosets = [[] for _ in range(self.attempts)]
            for i in range(self.attempts):
                for bucket in train_buckets:
                    sample = random.sample(bucket, min(self.examples_per_range, len(bucket)))
                    demosets[i].extend(sample)

            best_score = -100000
            for i, demoset in enumerate(demosets):
                new_program = self.program.deepcopy()
                new_program.generate_score._predict.demos = demoset
                score = evaluate(new_program)
                if score > best_score:
                    best_score = score
                    self.program = new_program
                    print(f"New best score: {best_score}")

            if output_prompt_path is not None:
                self.program.save(output_prompt_path)


        super().__init__(name, description, model=model, train_dataset=train_dataset, formatter=formatter, task_description=task_description, target_column=target_column, eval_function_name=eval_function_name, custom_eval_function=custom_eval_function, load_prompt=load_prompt, optimize=optimize, output_prompt_path=output_prompt_path, metric_name=metric_name, attempts=attempts, examples_per_range=examples_per_range, seed=seed)

    def _calculate_impl(self, input, output, references=None, **kwargs):
        row = {self.dataset.get_input_column(): input, self.dataset.get_output_column(): output}
        if references is not None:
            for i, ref in enumerate(references):
                row[self.dataset.get_reference_columns()[i]] = ref

        grade_row(row, self.metric_name, self.model, self.formatter, self.task_description, self.program, suggested_range=self.suggested_range)

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
                futures = {executor.submit(grade_row, row, self.metric_name, self.model, self.formatter, self.task_description, self.program, self.suggested_range): index for index, row in df.iterrows()}

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