import dspy
import os
import numpy as np
import pandas as pd
from autometrics.metrics.Metric import Metric
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math

G_EVAL_PROMPT = """You are evaluating the output of a model.  Here is the task description: {task_description}

Evaluation Criteria:

{evaluation_criteria} (Score 1 lowest - 5 highest)

Evaluation Steps:

{evaluation_steps}

Example:


Source Text:

{source}

References (if available):
{references}

Model Output:

{model_output}


Evaluation Form (scores ONLY):

- {evaluation_criteria_name}:"""

class GenerateEvaluationStepsSignature(dspy.Signature):
    """Given a task description and evaluation criteria please come up with evaluation steps that will lead to an accurate score.  Note that the task description is the prompt that was provided to the model."""
    task_description = dspy.InputField(desc="A description of the task that the model was trying to solve when it generated the text.  Could be left blank if not available.")
    evaluation_criteria = dspy.InputField(desc="The criteria that the model output should be evaluated on.")
    evaluation_steps = dspy.OutputField(desc="The steps that will be used to evaluate the model output. Please number the steps.")

class GenerateEvaluationSteps(dspy.Module):
    def __init__(self):
        self.generate_evaluation_steps = dspy.ChainOfThought(GenerateEvaluationStepsSignature)

    def forward(self, task_description, evaluation_criteria):
        evaluation_steps = self.generate_evaluation_steps(task_description=task_description, evaluation_criteria=evaluation_criteria).evaluation_steps
        return evaluation_steps

def find_score_token_index(logprobs_content, possible_scores=[1, 2, 3, 4, 5]):
    """
    Find the index of the last token that contains a score (1-5) by searching backwards.
    
    Args:
        logprobs_content: List of ChatCompletionTokenLogprob objects
        possible_scores: List of possible score values
        
    Returns:
        int: Index of the token containing the score, or None if not found
    """
    score_tokens = [str(score) for score in possible_scores]
    
    # Search backwards through the tokens
    for i in range(len(logprobs_content) - 1, -1, -1):
        token = logprobs_content[i].token
        # Check if this token is exactly one of our score tokens
        if token in score_tokens:
            return i
    
    return None

def extract_score_probabilities(logprobs_content, possible_scores=[1, 2, 3, 4, 5]):
    """
    Convert logprobs to probability distribution over possible scores.
    Finds the last token containing a score (1-5) and extracts probabilities from there.
    
    Args:
        logprobs_content: List of ChatCompletionTokenLogprob objects from the API response
        possible_scores: List of possible score values (default [1, 2, 3, 4, 5])
    
    Returns:
        dict: Mapping from score to probability, or None if no score token found
    """
    # Find the index of the last score token
    score_token_index = find_score_token_index(logprobs_content, possible_scores)
    
    if score_token_index is None:
        print("Warning: No score token (1-5) found in the model output!")
        return None
    
    # Get the top logprobs for the score token
    top_logprobs = logprobs_content[score_token_index].top_logprobs
    
    # Convert possible scores to strings since tokens are strings
    score_tokens = [str(score) for score in possible_scores]
    
    # Extract logprobs for score tokens, defaulting to very low probability for missing scores
    score_logprobs = {}
    for score_token in score_tokens:
        # Find the logprob for this score token
        found = False
        for top_logprob in top_logprobs:
            if top_logprob.token == score_token:
                score_logprobs[score_token] = top_logprob.logprob
                found = True
                break
        
        # If not found, assign a very low logprob (effectively zero probability)
        if not found:
            score_logprobs[score_token] = -20.0  # Very low logprob
    
    # Convert logprobs to probabilities using exp
    score_probs = {token: np.exp(logprob) for token, logprob in score_logprobs.items()}
    
    # Normalize to sum to 1 (softmax normalization)
    total_prob = sum(score_probs.values())
    normalized_probs = {token: prob / total_prob for token, prob in score_probs.items()}
    
    # Convert back to integer keys
    result = {int(token): prob for token, prob in normalized_probs.items()}
    
    return result

def GEval(formatted_prompt, source, references, model_output, dspy_lm):
    # Replace {user_query} placeholder with the actual source text if present
    formatted_prompt = formatted_prompt.replace("{user_query}", source)
    prompt = formatted_prompt.format(source=source, references=references, model_output=model_output)

    results = dspy_lm.forward(prompt=prompt, logprobs=True, top_logprobs=5)

    # Extract score probabilities using the smarter approach
    score_probs = extract_score_probabilities(results.choices[0].logprobs.content)

    weighted_score = 0
    
    if score_probs is not None:
        weighted_score = sum([score * prob for score, prob in score_probs.items()])
    else:
        print("Could not extract score probabilities - no valid score token found.")
    
    return weighted_score

class LLMJudgeGEval(Metric):
    def __init__(self, name, description, dataset, evaluation_criteria, model=None, task_description=None, 
                 evaluation_steps=None, auto_generate_steps=True, possible_scores=[1, 2, 3, 4, 5], 
                 criteria_generation_model=None):
        """
        Initialize LLMJudgeGEval metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric  
            dataset: Dataset to evaluate on
            evaluation_criteria: The criteria for evaluation
            model: DSPy language model for running the evaluation
            task_description: Description of the task being evaluated
            evaluation_steps: Pre-defined evaluation steps (optional)
            auto_generate_steps: Whether to auto-generate evaluation steps if not provided
            possible_scores: List of possible score values (default [1,2,3,4,5])
            criteria_generation_model: Separate model for generating evaluation criteria (optional, defaults to main model)
        """
        # Set default model if not provided
        if model is None:
            model = dspy.LM("litellm_proxy/meta-llama/Meta-Llama-3.1-8B-Instruct", 
                          api_base="http://future-hgx-1:7400/v1", api_key="None")
        
        # Set criteria generation model - defaults to main model if not provided
        if criteria_generation_model is None:
            criteria_generation_model = model
        
        # Pass parameters to parent constructor for caching
        super().__init__(
            name=name,
            description=description,
            evaluation_criteria=evaluation_criteria,
            model=model,
            criteria_generation_model=criteria_generation_model,
            task_description=task_description,
            evaluation_steps=evaluation_steps,
            auto_generate_steps=auto_generate_steps,
            possible_scores=possible_scores,
            dataset=dataset  # Will exclude this from cache
        )
        
        self.dataset = dataset
        self.evaluation_criteria = evaluation_criteria
        self.model = model  # Main evaluation model
        self.criteria_generation_model = criteria_generation_model  # Model for generating criteria
        self.task_description = task_description or "No task description provided"
        self.evaluation_steps = evaluation_steps
        self.auto_generate_steps = auto_generate_steps
        self.possible_scores = possible_scores
        
        # Generate evaluation steps if needed using the criteria generation model
        if self.evaluation_steps is None and self.auto_generate_steps:
            self._generate_evaluation_steps()
        elif self.evaluation_steps is None:
            self.evaluation_steps = "1. Read the source text and model output carefully.\n2. Compare against the evaluation criteria.\n3. Assign a score from 1-5."
            
        # Create the formatted prompt template
        self.formatted_prompt = G_EVAL_PROMPT.format(
            task_description=self.task_description,
            evaluation_criteria=self.evaluation_criteria,
            evaluation_steps=self.evaluation_steps,
            evaluation_criteria_name=self.evaluation_criteria.split(":")[0] if ":" in self.evaluation_criteria else self.evaluation_criteria,
            source="{source}",
            references="{references}",
            model_output="{model_output}"
        )
        
        # Exclude dataset from cache key as it doesn't affect computation
        self.exclude_from_cache_key('dataset')

    def _generate_evaluation_steps(self):
        """Generate evaluation steps using the criteria generation model if auto_generate_steps is True."""
        try:
            # Use the criteria generation model (which may be different from the evaluation model)
            with dspy.settings.context(lm=self.criteria_generation_model):
                generator = GenerateEvaluationSteps()
                self.evaluation_steps = generator.forward(
                    task_description=self.task_description,
                    evaluation_criteria=self.evaluation_criteria
                )
        except Exception as e:
            print(f"Warning: Could not auto-generate evaluation steps: {e}")
            self.evaluation_steps = "1. Read the source text and model output carefully.\n2. Compare against the evaluation criteria.\n3. Assign a score from 1-5."

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Calculate G-Eval score for a single input/output pair.
        
        Args:
            input: The source text/input
            output: The model output to evaluate
            references: Optional reference outputs
            **kwargs: Additional keyword arguments
            
        Returns:
            float: The weighted G-Eval score
        """
        # Prepare references string
        if references is None:
            references_str = "None provided"
        elif isinstance(references, list):
            references_str = "\n".join([f"Reference {i+1}: {ref}" for i, ref in enumerate(references)])
        else:
            references_str = str(references)
        
        # Use the main evaluation model (not the criteria generation model)
        score = GEval(
            formatted_prompt=self.formatted_prompt,
            source=input,
            references=references_str,
            model_output=output,
            dspy_lm=self.model  # Use main evaluation model
        )
        return score

    def _calculate_batched_impl(self, inputs, outputs, references=None, num_workers=64, **kwargs):
        """
        Calculate G-Eval scores for a batch of inputs/outputs using parallel processing.
        
        Args:
            inputs: List of input texts
            outputs: List of model outputs
            references: List of reference outputs (optional)
            num_workers: Number of worker threads
            **kwargs: Additional keyword arguments
            
        Returns:
            list: List of G-Eval scores
        """
        # Prepare references
        if references is None:
            references = [None] * len(inputs)
        
        # Use ThreadPoolExecutor for parallel processing
        results = [None] * len(inputs)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.calculate, inputs[i], outputs[i], 
                              [references[i]] if references[i] is not None else None, **kwargs): i
                for i in range(len(inputs))
            }
            
            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Processing G-Eval") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    pbar.update(1)
        return results

    def predict(self, dataset, update_dataset=True, num_workers=64, **kwargs):
        """
        Calculate G-Eval scores for the entire dataset.
        
        Args:
            dataset: Dataset to evaluate
            update_dataset: Whether to update the dataset with results
            num_workers: Number of worker threads
            **kwargs: Additional keyword arguments
            
        Returns:
            list: List of G-Eval scores
        """
        df = dataset.get_dataframe()
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()
        reference_columns = dataset.get_reference_columns()
        
        if not input_column:
            raise ValueError("Input column not found in dataset. Please provide input_column when constructing your Dataset.")
        if not output_column:
            raise ValueError("Output column not found in dataset. Please provide output_column when constructing your Dataset.")
        
        inputs = df[input_column].values.tolist()
        outputs = df[output_column].values.tolist()
        
        # Handle references
        references = None
        if reference_columns:
            references = []
            for _, row in df.iterrows():
                ref_list = [row[col] for col in reference_columns if pd.notna(row[col])]
                references.append(ref_list if ref_list else None)
        
        # Calculate scores
        results = self.calculate_batched(inputs, outputs, references, num_workers=num_workers, **kwargs)
        
        # Handle NaN values
        results = [result if result and not math.isnan(result) else 0.0 for result in results]
        
        # Update dataset if requested
        if update_dataset:
            df[self.name] = results
            dataset.set_dataframe(df)
            
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)
        
        return results

if __name__ == "__main__":
    # dspy_lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    # dspy_lm = dspy.LM("litellm_proxy/qwen/Qwen3-8B", api_base="http://future-hgx-1:7420/v1", api_key="None")
    dspy_lm = dspy.LM("litellm_proxy/meta-llama/Meta-Llama-3.1-8B-Instruct", api_base="http://future-hgx-1:7400/v1", api_key="None")

    dspy.configure(lm=dspy_lm)

    task_description = """Given a complicated original sentence, simplify it in a way such that a broader audience could easily understand it."""
    evaluation_criteria = "simplicity"
    evaluation_criteria_name = "simplicity"

    # Generate evaluation steps
    generate_evaluation_steps = GenerateEvaluationSteps()
    evaluation_steps = generate_evaluation_steps.forward(task_description=task_description, evaluation_criteria=evaluation_criteria)

    source = "a bastion on the eastern approaches was built later."
    references = "['a fort on the eastern access road was built later.', 'a fortification on the eastern approaches was built later.', 'a support on the east was built later.', 'an extension on the eastern side was built later.', 'a fort on the eastern side of the area was built later.', 'a wall on the east side was built later.', 'later, it was fortified on the easterrn side.', 'a  defense on the eastern side was built later.', 'a bastion on the eastern side was built later.', 'an eastern bastion was built later.']"
    model_output = "a bastion in the east nears was built at a later date."

    formatted_prompt = G_EVAL_PROMPT.format(task_description=task_description, evaluation_criteria=evaluation_criteria, evaluation_steps=evaluation_steps, evaluation_criteria_name=evaluation_criteria_name, source="{source}", references="{references}", model_output="{model_output}")

    score = GEval(formatted_prompt, source, references, model_output, dspy_lm)
    print(f"Score: {score}")