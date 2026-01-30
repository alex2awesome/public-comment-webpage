import dspy
from typing import List, Type, Optional
import math
import litellm

from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender


class LLMMetricRecommendationSignature(dspy.Signature):
    """I am looking for a metric to evaluate the attached task. In particular I care about the specific target measurement that I attached.

Please help me decide from among the metrics that I have attached documentation for which one is most relevant to the task and target.

Please provide a ranking of the metrics from most relevant to least relevant for the task and target above.
You can reason first about what makes a metric relevant for the task and target, and then provide your ranking.

IMPORTANT: The final ranking should be a list of EXACT metric class names (no hyphens, no spaces, no extra words).  Use the METRIC NAME not what it is called in the documentation.
For example, use "SelfBLEU" not "Self-BLEU", use "BERTScore" not "BERT Score", use "BLEU" not "BLEU Score".

The final ranking should just be a list of metric names, in order from most relevant to least relevant.
The list should be exactly `num_metrics_to_recommend` items long."""
    task_description: str = dspy.InputField(desc="A description of the task that an LLM performed and that I now want to evaluate.")
    target: str = dspy.InputField(desc="The specific target measurement that I want to evaluate about the task.")
    metric_documentation: List[str] = dspy.InputField(desc="A list of metric names and their documentation.  The documentation will contain the metric name, as well as many details about the metric.")
    num_metrics_to_recommend: int = dspy.InputField(desc="The number of metrics to recommend.  It is imperative to target this number or very very close to it.  We will do more extensive filtering later.")
    ranking: List[str] = dspy.OutputField(desc="A numbered list of EXACT metric class names (no hyphens, no spaces, no extra words), in order from most relevant to least relevant. The list should be of length `num_metrics_to_recommend`.  You should write the number in front of the metric name (e.g '1. METRIC1_NAME', '2. METRIC2_NAME', etc.).  REMEMBER: Put quotes around EACH number + metric name pair, not just one set of quotes for the full string.  IMPORTANT: Refer to «METRIC NAME: ...» for the exact name of the metric or it won't be a match.")

class LLMMetricRecommendation(dspy.Module):
    """
    A module that recommends metrics for a given task and target.
    """
    def __init__(self, model: dspy.LM = None):
        super().__init__()
        if model is None:
            self.recommender = dspy.ChainOfThought(LLMMetricRecommendationSignature, max_tokens=4096)
        elif getattr(model, 'model', None) is not None and "gpt-5" in getattr(model, 'model', None):
            self.recommender = dspy.ChainOfThought(LLMMetricRecommendationSignature)
        else:
            self.recommender = dspy.ChainOfThought(LLMMetricRecommendationSignature, max_tokens=4096)

    def forward(self, task_description: str, target: str, num_metrics_to_recommend: int, metric_documentation: List[str]) -> List[str]:
        results = self.recommender(task_description=task_description, target=target, num_metrics_to_recommend=num_metrics_to_recommend, metric_documentation=metric_documentation)

        # Normalize ranking entries:
        # 1) Remove common numeric prefixes like "1.", "1)", "1 -", "1:"
        # 2) Strip stray quotes/backticks and trailing punctuation (handles cases like: MetricName')
        cleaned_ranking = []
        for rank in results.ranking:
            text = str(rank).strip()
            try:
                import re
                # Remove a leading optional quote, digits, optional spaces and a punctuation separator
                # Examples handled: "1. Name", "1) Name", "1 - Name", "1: Name"
                text = re.sub(r"^\s*['\"`]?\s*\d+\s*[\.)\-:]\s*", "", text)
            except Exception:
                # Fallback to previous simple split behavior
                if "." in text:
                    parts = text.split(".", 1)
                    if parts[0].strip().isdigit():
                        text = parts[1].strip()
            # Strip wrapping quotes/backticks and common trailing punctuation
            text = text.strip(" `\'\"“”‘’.,;:-")
            cleaned_ranking.append(text)

        results.ranking = cleaned_ranking
        
        return results.ranking
    
class LLMRec(MetricRecommender):
    """
    A metric recommender that uses a LLM to recommend metrics.
    Uses systematic token counting to plan batching strategy upfront.
    """
    def __init__(self, metric_classes: List[Type[Metric]], index_path: str = None, force_reindex: bool = False, model: dspy.LM = None, use_description_only: bool = False):
        super().__init__(metric_classes, index_path, force_reindex)
        self.model = model
        self.recommender = LLMMetricRecommendation(model=model)
        self.use_description_only = use_description_only
        # If upstream proxy reports a smaller effective input limit, cache it here
        self._override_max_input_tokens: Optional[int] = None
        
        # Token budget constants - being more conservative based on actual failures
        self.DSPY_OVERHEAD_TOKENS = 4096  # Reserve for DSPy system prompts, reasoning, etc. (increased)
        self.OUTPUT_TOKENS = 4096  # Reserve for model output
        self.SAFETY_MARGIN = 2000   # Additional safety margin (increased significantly)
        
    def _get_model_name(self) -> str:
        """Extract model name from dspy.LM model for litellm functions."""
        if self.model is None:
            # Fall back to current global dspy model
            try:
                current_lm = dspy.settings.lm
                if current_lm is not None:
                    return getattr(current_lm, 'model', 'gpt-3.5-turbo')
            except:
                pass
            return 'gpt-3.5-turbo'  # Safe default
        
        return getattr(self.model, 'model', 'gpt-3.5-turbo')

    def _count_tokens(self, text: str, model_name: str = None) -> int:
        """Count tokens in text using litellm.token_counter."""
        if model_name is None:
            model_name = self._get_model_name()
        
        try:
            # Use litellm to count tokens
            return litellm.token_counter(model=model_name, text=text)
        except Exception as e:
            print(f"Warning: litellm.token_counter failed with {e}, using fallback estimation")
            # Fallback: rough estimation (1.3 tokens per word on average)
            return int(len(text.split()) * 1.3)
    
    def _get_max_context_tokens(self, model_name: str = None) -> int:
        """Get max context window for model, preferring input tokens over general max."""
        # Honor detected effective limit from previous errors if present
        if self._override_max_input_tokens is not None:
            return self._override_max_input_tokens
        if model_name is None:
            model_name = self._get_model_name()
            
        try:
            # First try to get detailed model cost info which may have input/output breakdown
            model_cost_info = litellm.model_cost.get(model_name)
            if model_cost_info is None:
                model_name = model_name.split("/")[-1]
                model_cost_info = litellm.model_cost.get(model_name)
            
            if model_cost_info:
                # Prefer max_input_tokens if available (for models like GPT-4o-mini)
                if 'max_input_tokens' in model_cost_info:
                    max_input = model_cost_info['max_input_tokens']
                    print(f"Using max_input_tokens for {model_name}: {max_input}")
                    return max_input
                elif 'max_tokens' in model_cost_info:
                    max_general = model_cost_info['max_tokens']
                    print(f"Using max_tokens for {model_name}: {max_general}")
                    return max_general
            
            # Fall back to litellm.get_max_tokens
            max_tokens = litellm.get_max_tokens(model=model_name)
            if max_tokens is not None:
                return max_tokens
                
            # Default to Qwen3's context length (40960) as a reasonable modern default
            print(f"Warning: Could not get max tokens for {model_name}, using default 40960 (Qwen3 context)")
            return 40960
            
        except Exception as e:
            print(f"Warning: litellm token limit lookup failed with {e}, using default 40960")
            return 40960

    def _compute_available_tokens(self, max_context: int) -> int:
        """Compute available prompt tokens with dynamic budgeting for small contexts."""
        # Adapt reserves for small contexts (e.g., 8k) to avoid negative budgets
        dynamic_dspy_overhead = min(self.DSPY_OVERHEAD_TOKENS, int(max_context * 0.20))
        dynamic_output_tokens = min(self.OUTPUT_TOKENS, int(max_context * 0.20))
        dynamic_safety_margin = min(self.SAFETY_MARGIN, int(max_context * 0.10))
        available_tokens = max_context - dynamic_dspy_overhead - dynamic_output_tokens - dynamic_safety_margin
        # Ensure a minimal usable budget
        return max(512, available_tokens)

    def _is_context_error(self, e: Exception) -> bool:
        msg = str(e)
        return any(p in msg for p in [
            'ContextWindowExceededError',
            'BadRequestError',
            'context length',
            'maximum context length',
            'maximum allowed length',
            'exceeds the maximum',
            'Input length',
            '--allow-auto-truncate'
        ])

    def _extract_max_allowed_from_error(self, e: Exception) -> Optional[int]:
        """Extract effective max input tokens from upstream proxy error messages."""
        import re
        msg = str(e)
        m = re.search(r"maximum allowed length \((\d+)\)", msg)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        m2 = re.search(r"maximum context length\s*:?\s*(\d+)", msg)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                return None
        return None

    def _handle_context_error(self, e: Exception) -> None:
        inferred = self._extract_max_allowed_from_error(e)
        if inferred is not None:
            if self._override_max_input_tokens != inferred:
                print(f"Detected effective max input tokens from error: {inferred}")
            self._override_max_input_tokens = inferred
    
    def _estimate_prompt_tokens(self, task_description: str, target_measurement: str, num_metrics: int, avg_metric_doc_tokens: int) -> int:
        """Estimate total prompt tokens for a given batch size."""
        # Create a realistic prompt template to get better estimates
        system_prompt = """Your input fields are:
1. `task_description` (str): A description of the task that an LLM performed and that I now want to evaluate.
2. `target` (str): The specific target measurement that I want to evaluate about the task.
3. `metric_documentation` (list[str]): A list of metric names and their documentation.
4. `num_metrics_to_recommend` (int): The number of metrics to recommend.

Your output fields are:
1. `reasoning` (str)
2. `ranking` (list[str]): A numbered list of EXACT metric class names

I am looking for a metric to evaluate the attached task. In particular I care about the specific target measurement that I attached.
Please help me decide from among the metrics that I have attached documentation for which one is most relevant to the task and target.
Please provide a ranking of the metrics from most relevant to least relevant for the task and target above."""

        user_prompt = f"""[[ ## task_description ## ]]
{task_description}

[[ ## target ## ]]
{target_measurement}

[[ ## metric_documentation ## ]]
[METRIC_DOCS_PLACEHOLDER]

[[ ## num_metrics_to_recommend ## ]]
{num_metrics}"""
        
        # Count tokens for system + user prompt structure
        base_tokens = self._count_tokens(system_prompt + user_prompt)
        
        # Add metric documentation tokens (with significant overhead for DSPy formatting)
        metric_docs_tokens = num_metrics * avg_metric_doc_tokens * 1.3  # 30% overhead for DSPy formatting
        
        total_estimated = base_tokens + metric_docs_tokens
        print(f"    Token estimation: base={base_tokens}, docs={metric_docs_tokens:.0f}, total={total_estimated:.0f}")
        
        return int(total_estimated)
    
    def _calculate_avg_metric_doc_tokens(self, metric_classes: List[Type[Metric]]) -> int:
        """Calculate average tokens per metric documentation."""
        if not metric_classes:
            return 100  # Safe default
            
        # Sample a few metrics to estimate average doc size
        sample_size = min(5, len(metric_classes))
        sample_metrics = metric_classes[:sample_size]
        
        total_tokens = 0
        for metric in sample_metrics:
            doc_text = f"«METRIC NAME: {metric.__name__}\nMETRIC DOCUMENTATION: {metric.__doc__}»"
            total_tokens += self._count_tokens(doc_text)
        
        avg_tokens = total_tokens // sample_size if sample_size > 0 else 100
        print(f"Estimated average tokens per metric documentation: {avg_tokens}")
        return avg_tokens

    def _plan_batching_strategy(self, metric_classes: List[Type[Metric]], dataset: Dataset, target_measurement: str) -> dict:
        """Plan the batching strategy using token counting."""
        model_name = self._get_model_name()
        max_context = self._get_max_context_tokens(model_name)
        
        # Calculate available tokens for prompt
        available_tokens = max_context - self.DSPY_OVERHEAD_TOKENS - self.OUTPUT_TOKENS - self.SAFETY_MARGIN
        
        print(f"Model: {model_name}")
        print(f"Max context: {max_context}")
        # Allow dynamic budgeting (honors small contexts like 8k from proxies)
        available_tokens = self._compute_available_tokens(max_context)
        print(f"Available for prompt: {available_tokens}")
        
        # Get task description and estimate base prompt tokens
        task_description = dataset.get_task_description()
        avg_metric_doc_tokens = self._calculate_avg_metric_doc_tokens(metric_classes)
        
        # Binary search to find maximum metrics per batch
        left, right = 1, len(metric_classes)
        max_metrics_per_batch = 1
        
        print(f"Binary search for max batch size (available: {available_tokens} tokens):")
        while left <= right:
            mid = (left + right) // 2
            estimated_tokens = self._estimate_prompt_tokens(
                task_description, target_measurement, mid, avg_metric_doc_tokens
            )
            
            if estimated_tokens <= available_tokens:
                max_metrics_per_batch = mid
                print(f"  {mid} metrics OK ({estimated_tokens} tokens)")
                left = mid + 1
            else:
                print(f"  {mid} metrics TOO BIG ({estimated_tokens} tokens)")
                right = mid - 1
        
        print(f"Final max metrics per batch: {max_metrics_per_batch}")
        
        # Calculate number of batches needed
        num_batches = math.ceil(len(metric_classes) / max_metrics_per_batch)
        
        # Ensure no batch returns more than 75% of its contents
        if max_metrics_per_batch > 3:  # Only apply this rule if we have room
            max_recommend_per_batch = int(max_metrics_per_batch * 0.75)
        else:
            max_recommend_per_batch = max_metrics_per_batch
        
        return {
            'max_metrics_per_batch': max_metrics_per_batch,
            'num_batches': num_batches,
            'max_recommend_per_batch': max_recommend_per_batch,
            'total_available_tokens': available_tokens,
            'avg_metric_doc_tokens': avg_metric_doc_tokens
        }

    def _recommend_batch(self, metric_classes: List[Type[Metric]], dataset: Dataset, target_measurement: str, k: int) -> List[str]:
        """
        Helper method to recommend metrics from a batch of metric classes.
        Returns the raw ranking (list of metric names) rather than metric classes.
        """
        task_description = dataset.get_task_description()
        docs = []
        for metric in metric_classes:
            if self.use_description_only:
                desc = getattr(metric, 'description', None)
                if desc is None:
                    desc = metric.__doc__
                docs.append(f"«METRIC NAME: {metric.__name__}\nMETRIC DOCUMENTATION: {desc}»")
            else:
                docs.append(f"«METRIC NAME: {metric.__name__}\nMETRIC DOCUMENTATION: {metric.__doc__}»")
        metric_documentation = docs

        # Never request more than we provide to avoid hallucinated extras
        k = min(k, len(metric_classes))
        print(f"  _recommend_batch: {len(metric_classes)} metrics -> requesting {k} recommendations")
        
        if self.model is not None:
            with dspy.settings.context(lm=self.model):
                ranking = self.recommender(task_description, target_measurement, k, metric_documentation)
        else:
            ranking = self.recommender(task_description, target_measurement, k, metric_documentation)
        
        print(f"  _recommend_batch: LLM returned {len(ranking)} recommendations")
        return ranking

    def _process_split_batch(self, batch: List[Type[Metric]], dataset: Dataset, target_measurement: str, batch_target: int) -> List[Type[Metric]]:
        """Recursively split a batch until it fits; collect up to batch_target results."""
        if batch_target <= 0 or not batch:
            return []
        try:
            ranking = self._recommend_batch(batch, dataset, target_measurement, batch_target)
            results = [self.metric_name_to_class(name) for name in ranking]
            results = [r for r in results if r is not None]
            return results[:batch_target]
        except Exception as e:
            if self._is_context_error(e):
                print(f"  Batch of size {len(batch)} failed with context error; splitting. Error: {e}")
                self._handle_context_error(e)
                if len(batch) == 1:
                    # Try with k=1; if still fails, give up on this item
                    try:
                        ranking = self._recommend_batch(batch, dataset, target_measurement, 1)
                        results = [self.metric_name_to_class(name) for name in ranking]
                        return [r for r in results if r is not None][:1]
                    except Exception:
                        return []
                mid = max(1, len(batch) // 2)
                left, right = batch[:mid], batch[mid:]
                # Proportionally distribute targets; ensure non-negative and not exceeding sub-batch sizes
                left_target = min(max(1, int(round(batch_target * len(left) / len(batch)))), len(left)) if batch_target > 1 else 1
                right_target = max(0, batch_target - left_target)
                right_target = min(right_target, len(right))
                left_results = self._process_split_batch(left, dataset, target_measurement, left_target)
                remaining = max(0, batch_target - len(left_results))
                right_results = self._process_split_batch(right, dataset, target_measurement, remaining)
                return (left_results + right_results)[:batch_target]
            else:
                raise e

    def _split_metrics_into_batches(self, metric_classes: List[Type[Metric]], max_per_batch: int) -> List[List[Type[Metric]]]:
        """Split metric classes into batches with maximum size.""" 
        batches = []
        for i in range(0, len(metric_classes), max_per_batch):
            batch = metric_classes[i:i + max_per_batch]
            batches.append(batch)
        return batches

    def _recommend_with_token_planning(self, metric_classes: List[Type[Metric]], dataset: Dataset, target_measurement: str, k: int) -> List[Type[Metric]]:
        """
        Recommend metrics using systematic token counting and batch planning.
        """
        # Filter out None metrics
        metric_classes = [metric for metric in metric_classes if metric is not None]
        
        if not metric_classes:
            return []
        
        print(f"Planning recommendation strategy for {len(metric_classes)} metrics, k={k}")
        
        # Plan batching strategy
        strategy = self._plan_batching_strategy(metric_classes, dataset, target_measurement)
        
        max_per_batch = strategy['max_metrics_per_batch']
        max_recommend_per_batch = strategy['max_recommend_per_batch']
        
        # If all metrics fit in one batch, recommend directly
        if len(metric_classes) <= max_per_batch:
            print("All metrics fit in one batch, recommending directly")
            try:
                ranking = self._recommend_batch(metric_classes, dataset, target_measurement, min(k, len(metric_classes)))
                print(f"Raw ranking received: {ranking}")
                
                results = [self.metric_name_to_class(metric_name) for metric_name in ranking]
                print(f"After metric_name_to_class: {len([r for r in results if r is not None])} valid conversions")
                
                results = [r for r in results if r is not None]
                results = list(dict.fromkeys(results))  # Remove duplicates
                print(f"After filtering and deduplication: {len(results)} final metrics")
                
                final_results = results[:k]
                print(f"Returning {len(final_results)} metrics (requested {k})")
                return final_results
            except Exception as e:
                if self._is_context_error(e):
                    print(f"Context window exceeded even with token planning. Error: {e}")
                    self._handle_context_error(e)
                    # Recursively split until it fits
                    return self._process_split_batch(metric_classes, dataset, target_measurement, min(k, len(metric_classes)))
                raise e
        
        # Multiple batches needed
        batches = self._split_metrics_into_batches(metric_classes, max_per_batch)
        
        # Calculate how many to recommend from each batch
        recommendation_ratio = k / len(metric_classes)
        
        print(f"Using {len(batches)} batches, recommendation ratio: {recommendation_ratio:.3f}")
        
        all_batch_results = []
        
        for i, batch in enumerate(batches):
            # Calculate target recommendations for this batch
            batch_target = max(1, min(int(len(batch) * recommendation_ratio), max_recommend_per_batch))
            batch_target = min(batch_target, len(batch))
            print(f"Processing batch {i+1}/{len(batches)}: {len(batch)} metrics -> {batch_target} recommendations")
            results = self._process_split_batch(batch, dataset, target_measurement, batch_target)
            if results:
                all_batch_results.extend(results)
        
        # Remove duplicates while preserving order
        seen = set()
        merged_results = []
        for metric in all_batch_results:
            if metric not in seen:
                merged_results.append(metric)
                seen.add(metric)
        
        print(f"Merged {len(merged_results)} unique recommendations from {len(batches)} batches")
        
        # If we used multiple batches, do a final ranking pass to get proper global ordering
        # (but only if it's feasible - don't attempt if we have too many results to fit in context)
        if len(batches) > 1 and len(merged_results) <= max_per_batch and len(merged_results) > 0:
            print(f"Performing final ranking on {len(merged_results)} candidates to fix cross-batch ordering")
            try:
                final_ranking = self._recommend_batch(merged_results, dataset, target_measurement, min(k, len(merged_results)))
                final_results = [self.metric_name_to_class(metric_name) for metric_name in final_ranking]
                final_results = [r for r in final_results if r is not None]
                return final_results[:k]
            except Exception as e:
                if self._is_context_error(e):
                    print(f"Final ranking exceeded context window, returning merged results: {e}")
                    return merged_results[:k]
                else:
                    raise e
        elif len(batches) > 1:
            print(f"Skipping final ranking: {len(merged_results)} results {'too many' if len(merged_results) > max_per_batch else 'too few'} for safe reranking")
        
        return merged_results[:k]

    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        """
        Recommend metrics for a given dataset and target measurement.
        Uses systematic token counting to plan batching strategy upfront.
        Implements iterative quota-filling when LLM returns fewer than 90% of requested metrics.
        """
        original_k = k
        all_recommended_metrics = []
        remaining_metrics = self.metric_classes.copy()
        
        print(f"Starting iterative recommendation: requesting {k} metrics from {len(remaining_metrics)} available")
        
        iteration = 1
        max_iterations = 5  # Prevent infinite loops
        
        while k > 0 and len(remaining_metrics) > 0 and iteration <= max_iterations:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Requesting {k} metrics from {len(remaining_metrics)} remaining metrics")
            
            # Get recommendations from remaining metrics
            try:
                batch_results = self._recommend_with_token_planning(remaining_metrics, dataset, target_measurement, k)
                print(f"Iteration {iteration}: LLM returned {len(batch_results)} metrics (requested {k})")
                
                if len(batch_results) == 0:
                    print("No metrics returned, stopping iterations")
                    break
                
                # Add new recommendations to our collection
                all_recommended_metrics.extend(batch_results)
                
                # Remove recommended metrics from remaining pool
                recommended_set = set(batch_results)
                remaining_metrics = [metric for metric in remaining_metrics if metric not in recommended_set]
                
                # Check if we need to continue iterating
                current_total = len(all_recommended_metrics)
                quota_filled_ratio = current_total / original_k
                
                print(f"Total metrics so far: {current_total}/{original_k} ({quota_filled_ratio:.1%})")
                
                if quota_filled_ratio >= 0.9:
                    print(f"Quota filled to {quota_filled_ratio:.1%}, stopping iterations")
                    break
                
                # Calculate how many more we need
                k = original_k - current_total
                print(f"Need {k} more metrics, {len(remaining_metrics)} metrics remaining")
                
                if k <= 0:
                    print("Quota exceeded, stopping iterations")
                    break
                    
            except Exception as e:
                print(f"Iteration {iteration} failed with error: {e}")
                # If we have some results, return what we have
                if all_recommended_metrics:
                    print(f"Returning {len(all_recommended_metrics)} metrics despite error")
                    break
                else:
                    # If no results at all, re-raise the error
                    raise e
            
            iteration += 1
        
        if iteration > max_iterations:
            print(f"Warning: Reached maximum iterations ({max_iterations}), returning current results")
        
        # Remove duplicates while preserving order
        seen = set()
        final_results = []
        for metric in all_recommended_metrics:
            if metric not in seen:
                final_results.append(metric)
                seen.add(metric)
        
        print(f"\nFinal result: {len(final_results)} unique metrics (requested {original_k})")
        return final_results[:original_k]