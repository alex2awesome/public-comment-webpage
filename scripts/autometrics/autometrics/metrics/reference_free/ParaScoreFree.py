from parascore import ParaScorer
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
import torch
from typing import ClassVar
import os

class ParaScoreFree(ReferenceFreeMetric):
    """---
# Metric Card for ParaScoreFree

ParaScoreFree is a reference-free evaluation metric designed for paraphrase generation. It evaluates candidate paraphrases based on semantic similarity to the input source while encouraging lexical diversity. ParaScoreFree outputs a scalar quality score that combines BERT-based semantic similarity and normalized edit distance, offering a balance between meaning preservation and surface-level rewriting. It enables paraphrase evaluation without the need for gold reference texts, making it suitable for low-resource or open-domain settings.

## Metric Details

### Metric Description

ParaScoreFree computes a hybrid score by combining:
- Semantic similarity between the input source and the candidate paraphrase, measured using BERTScore-style contextual embeddings.
- Lexical divergence, modeled using normalized edit distance and a sectional function to reward moderate levels of surface variation.

This design explicitly balances fidelity and diversity in paraphrase generation evaluation without relying on reference sentences.

- **Metric Type:** Reference-Free
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Let $X$ be the input (source) sentence and $C$ the candidate paraphrase.

The ParaScoreFree score is defined as:

$$
\text{ParaScoreFree}(X, C) = \text{Sim}(X, C) + \omega \cdot \text{DS}(X, C)
$$

Where:
- $\text{Sim}(X, C)$ is the semantic similarity score between $X$ and $C$, computed using BERTScore (cosine similarity of contextual embeddings).
- $\text{DS}(X, C)$ is the divergence score based on normalized edit distance (NED) between $X$ and $C$.
- $\omega$ is a small positive weight (default 0.05).

The divergence score $\text{DS}(X, C)$ uses a sectional function:

$$
\text{DS}(X, C) =
\begin{cases}
-1 + \frac{\gamma + 1}{\gamma} \cdot d, & \text{if } d \leq \gamma \\
\gamma, & \text{if } d > \gamma
\end{cases}
$$

where:
- $d$ is the normalized edit distance between $X$ and $C$,
- $\gamma$ is a threshold (typically $\gamma = 0.35$).

### Inputs and Outputs

- **Inputs:**  
  - Source text (input sentence)  
  - Candidate text (paraphrased sentence)

- **Outputs:**  
  - Scalar quality score (range: 0 to 1)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Paraphrasing, Data Augmentation, Style Transfer

### Applicability and Limitations

- **Best Suited For:**  
  Evaluation of paraphrase generation systems when no human references are available, or when large-scale automatic evaluation is needed in low-resource settings.

- **Not Recommended For:**  
  Tasks where exact content preservation is required without stylistic divergence (e.g., faithful summarization, strict translation).

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [parascore](https://pypi.org/project/parascore/) (official PyPI package)

### Computational Complexity

- **Efficiency:**  
  Moderate — dominated by transformer model inference for semantic similarity and edit distance computation.

- **Scalability:**  
  Scales linearly with the number of candidate-source pairs. Batched BERT inference and parallel edit distance computation can improve efficiency.

## Known Limitations

- **Biases:**  
  Relies on the pre-trained language models (e.g., BERT, RoBERTa), which may encode societal or linguistic biases.

- **Task Misalignment Risks:**  
  The metric assumes that some degree of surface-level difference is desirable; it may undervalue outputs that are faithful but lexically conservative.

- **Failure Cases:**  
  - If the candidate is too short, lexical diversity rewards may dominate.  
  - Semantic similarity estimates can be noisy for highly creative or informal texts.

## Related Metrics

- **ParaScore:** Reference-based counterpart that compares candidate paraphrases to human references.
- **BERTScore:** Measures semantic similarity but does not model diversity.
- **BLEU, ROUGE:** Surface-based metrics less aligned with paraphrase evaluation.
- **BERT-iBLEU:** Earlier hybrid semantic-lexical metric, but less effective according to ParaScore authors.

## Further Reading

- **Papers:**  
  - [On the Evaluation Metrics for Paraphrase Generation (Shen et al., 2022)](https://aclanthology.org/2022.emnlp-main.208/)

- **Blogs/Tutorials:**  
  Needs more information.


## Citation

```
@inproceedings{shen-etal-2022-evaluation,
    title = "On the Evaluation Metrics for Paraphrase Generation",
    author = "Shen, Lingfeng  and
      Liu, Lemao  and
      Jiang, Haiyun  and
      Shi, Shuming",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.208/",
    doi = "10.18653/v1/2022.emnlp-main.208",
    pages = "3178--3190",
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided documents and source materials. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 346.365234375  # in MB
    cpu_mem: ClassVar[float] = 1718.91796875  # in MB
    description: ClassVar[str] = "ParaScoreFree is a reference-free evaluation metric designed for paraphrase generation. It evaluates candidate paraphrases based on semantic similarity to the input source while encouraging lexical diversity. ParaScoreFree outputs a scalar quality score that combines BERT-based semantic similarity and normalized edit distance, offering a balance between meaning preservation and surface-level rewriting. It enables paraphrase evaluation without the need for gold reference texts, making it suitable for low-resource or open-domain settings."
    
    def __init__(
        self,
        name: str = "ParaScoreFree",
        description: str = "ParaScoreFree is a reference-free evaluation metric designed for paraphrase generation. It evaluates candidate paraphrases based on semantic similarity to the input source while encouraging lexical diversity. ParaScoreFree outputs a scalar quality score that combines BERT-based semantic similarity and normalized edit distance, offering a balance between meaning preservation and surface-level rewriting. It enables paraphrase evaluation without the need for gold reference texts, making it suitable for low-resource or open-domain settings.",
        seed: int = 42,
        device: str | None = None,
        **scorer_kwargs
    ):
        if "lang" not in scorer_kwargs:
            scorer_kwargs["lang"] = "en"

        if "model_type" not in scorer_kwargs:
            scorer_kwargs["model_type"] = "bert-base-uncased"

        # Decide device preference (default to CPU for robustness)
        if device is None:
            device = 'cpu'

        super().__init__(name, description, seed=seed, device=device, **scorer_kwargs)

        # remove the following from scorer_kwargs:
        scorer_kwargs.pop("cache_dir", None)
        scorer_kwargs.pop("seed", None)
        scorer_kwargs.pop("use_cache", None)
        # Keep a copy of ctor kwargs for safe re-instantiation
        self._scorer_ctor_kwargs = dict(scorer_kwargs)
        scorer_kwargs.pop("cache_size_limit", None)
        scorer_kwargs.pop("cache_ttl", None)
        scorer_kwargs.pop("force_cache", None)
        scorer_kwargs.pop("_hint_gpu_index", None)

        # Try to create the scorer; avoid passing unknown device kwargs
        try:
            self.scorer = ParaScorer(**self._scorer_ctor_kwargs)
        except RuntimeError as e:
            if 'CUBLAS_STATUS_EXECUTION_FAILED' in str(e) or 'cublas' in str(e).lower() or 'device-side assert' in str(e).lower():
                # Reconstruct on CPU if supported, else hide CUDA
                try:
                    cpu_kwargs = dict(self._scorer_ctor_kwargs)
                    cpu_kwargs['device'] = 'cpu'
                    self.scorer = ParaScorer(**cpu_kwargs)
                except Exception:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    self.scorer = ParaScorer(**self._scorer_ctor_kwargs)
            else:
                raise

    def _calculate_impl(self, input, output, references=None, **kwargs):
        cands = [str(output)]
        srcs = [str(input)]
        try:
            result = self.scorer.free_score(cands, srcs, **kwargs)
        except RuntimeError as e:
            msg = str(e).lower()
            if 'cublas_status_execution_failed' in msg or 'device-side assert' in msg or 'cublas' in msg:
                # Recreate scorer on CPU and retry once
                try:
                    cpu_kwargs = dict(self._scorer_ctor_kwargs)
                    cpu_kwargs['device'] = 'cpu'
                    self.scorer = ParaScorer(**cpu_kwargs)
                    result = self.scorer.free_score(cands, srcs, **kwargs)
                except Exception:
                    raise
            else:
                raise

        # ParaScorer.free_score might return a list of torch.Tensors or a list of
        # python floats depending on the version.  Handle both gracefully.
        first = result[0]
        if hasattr(first, "cpu"):
            # torch.Tensor -> if multi-element take mean, then convert to float
            if first.numel() == 1:
                return float(first.cpu().item())
            else:
                return float(first.mean().cpu().item())
        # Assume it is already a python (float, int, np.float) value
        return float(first)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        # Ensure string casting for all elements before passing to scorer
        outputs = [str(o) for o in outputs]
        inputs = [str(i) for i in inputs]
        try:
            results = self.scorer.free_score(outputs, inputs, **kwargs)
        except RuntimeError as e:
            msg = str(e).lower()
            if 'cublas_status_execution_failed' in msg or 'device-side assert' in msg or 'cublas' in msg:
                # Retry on CPU with a fresh scorer if necessary
                try:
                    cpu_kwargs = dict(self._scorer_ctor_kwargs)
                    cpu_kwargs['device'] = 'cpu'
                    self.scorer = ParaScorer(**cpu_kwargs)
                    results = self.scorer.free_score(outputs, inputs, **kwargs)
                except Exception:
                    raise
            else:
                raise

        # Newer versions of parascore return a plain python list. Older versions
        # may return a torch.Tensor.  Normalize to a list of floats.
        if hasattr(results, "cpu"):
            # torch tensor -> convert to cpu list
            return results.cpu().tolist()
        elif isinstance(results, list):
            # Case: results is list where each element could be a tensor or numeric.
            # If *all* elements are tensors of the same length, aggregate by taking
            # the mean across the list (mirrors how ParaScorer.base_score picks the
            # F1 component). This yields a length-N list matching the batch size.
            if results and all(hasattr(r, "cpu") for r in results):
                try:
                    stacked = torch.stack([r.float() for r in results], dim=0)  # shape (k, N)
                    mean_vals = stacked.mean(0).cpu().tolist()
                    return [float(v) for v in mean_vals]
                except Exception:
                    # Fallback to element-wise processing if shapes mismatch
                    pass
            # Otherwise process each element individually
            processed = []
            for r in results:
                if hasattr(r, "cpu"):
                    # torch tensor: if multi-element, take mean
                    r_val = r.mean() if r.numel() > 1 else r
                    processed.append(float(r_val.cpu().item()))
                else:
                    # plain numeric
                    processed.append(float(r))
            return processed
        else:
            # Fallback: attempt to cast iterable to list of floats
            try:
                return [float(r) for r in results]
            except Exception:
                raise TypeError(
                    "Unexpected return type from ParaScorer.free_score: "
                    f"{type(results)}. Expected list or tensor."
                )