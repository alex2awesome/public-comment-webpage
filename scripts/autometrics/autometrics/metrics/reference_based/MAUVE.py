from autometrics.metrics.reference_based.HuggingFaceReferenceBasedMetric import HuggingFaceReferenceBasedMetric
from typing import ClassVar

def _aggregate(values, method):
    """
    Aggregates a list of numeric values using the specified method.
    
    Args:
        values (list of float/int): The values to aggregate.
        method (str): Aggregation method: "min", "max", or "avg".
    
    Returns:
        The aggregated value.
    """
    if not values:
        return None
    if method == "min":
        return min(values)
    elif method == "max":
        return max(values)
    elif method == "avg":
        return sum(values) / len(values)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

class MAUVE(HuggingFaceReferenceBasedMetric):
    """---
# Metric Card for MAUVE

MAUVE (Measuring the Alignment of Unconditional VErsions) quantifies the similarity between two text distributions (e.g., generated vs. human-written) by computing divergence frontiers based on KL divergences between their quantized representations in the embedding space of a large language model. MAUVE can capture nuanced differences due to model size, decoding strategy, or topic shift, and is especially useful in evaluating open-ended text generation tasks.

## Metric Details

### Metric Description

MAUVE evaluates how close the distribution of generated text is to that of human-written text. It operates by:
1. Embedding text samples using a large pretrained language model (e.g., GPT-2).
2. Applying PCA for dimensionality reduction and k-means for quantization to discretize the feature space.
3. Estimating histograms over quantized regions for both distributions.
4. Computing divergence curves using KL divergences of mixtures between these histograms.
5. Calculating the area under the divergence frontier curve to yield a scalar MAUVE score.

MAUVE can distinguish between generations of different quality, decoding strategies, and model sizes. The smoothed variant, MAUVE*, incorporates Krichevsky-Trofimov smoothing for improved robustness.

- **Metric Type:** Semantic Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

Let $p$ and $q$ be quantized histograms of two text distributions (e.g., reference and generated text). MAUVE is defined using divergence frontiers derived from KL divergences between convex combinations of $p$ and $q$.

Let $r = w p + (1 - w) q$ for $w \in (0,1)$, and define:

$$
\text{KL}(p \| r) = \sum _{i} p_i \log \frac{p_i}{r_i}, \quad \text{KL}(q \| r) = \sum _{i} q_i \log \frac{q_i}{r_i}
$$

The divergence curve consists of points:

$$
\left( \exp(-c \cdot \text{KL}(q \| r)), \exp(-c \cdot \text{KL}(p \| r)) \right)
$$

The MAUVE score is the symmetric area under the divergence curve:

$$
\text{MAUVE}(p, q) = \frac{1}{2} \left[ \text{AUC}(x, y) + \text{AUC}(y, x) \right]
$$

where $c$ is a scaling factor (default: 5), and AUC is the area under the curve computed using sorted points.

### Inputs and Outputs

- **Inputs:**  
  - `p_text`: list of generated texts (or `p_tokens`, `p_features`)
  - `q_text`: list of human/reference texts (or `q_tokens`, `q_features`)

- **Outputs:**  
  - `mauve`: scalar score ∈ [0, 1]  
  - `frontier_integral`: alternative distance measure (lower is better)  
  - `mauve_star`: smoothed variant of MAUVE  
  - `p_hist`, `q_hist`: quantized histograms  
  - `divergence_curve`: array of $(\text{KL}_q, \text{KL}_p)$ points

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Open-Ended Text Generation, Dialogue Generation, Storytelling, Summarization, Style Transfer

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating large-scale text generation systems where distributional similarity to human writing is important (e.g., language modeling, creative writing, summarization at scale).

- **Not Recommended For:**  
  Fine-grained sentence-level comparison or evaluation settings where token-level matching (e.g., translation) is prioritized.

- ⚠️ **Note on Current Usage in Metric Bank:**  
  In this metric bank, MAUVE is currently used in a setting with **one generated output and multiple references**, despite the original paper recommending **≥ 1000 samples per distribution** for reliable estimation. This usage is therefore **out-of-distribution for the intended application of MAUVE**. A future refactor may support true distributional evaluation, but users should interpret results with caution in the current setup.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [mauve-text (official)](https://github.com/krishnap25/mauve)
  - [HuggingFace Evaluate](https://huggingface.co/spaces/evaluate-metric/mauve)

### Computational Complexity

- **Efficiency:**  
  Moderately expensive due to LLM featurization and clustering; PCA and k-means add computational cost.

- **Scalability:**  
  Scales well to large datasets when using pre-computed features; authors recommend ≥1000 samples per distribution.

## Known Limitations

- **Biases:**  
  Embedding bias: MAUVE inherits any inductive biases present in the embedding model (e.g., GPT-2), which can affect distributional similarity estimates.

- **Task Misalignment Risks:**  
  May underestimate quality when high-quality text differs stylistically, topically, or length-wise from the human reference corpus.

- **Failure Cases:**  
  Cannot reliably detect subtle decoding changes (e.g., $p=0.95$ vs $p=0.96$ in top-p sampling); sensitive to seed and stochasticity in generation/clustering.

## Related Metrics

- **Fréchet Distance (FVD/TVD):** Similar in spirit for generative image evaluation.
- **BERTScore:** Evaluates semantic similarity using contextual embeddings but is sentence-level.
- **Self-BLEU / Distinct-n:** Capture diversity but not quality.
- **Perplexity:** Captures fluency but not alignment with a reference distribution.

## Further Reading

- **Papers:**  
  - [Pillutla et al., 2023 (JMLR)](https://www.jmlr.org/papers/v24/21-1214.html)  
  - [Pillutla et al., 2021 (NeurIPS)](https://arxiv.org/abs/2102.01454)  
  - [Liu et al., 2021 (NeurIPS)](https://arxiv.org/abs/2102.04130)

- **Blogs/Tutorials:**  
  - [MAUVE GitHub](https://github.com/krishnap25/mauve)  
  - [HuggingFace Metric Card](https://huggingface.co/spaces/evaluate-metric/mauve)

## Citation

```
@article{pillutla-etal:mauve:jmlr2023,  
  title={{MAUVE Scores for Generative Models: Theory and Practice}},  
  author={Pillutla, Krishna and Liu, Lang and Thickstun, John and Welleck, Sean and Swayamdipta, Swabha and Zellers, Rowan and Oh, Sewoong and Choi, Yejin and Harchaoui, Zaid},  
  journal={JMLR},  
  year={2023}  
}
```

```
@inproceedings{pillutla-etal:mauve:neurips2021,  
  title={MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers},  
  author={Pillutla, Krishna and Swayamdipta, Swabha and Zellers, Rowan and Thickstun, John and Welleck, Sean and Choi, Yejin and Harchaoui, Zaid},  
  booktitle = {NeurIPS},  
  year      = {2021}  
}
```

```
@inproceedings{liu-etal:mauve-theory:neurips2021,  
  title={{Divergence Frontiers for Generative Models: Sample Complexity, Quantization Effects, and Frontier Integrals}},  
  author={Liu, Lang and Pillutla, Krishna and Welleck, Sean and Oh, Sewoong and Choi, Yejin and Harchaoui, Zaid},  
  booktitle={NeurIPS},  
  year={2021}  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 4322.96484375  # in MB
    description: ClassVar[str] = "MAUVE (Measuring the Alignment of Unconditional VErsions) quantifies the similarity between two text distributions (e.g., generated vs. human-written) by computing divergence frontiers based on KL divergences between their quantized representations in the embedding space of a large language model. MAUVE can capture nuanced differences due to model size, decoding strategy, or topic shift, and is especially useful in evaluating open-ended text generation tasks."  

    def __init__(
        self,
        name: str = "MAUVE",
        description: str = "MAUVE (Measuring the Alignment of Unconditional VErsions) quantifies the similarity between two text distributions (e.g., generated vs. human-written) by computing divergence frontiers based on KL divergences between their quantized representations in the embedding space of a large language model. MAUVE can capture nuanced differences due to model size, decoding strategy, or topic shift, and is especially useful in evaluating open-ended text generation tasks.",
        persistent: bool = True,
        aggregation: str = "max",
        metric_id: str = "mauve",
        score_key: str = "mauve",
        load_kwargs: dict = {},
        **kwargs
    ):
        # Add aggregation to the name if provided
        if aggregation not in ["min", "max", "avg"]:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
            
        metric_name = f"{name}_{aggregation}" if name == "MAUVE" else name

        # Fix for device mapping issues: always use CPU to avoid conflicts
        # The HuggingFace evaluate library has internal device management that conflicts
        # with our GPU allocation system, so we force CPU usage
        load_kwargs["device"] = "cpu"
        
        # Pass parameters to parent constructor, avoiding duplicate arguments
        super().__init__(
            name=metric_name,
            description=description,
            metric_id=metric_id,
            score_key=score_key,
            load_kwargs=load_kwargs,
            persistent=persistent,
            aggregation=aggregation,
            **kwargs
        )
        self.persistent = persistent
        self.aggregation = aggregation
        self.exclude_from_cache_key("persistent")
        
    def _calculate_impl(self, input: str, output: str, references=None, **kwargs) -> float:
        """
        Compute MAUVE score for one example with multiple references.
        
        For each reference, a separate MAUVE score is computed, then scores are
        aggregated according to the specified method (min, max, avg).
        """
        self._load_metric()
        
        if references is None or len(references) == 0:
            return 0.0  # No references; return default score
            
        # Handle different reference formats
        all_refs = []
        
        # Handle string reference
        if isinstance(references, str):
            all_refs = [references]
        # Handle nested list references
        elif isinstance(references[0], list) or isinstance(references[0], tuple):
            # Flatten one level - we want a list of strings
            for ref_list in references:
                if isinstance(ref_list, (list, tuple)) and ref_list:
                    all_refs.extend(ref_list)
                elif isinstance(ref_list, str):
                    all_refs.append(ref_list)
        # Handle array of strings reference
        else:
            all_refs = references
            
        if not all_refs:
            return 0.0  # No valid references
        
        # Compute scores for each reference
        scores = []
        for ref in all_refs:
            # Call compute with single reference
            result = self.metric.compute(predictions=[output], references=[ref], **kwargs)
            # Handle both dict-like and SimpleNamespace-like result objects
            if hasattr(result, "get"):  # Dict-like object
                val = result.get(self.score_key)
            elif hasattr(result, self.score_key):  # SimpleNamespace-like object
                val = getattr(result, self.score_key)
            else:
                # If we can't find the score key, try to access 'mauve' directly
                if hasattr(result, "mauve"):
                    val = result.mauve
                else:
                    raise ValueError(f"Cannot extract {self.score_key} from MAUVE result: {result}")
            score = float(val) if not isinstance(val, (list, tuple)) else float(val[0])
            scores.append(score)
        
        # Aggregate scores
        return _aggregate(scores, self.aggregation)
    
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate scores for a batch of inputs/outputs with multiple references per example.
        
        For each example, compute scores for all of its references and aggregate them.
        """
        self._load_metric()
        
        if references is None:
            references = [None] * len(inputs)
            
        # Process one example at a time, handling multiple references
        scores = []
        for i, (inp, out, refs) in enumerate(zip(inputs, outputs, references)):
            # Use _calculate_impl to handle the reference formatting and aggregation
            score = self._calculate_impl(inp, out, refs, **kwargs)
            scores.append(score)
        
        return scores 