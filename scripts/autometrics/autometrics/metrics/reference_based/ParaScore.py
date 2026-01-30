from parascore import ParaScorer
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from typing import ClassVar

class ParaScore(ReferenceBasedMetric):
    """---
# Metric Card for ParaScore

ParaScore is a reference-based evaluation metric designed for paraphrase generation. It combines advantages of both reference-free and reference-based approaches by explicitly modeling semantic similarity and lexical divergence between generated and reference texts. ParaScore outputs three scores — Precision (P), Recall (R), and F1 — reflecting quality from different perspectives. It was introduced to address weaknesses found in traditional metrics when evaluating paraphrasing tasks.

## Metric Details

### Metric Description

ParaScore evaluates the quality of generated paraphrases by considering both semantic similarity and lexical diversity. It extends BERTScore-based similarity measures by incorporating a lexical divergence term. ParaScore's "base_score" formulation combines:
- Semantic similarity between candidate and source,
- Semantic similarity between candidate and reference,
- Lexical diversity between candidate and source.

Specifically, the maximum similarity between the candidate and either the source or reference is combined with a scaled lexical diversity score (edit distance-based). This hybrid design enables ParaScore to better model paraphrase generation quality, which demands both meaning preservation and surface-level divergence.

- **Metric Type:** Semantic Similarity
- **Range:** [0, 1]
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Given candidate sentence $c$, source sentence $s$, and reference sentence(s) $r$, ParaScore is defined as:

1. Compute semantic similarity scores:
   - $S_{s\text{-}c}$: Semantic similarity between source $s$ and candidate $c$.
   - $S_{r\text{-}c}$: Semantic similarity between reference $r$ and candidate $c$.

2. Compute lexical diversity score:
   - $D_{s\text{-}c}$: Lexical diversity between source $s$ and candidate $c$, based on normalized edit distance.

3. ParaScore base score for each candidate is:

$$
\text{BaseScore}(c, s, r) = \max(S_{s\text{-}c}, S_{r\text{-}c}) + 0.05 \times D_{s\text{-}c}
$$

The semantic similarity scores are derived from a cosine similarity of contextual embeddings (BERT-style models), and the diversity score uses a linear scaling of normalized edit distance.

Each output (P, R, F1) reflects standard precision, recall, and F1 scoring over contextual embeddings weighted by IDF.

### Inputs and Outputs

- **Inputs:**  
  - Source sentence (input)  
  - Candidate sentence (output)  
  - Reference sentence(s)  

- **Outputs:**  
  - A tuple of three scalar scores: Precision (P), Recall (R), and F1

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Paraphrasing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating paraphrase generation tasks where preserving meaning while achieving sufficient lexical variation is critical. Suitable for both academic evaluation and production benchmarking in paraphrase generation systems.

- **Not Recommended For:**  
  Tasks focused solely on content fidelity (e.g., translation, summarization) where lexical diversity is not desired, as ParaScore explicitly rewards divergence from the input.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [parascore](https://pypi.org/project/parascore/) (official PyPI package)
  - Custom implementations in research pipelines based on ParaScorer class.

### Computational Complexity

- **Efficiency:**  
  Moderately expensive due to contextual embedding computations using large language models (e.g., RoBERTa-large, BERT-based encoders).

- **Scalability:**  
  ParaScore supports batched processing and GPU acceleration via PyTorch, making it scalable to moderately large datasets. However, inference time grows linearly with the number of candidate-reference pairs and the size of the embedding model.

## Known Limitations

- **Biases:**  
  - Dependency on the pre-trained language model (e.g., BERT, RoBERTa) could introduce biases learned during model pretraining (e.g., biases in language, domain-specific vocabulary).

- **Task Misalignment Risks:**  
  - ParaScore assumes that lexical diversity is *desirable*. In tasks where copying the source is acceptable or even preferred (e.g., summarization, certain translation setups), ParaScore may penalize good outputs.

- **Failure Cases:**  
  - ParaScore may underperform when either the source or the reference are noisy or semantically ambiguous, since it relies on maximum similarity across source and reference.  
  - When candidates are very short or trivial paraphrases, the diversity term might disproportionately influence the score.

## Related Metrics

- **BERTScore:** ParaScore extends BERTScore by adding explicit modeling of lexical diversity.
- **BLEU, ROUGE:** Traditional surface-level metrics; less suitable for paraphrase evaluation due to focus on exact n-gram overlap.
- **MoverScore:** Another semantic similarity metric, but without explicit lexical diversity modeling.
- **ParaSim:** A related metric focused on semantic preservation for paraphrasing without the diversity component.

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
    gpu_mem: ClassVar[float] = 338.13323437500003  # in MB
    cpu_mem: ClassVar[float] = 1079.51953125  # in MB
    description: ClassVar[str] = "ParaScore is a reference-based evaluation metric designed for paraphrase generation. It combines advantages of both reference-free and reference-based approaches by explicitly modeling semantic similarity and lexical divergence between generated and reference texts. ParaScore outputs three scores — Precision (P), Recall (R), and F1 — reflecting quality from different perspectives. It was introduced to address weaknesses found in traditional metrics when evaluating paraphrasing tasks."

    def __init__(
        self,
        name: str = "ParaScore",
        description: str = "ParaScore is a reference-based evaluation metric designed for paraphrase generation. It combines advantages of both reference-free and reference-based approaches by explicitly modeling semantic similarity and lexical divergence between generated and reference texts. ParaScore outputs three scores — Precision (P), Recall (R), and F1 — reflecting quality from different perspectives. It was introduced to address weaknesses found in traditional metrics when evaluating paraphrasing tasks.",
        seed: int = 42,
        **scorer_kwargs
    ):
        if "lang" not in scorer_kwargs:
            scorer_kwargs["lang"] = "en"

        if "model_type" not in scorer_kwargs:
            scorer_kwargs["model_type"] = "bert-base-uncased"

        super().__init__(name=name, description=description, seed=seed, **scorer_kwargs)

        # remove the following from scorer_kwargs:
        scorer_kwargs.pop("cache_dir", None)
        scorer_kwargs.pop("seed", None)
        scorer_kwargs.pop("use_cache", None)
        scorer_kwargs.pop("device", None)
        scorer_kwargs.pop("cache_size_limit", None)
        scorer_kwargs.pop("cache_ttl", None)
        scorer_kwargs.pop("force_cache", None)
        scorer_kwargs.pop("_hint_gpu_index", None)

        self.scorer = ParaScorer(**scorer_kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):
        if not references:
            raise ValueError("ParaScore (reference-based) requires `references`.")
        
        # Ensure inputs are strings to avoid parascore errors on numeric types
        cands = [str(output)]
        srcs = [str(input)]
        
        # ensure list-of-lists
        if isinstance(references[0], list):
            refs_batch = [[str(r) for r in ref_list] for ref_list in references]
        else:
            refs_batch = [[str(r) for r in references]]
            
        # use hybrid base_score
        result = self.scorer.base_score(cands, srcs, refs_batch, **kwargs)
        
        # If we got an empty list, return zero
        if not result:
            return 0.0
            
        # ParaScorer returns a list with a single value
        return result[0]

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate scores for a batch of inputs/outputs.
        ParaScorer.base_score returns a list of scores [score1, score2, ...] for each input.
        """
        # Cast to strings as well
        inputs = [str(i) for i in inputs]
        outputs = [str(o) for o in outputs]
        refs_batch = references if references is not None else [[] for _ in inputs]
        
        # Cast references to strings (handles nested list)
        refs_batch = [[str(r) for r in ref_list] for ref_list in refs_batch]
        
        # Get scores
        scores_list = self.scorer.base_score(outputs, inputs, refs_batch, **kwargs)
        
        # If we got an empty list or the scores don't match the inputs, return zeros
        if not scores_list or len(scores_list) != len(inputs):
            return [0.0] * len(inputs)
        
        # Return the scores directly
        return scores_list