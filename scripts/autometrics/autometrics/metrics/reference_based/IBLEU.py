from sacrebleu.metrics import BLEU as bleu
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from typing import ClassVar

class IBLEU(ReferenceBasedMetric):
    """---
# Metric Card for iBLEU

iBLEU (input-aware BLEU) is a reference-based metric introduced to evaluate paraphrase generation quality by simultaneously measuring semantic adequacy and surface dissimilarity. It modifies the traditional BLEU score by penalizing outputs that are too similar to the input (source) sentence, thus encouraging more diverse and non-trivial paraphrases.

## Metric Details

### Metric Description

iBLEU evaluates the quality of generated text by computing a weighted combination of two BLEU scores:
- BLEU between the candidate output and the reference(s), which captures adequacy.
- BLEU between the candidate output and the input sentence, which is used as a penalty to encourage dissimilarity.

This metric is particularly useful for paraphrase generation tasks, where outputs should preserve meaning while varying in surface form. The weighting factor $\alpha$ balances the trade-off between adequacy and diversity.

- **Metric Type:** Surface-Level Similarity  
- **Range:** Unbounded below, typically between 0 and 1 for reasonable $\alpha$  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** Yes  

### Formal Definition

Let $O$ be the output (candidate), $R$ be the set of references, and $I$ be the input sentence. Then:

$$
\text{iBLEU}(O, R, I) = \alpha \cdot \text{BLEU}(O, R) - (1 - \alpha) \cdot \text{BLEU}(O, I)
$$

Where:
- $\alpha \in [0, 1]$ is a tunable weight balancing semantic adequacy and dissimilarity.
- $\text{BLEU}(O, R)$ is the BLEU score of the candidate against the references.
- $\text{BLEU}(O, I)$ is the BLEU score of the candidate against the input (self-BLEU penalty).

### Inputs and Outputs

- **Inputs:**  
  - Input sentence (source)  
  - Generated output (candidate)  
  - One or more reference sentences  

- **Outputs:**  
  - Scalar iBLEU score (float)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Paraphrasing, Text Simplification, Dialogue Generation  

### Applicability and Limitations

- **Best Suited For:**  
  Paraphrasing and text simplification tasks where output diversity from the input is desired but semantic meaning must be preserved.

- **Not Recommended For:**  
  Tasks where high lexical overlap with the input is expected (e.g., extractive summarization), or where references are unavailable.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Custom implementations exist in evaluation libraries such as EditEval.  
  - Can be implemented using SacreBLEU or Hugging Face `evaluate` for BLEU calculation.

### Computational Complexity

- **Efficiency:**  
  Linear in the number of tokens in the input, candidate, and reference(s); efficiency is similar to standard BLEU.

- **Scalability:**  
  Scales well for batch evaluations. The major bottleneck is BLEU computation, which is efficient with libraries like SacreBLEU.

## Known Limitations

- **Biases:**  
  - Inherits limitations from BLEU, including sensitivity to tokenization and inability to capture semantic similarity beyond lexical overlap.
  - Penalizes outputs that appropriately reuse tokens from the input, even when such reuse is semantically justified.

- **Task Misalignment Risks:**  
  - Over-penalizing similarity to input can lead to degraded semantic adequacy if $\alpha$ is set too low.  
  - Can encourage unnecessary lexical variation that harms meaning fidelity.

- **Failure Cases:**  
  - When paraphrases are semantically faithful but lexically similar (e.g., minor rewordings), iBLEU may under-score their quality.  
  - Inappropriate for tasks where copying or high overlap with input is beneficial.

## Related Metrics

- **BLEU:** iBLEU extends BLEU with an input-aware penalty.  
- **Self-BLEU:** Used as the penalty term in iBLEU to measure output-input similarity.  
- **SARI:** Another metric that explicitly evaluates simplicity, addition, and deletion, especially for text simplification.  
- **BERTScore:** Captures semantic similarity using contextual embeddings; can complement iBLEU in cases where meaning preservation is subtle.

## Further Reading

- **Papers:**  
  - [Joint Learning of a Dual SMT System for Paraphrase Generation (Sun and Zhou, 2012)](https://aclanthology.org/P12-2008/)

- **Blogs/Tutorials:**  
  [More Information Needed]

## Citation

```
@inproceedings{sun-zhou-2012-joint,  
    title = "Joint Learning of a Dual {SMT} System for Paraphrase Generation",  
    author = "Sun, Hong  and  
      Zhou, Ming",  
    editor = "Li, Haizhou  and  
      Lin, Chin-Yew  and  
      Osborne, Miles  and  
      Lee, Gary Geunbae  and  
      Park, Jong C.",  
    booktitle = "Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",  
    month = jul,  
    year = "2012",  
    address = "Jeju Island, Korea",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/P12-2008/",  
    pages = "38--42"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 730.34765625  # in MB
    description: ClassVar[str] = "iBLEU (input-aware BLEU) is a reference-based metric introduced to evaluate paraphrase generation quality by simultaneously measuring semantic adequacy and surface dissimilarity. It modifies the traditional BLEU score by penalizing outputs that are too similar to the input (source) sentence, thus encouraging more diverse and non-trivial paraphrases."

    def __init__(self,
                 name: str = "iBLEU",
                 description: str = "iBLEU (input-aware BLEU) is a reference-based metric introduced to evaluate paraphrase generation quality by simultaneously measuring semantic adequacy and surface dissimilarity. It modifies the traditional BLEU score by penalizing outputs that are too similar to the input (source) sentence, thus encouraging more diverse and non-trivial paraphrases.",
                 alpha: float = 0.9,
                 **kwargs):
        super().__init__(name, description, alpha=alpha, **kwargs)
        self.metric = bleu()
        self.alpha = alpha

    def _calculate_impl(self, input: str, output: str, references=None, alpha: float = None, **kwargs) -> float:
        """
        Calculate the iBLEU score for a hypothesis.

        Args:
            input: Source sentence (string).
            output: Candidate translation (string).
            references: List of reference translation strings.
            alpha: Weight for reference BLEU (overrides default).
        Returns:
            A float iBLEU score: alpha * BLEU(refs, cand) - (1-alpha) * BLEU(src, cand).
        """
        if references is None:
            references = []
        # determine alpha
        alpha_val = alpha if alpha is not None else self.alpha

        # prepare streams for sacreBLEU
        sys_stream = [output]
        # list of reference streams, one per reference
        ref_streams = [[r] for r in references]
        # BLEU against references
        bleu_ref = self.metric.corpus_score(sys_stream, ref_streams).score

        # BLEU against source (self-BLEU)
        src_streams = [[input]]
        bleu_self = self.metric.corpus_score(sys_stream, src_streams).score

        return alpha_val * bleu_ref - (1 - alpha_val) * bleu_self 