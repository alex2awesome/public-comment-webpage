from nltk.translate.nist_score import sentence_nist
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from typing import ClassVar

class NIST(ReferenceBasedMetric):
    """---
# Metric Card for NIST

NIST is an automatic evaluation metric for machine translation that measures n-gram overlap between a system hypothesis and one or more reference translations. It extends BLEU by weighting n-grams based on their informativeness, giving greater importance to rare n-grams. Additionally, it employs a smoother brevity penalty to minimize penalization for small variations in output length.

## Metric Details

### Metric Description

NIST (National Institute of Standards and Technology) is a reference-based evaluation metric designed to measure the quality of machine translation outputs. It modifies BLEU in two important ways:

1. **Informativeness Weighting**: Rather than giving equal weight to all matched n-grams, NIST emphasizes rarer (and thus more informative) n-grams. The intuition is that correct generation of uncommon phrases like "interesting calculations" should be rewarded more than very frequent ones like "on the".

2. **Modified Brevity Penalty**: BLEU's brevity penalty is replaced with a smoother version that reduces the sensitivity of the score to minor length differences between system output and reference translations. This avoids disproportionately penalizing outputs that are valid but slightly shorter than references.

The NIST score is computed as an arithmetic mean over n-gram information contributions, in contrast to BLEU's geometric mean of n-gram precisions.

- **Metric Type:** Surface-Level Similarity  
- **Range:** Unbounded, typically non-negative and upper-bounded by dataset characteristics  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** No

### Formal Definition

Let $w_1, w_2, \dots, w_n$ be an n-gram observed in the system output and reference. The information gain of the n-gram is:

$$
\text{Info}(w_1 \dots w_n) = \log_2 \left( \frac{\#(w_1 \dots w_{n-1})}{\#(w_1 \dots w_n)} \right)
$$

where $\#(w_1 \dots w_n)$ is the count of the full n-gram in the reference corpus and $\#(w_1 \dots w_{n-1})$ is the count of its prefix.

The total score is computed as:

$$
\text{NIST} = \left( \sum_{i=1}^N \frac{\sum_{\text{matched } n\text{-grams}} \text{Info}(n\text{-gram})}{\text{Total number of } n\text{-grams in hypothesis}} \right) \cdot \text{Penalty}
$$

The brevity penalty is defined as:

$$
\text{Penalty} = \begin{cases}
\exp \left( \beta \cdot \left[\log\left(\frac{L_\text{sys}}{L_\text{ref}}\right)\right]^2 \right) & \text{if } L_\text{sys} < L_\text{ref} \\
1 & \text{otherwise}
\end{cases}
$$

where $\beta$ is a constant (typically chosen so that penalty = 0.5 when $L_\text{sys} = 2/3 \cdot L_\text{ref}$).

### Inputs and Outputs

- **Inputs:**  
  - System-generated translation (hypothesis)  
  - One or more human-written reference translations  

- **Outputs:**  
  - A scalar NIST score (typically ≥ 0)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation, Summarization, Paraphrasing

### Applicability and Limitations

- **Best Suited For:**  
  Tasks where closer alignment with reference outputs is desirable, and informativeness of generated phrases is important (e.g., MT evaluation for official benchmarks).

- **Not Recommended For:**  
  Open-ended generation tasks like creative writing or dialogue, where multiple diverse outputs can be valid but do not match references lexically.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [NLTK](https://www.nltk.org/) (`nltk.translate.nist_score`)  
  - [mteval-14.pl](https://www.nist.gov/speech/tests/mt/) — official Perl script by NIST  

### Computational Complexity

- **Efficiency:**  
  Linear in the number of n-grams in hypotheses and references, with logarithmic cost per n-gram due to information calculation.

- **Scalability:**  
  Efficient for batch evaluation across corpora; however, performance may degrade slightly with high-order n-grams due to data sparsity in reference statistics.

## Known Limitations

- **Biases:**  
  Rare n-grams can be over-weighted if poorly estimated; n-gram distributions depend heavily on reference corpus.

- **Task Misalignment Risks:**  
  The metric assumes lexical overlap is necessary for quality, which is not always valid for creative or abstractive tasks.

- **Failure Cases:**  
  - Penalizes correct paraphrases that do not use reference vocabulary.  
  - Performance drops with fewer reference translations.  
  - Information weights may be unreliable for higher-order n-grams (N > 3) if computed only from references.

## Related Metrics

- **BLEU:** Original formulation from which NIST is derived; uses equal weights and geometric mean.  
- **METEOR:** Adds synonym and paraphrase matching for better semantic alignment.  
- **BERTScore:** Uses contextual embeddings for semantic similarity comparison.  

## Further Reading

- **Papers:**  
  - Doddington, George. "Automatic Evaluation of Machine Translation Quality Using N-gram Co-Occurrence Statistics." *HLT 2002*.  
  - [NIST MT Evaluations](https://www.nist.gov/speech/tests/mt)

- **Blogs/Tutorials:**  
  - [MachineTranslate.org summary](https://machinetranslate.org/nist)  
  - [Wikipedia Overview](https://en.wikipedia.org/wiki/NIST_(metric))

## Citation

```
@inproceedings{10.5555/1289189.1289273,  
  author = {Doddington, George},  
  title = {Automatic evaluation of machine translation quality using n-gram co-occurrence statistics},  
  year = {2002},  
  publisher = {Morgan Kaufmann Publishers Inc.},  
  address = {San Francisco, CA, USA},  
  booktitle = {Proceedings of the Second International Conference on Human Language Technology Research},  
  pages = {138–145},  
  numpages = {8},  
  location = {San Diego, California},  
  series = {HLT '02}  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 727.6015625  # in MB
    description: ClassVar[str] = "NIST is an automatic evaluation metric for machine translation that measures n-gram overlap between a system hypothesis and one or more reference translations. It extends BLEU by weighting n-grams based on their informativeness, giving greater importance to rare n-grams. Additionally, it employs a smoother brevity penalty to minimize penalization for small variations in output length."

    def __init__(self,
                 name: str = "NIST",
                 description: str = "NIST is an automatic evaluation metric for machine translation that measures n-gram overlap between a system hypothesis and one or more reference translations. It extends BLEU by weighting n-grams based on their informativeness, giving greater importance to rare n-grams. Additionally, it employs a smoother brevity penalty to minimize penalization for small variations in output length.",
                 n: int = 5,
                 **kwargs):
        super().__init__(name, description, n=n, **kwargs)
        self.default_n = n

    def _calculate_impl(self, input: str, output: str, references=None, n=None, **kwargs) -> float:
        """
        Calculate the NIST score for a single hypothesis and its references.

        Args:
            input: The source text (not used in NIST calculation).
            output: The hypothesis translation string.
            references: A list of reference translation strings.
            n: The highest n-gram order to use (overrides default).
        Returns:
            A float NIST score.
        """
        # Quick sanity checks
        if output is None or (isinstance(output, str) and output.strip() == ""):
            return 0.0
        if references is None or len(references) == 0:
            return 0.0

        # Determine the n-gram order, but ensure it does not exceed hypothesis length
        hyp_tokens = output.split() if isinstance(output, str) else output
        if len(hyp_tokens) == 0:
            return 0.0

        max_n_requested = n if n is not None else self.default_n
        max_n = min(max_n_requested, len(hyp_tokens))
        if max_n == 0:
            return 0.0

        # Tokenize references
        ref_tokens = [ref.split() if isinstance(ref, str) else ref for ref in references]

        try:
            return sentence_nist(ref_tokens, hyp_tokens, n=max_n)
        except ZeroDivisionError:
            # This happens when there is no n-gram overlap or denominator is zero
            return 0.0 