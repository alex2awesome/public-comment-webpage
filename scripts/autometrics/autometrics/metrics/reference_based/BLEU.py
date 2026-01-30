from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from sacrebleu.metrics import BLEU as bleu
from typing import ClassVar

class BLEU(ReferenceBasedMetric):
    """---
# Metric Card for BLEU

BLEU (Bilingual Evaluation Understudy) is a widely used metric for evaluating the quality of text generated in tasks like machine translation and summarization. It measures the overlap of n-grams between a generated text and one or more reference texts, with a brevity penalty to penalize overly short translations. SacreBLEU, a modern implementation, ensures reproducibility and standardization of BLEU scores across research.

## Metric Details

### Metric Description

BLEU evaluates the quality of text generation by comparing n-grams in the generated output with those in one or more reference texts. It computes modified precision for n-grams and combines scores using a geometric mean, with a brevity penalty to ensure the length of the generated text matches that of the references. Higher BLEU scores indicate closer similarity to the references.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

$$
\text{BLEU} = \text{BP} \times \exp \left( \sum_{n=1}^N w_n \log p_n \right)
$$

where:
- $\text{BP} = \min(1, e^{1 - r/c})$ is the brevity penalty,
- $r$ is the effective reference length (based on the closest matching reference length for each sentence),
- $c$ is the candidate translation length,
- $p_n$ is the modified precision for n-grams of length $n$,
- $w_n$ are weights for each n-gram (commonly uniform, $w_n = \frac{1}{N}$).

### Inputs and Outputs

- **Inputs:**  
- Generated text (candidate translation)  
- Reference text(s) (gold-standard translations)  

- **Outputs:**  
- Scalar BLEU score (range: 0 to 1)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Machine Translation, Summarization, Data-to-Text Generation

### Applicability and Limitations

- **Best Suited For:**  
Structured tasks with a clear correspondence between generated and reference texts, such as translation or summarization.

- **Not Recommended For:**  
Open-ended or creative generation tasks where diversity or semantic similarity matters more than lexical overlap (e.g., storytelling, dialogue).

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
- [SacreBLEU](https://github.com/mjpost/sacrebleu) (robust, standard implementation)
- [NLTK](https://www.nltk.org/api/nltk.translate.html) (basic Python implementation)
- [Hugging Face `evaluate`](https://huggingface.co/docs/evaluate) (integrated metric framework)

### Computational Complexity

- **Efficiency:**  
BLEU is computationally efficient, requiring $O(n \times m)$ operations for $n$-gram matching where $n$ is the number of words in the candidate text and $m$ is the number of reference words. SacreBLEU optimizes tokenization and scoring, making it highly suitable for large-scale evaluations.

- **Scalability:**  
BLEU scales well across datasets of varying sizes due to its simple design. SacreBLEU further supports evaluation with multiple references, diverse tokenization schemes, and language-specific preprocessing, making it adaptable to diverse evaluation setups.

## Known Limitations

- **Biases:**  
- BLEU penalizes valid paraphrases or semantically equivalent outputs that do not match reference n-grams exactly.  
- The brevity penalty can overly penalize valid shorter outputs, particularly for tasks where shorter text may be acceptable or even preferred (e.g., summarization).  

- **Task Misalignment Risks:**  
- BLEU is not designed for evaluating tasks with high diversity in acceptable outputs (e.g., open-ended dialogue).  
- Scores depend on the quality and number of references; fewer or inconsistent references can lead to misleading evaluations.

- **Failure Cases:**  
- BLEU struggles to capture semantic adequacy beyond lexical similarity. For instance, it cannot identify whether a translation preserves the meaning of the original sentence if word choices diverge significantly.

## Related Metrics

- **ROUGE:** Often used for summarization tasks, emphasizing recall over precision.  
- **METEOR:** Incorporates synonym matching for better semantic alignment.  
- **BERTScore:** Uses contextual embeddings for semantic similarity.  

## Further Reading

- **Papers:**  
- [Original BLEU Paper (Papineni et al., 2002)](https://www.aclweb.org/anthology/P02-1040)  
- [SacreBLEU: A Call for Clarity in Reporting BLEU Scores (Post, 2018)](https://www.aclweb.org/anthology/W18-6319)

- **Blogs/Tutorials:**  
- [Understanding BLEU](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)  
- [SacreBLEU Documentation](https://github.com/mjpost/sacrebleu)

## Citation

```
@inproceedings{papineni-etal-2002-bleu,
    title = "{B}leu: a Method for Automatic Evaluation of Machine Translation",
    author = "Papineni, Kishore  and
      Roukos, Salim  and
      Ward, Todd  and
      Zhu, Wei-Jing",
    editor = "Isabelle, Pierre  and
      Charniak, Eugene  and
      Lin, Dekang",
    booktitle = "Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2002",
    address = "Philadelphia, Pennsylvania, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P02-1040/",
    doi = "10.3115/1073083.1073135",
    pages = "311--318"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu
    """

    # BLEU is fast enough without caching
    DEFAULT_USE_CACHE = False

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 729.32514375  # in MB
    description: ClassVar[str] = "BLEU (Bilingual Evaluation Understudy) is a widely used metric for evaluating the quality of text generated in tasks like machine translation and summarization. It measures the overlap of n-grams between a generated text and one or more reference texts, with a brevity penalty to penalize overly short translations. SacreBLEU, a modern implementation, ensures reproducibility and standardization of BLEU scores across research."

    def __init__(self,
                 name: str = "BLEU",
                 description: str = "BLEU (Bilingual Evaluation Understudy) is a widely used metric for evaluating the quality of text generated in tasks like machine translation and summarization. It measures the overlap of n-grams between a generated text and one or more reference texts, with a brevity penalty to penalize overly short translations. SacreBLEU, a modern implementation, ensures reproducibility and standardization of BLEU scores across research.",
                 **kwargs):
        super().__init__(name, description, **kwargs)
        self.metric = bleu()

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Actual implementation of the BLEU metric
        """
        if references is None:
            references = []

        output = [output]
        references = [[r] for r in references]

        return self.metric.corpus_score(output, references).score
