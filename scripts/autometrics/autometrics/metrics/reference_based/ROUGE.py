from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import ReferenceBasedMultiMetric
import json
from rouge_score import rouge_scorer
from typing import ClassVar

def parse_rouge_dict(rouge_dict):
    """
    Parse the ROUGE dictionary to extract scores.
    """
    scores = []
    scores.append(rouge_dict['rouge1'].precision)
    scores.append(rouge_dict['rouge1'].recall)
    scores.append(rouge_dict['rouge1'].fmeasure)
    scores.append(rouge_dict['rouge2'].precision)
    scores.append(rouge_dict['rouge2'].recall)
    scores.append(rouge_dict['rouge2'].fmeasure)
    scores.append(rouge_dict['rougeL'].precision)
    scores.append(rouge_dict['rougeL'].recall)
    scores.append(rouge_dict['rougeL'].fmeasure)
    scores.append(rouge_dict['rougeLsum'].precision)
    scores.append(rouge_dict['rougeLsum'].recall)
    scores.append(rouge_dict['rougeLsum'].fmeasure)

    return scores

class ROUGE(ReferenceBasedMultiMetric):
    """---
# Metric Card for ROUGE (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-LSum)

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a widely used evaluation metric for text summarization, machine translation, and text generation tasks. It measures the overlap between an automatically generated text and reference texts using various methods such as **n-gram overlap (ROUGE-1, ROUGE-2), longest common subsequence (ROUGE-L), and summary-level longest common subsequence (ROUGE-LSum)**.

The **rouge-score** Python package provides a native implementation that replicates results from the original Perl-based ROUGE package. It supports **text normalization, Porter stemming, and confidence interval calculation** while omitting stopword removal due to licensing restrictions.

## Metric Details

### Metric Description

ROUGE evaluates generated text by comparing it with human-written references. The key variants included in this implementation are:

- **ROUGE-1**: Measures unigram (single-word) overlap between candidate and reference texts.
- **ROUGE-2**: Measures bigram (two-word sequence) overlap.
- **ROUGE-L**: Measures the longest common subsequence (LCS) between candidate and reference texts, capturing sentence-level structure similarity.
- **ROUGE-LSum**: A summary-level variant of ROUGE-L, treating newlines as sentence boundaries and computing LCS across sentence pairs.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

#### ROUGE-N (N-gram Overlap)

For an n-gram of length $n$:

$$
\text{ROUGE-N} = \frac{\sum _{S \in \text{Reference Summaries}} \sum _{\text{gram} _{n} \in S} \text{Count} _{\text{match}}(\text{gram} _{n})}
{\sum _{S \in \text{Reference Summaries}} \sum _{\text{gram} _{n} \in S} \text{Count}(\text{gram} _{n})}
$$

where $\text{Count} _{\text{match}}(\text{gram} _{n})$ is the number of n-grams appearing in both the candidate and reference summaries.

## **ROUGE-L (Longest Common Subsequence)**

ROUGE-L evaluates the longest common subsequence (LCS) between the candidate and reference texts. The LCS captures sentence structure similarity by considering word order while allowing gaps.

Given a candidate summary $X$ of length $m$ and a reference summary $Y$ of length $n$, let $LCS(X, Y)$ denote the length of their longest common subsequence.

### **Recall ($R_{LCS}$):**

$$
R_{LCS} = \frac{LCS(X, Y)}{n}
$$

Measures the proportion of the reference summary captured by the candidate summary.

### **Precision ($P_{LCS}$):**

$$
P_{LCS} = \frac{LCS(X, Y)}{m}
$$

Measures the proportion of the candidate summary that is part of the LCS.

### **F-measure ($F_{LCS}$):**

$$
F_{LCS} = \frac{(1 + \beta^2) \cdot R_{LCS} \cdot P_{LCS}}{R_{LCS} + \beta^2 \cdot P_{LCS}}
$$

Where $\beta$ determines the relative weight of recall versus precision. A common choice is $\beta = 1$, giving equal weight to both.

#### ROUGE-LSum (Summary-Level LCS)

ROUGE-LSum extends ROUGE-L to the summary level by treating newlines as sentence boundaries. Instead of computing a single LCS over the entire text, it:

1. Splits the candidate and reference summaries into sentences.
2. Computes LCS for each candidate-reference sentence pair.
3. Aggregates results to produce an overall ROUGE-LSum score.

### Inputs and Outputs

- **Inputs:**  
  - Generated text (candidate summary)  
  - Reference text(s) (human-written summary)

- **Outputs:**  
  - Scalar ROUGE score (range: 0 to 1), providing recall, precision, and F1-score.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Summarization, Machine Translation, Paraphrasing, Data-to-Text Generation

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating text generation tasks where lexical similarity is a reliable proxy for quality.
  - Comparing multiple summarization systems against a reference standard.

- **Not Recommended For:**  
  - Evaluating abstractiveness, coherence, fluency, or factual consistency.
  - Tasks where paraphrasing or rewording is expected, as ROUGE penalizes non-exact matches.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Google Research ROUGE](https://github.com/google-research/google-research/tree/master/rouge)
  - [Hugging Face `evaluate`](https://huggingface.co/docs/evaluate)
  - [Python `rouge_score` package](https://pypi.org/project/rouge-score/)

### Computational Complexity

- **Efficiency:**  
  - ROUGE-N complexity is $O(n \cdot m)$ for n-gram counting, where $n$ is the candidate text length and $m$ is the reference text length.
  - ROUGE-L requires LCS computation, which is $O(n \cdot m)$ using dynamic programming.

- **Scalability:**  
  - ROUGE scales well to large datasets but can be computationally intensive when multiple reference texts are used.

## Known Limitations

- **Biases:**  
  - Prefers texts with high lexical overlap, penalizing valid paraphrases.
  - Highly sensitive to the number and quality of reference summaries.

- **Task Misalignment Risks:**  
  - Cannot capture meaning beyond exact n-gram matches.
  - Does not account for factual correctness or grammaticality.

- **Failure Cases:**  
  - Overestimates quality for summaries with high recall but poor readability.
  - Struggles with abstractive summarization, which may use different wording.

## Related Metrics

- **BLEU:** A precision-based alternative used in machine translation.  
- **METEOR:** Incorporates synonym matching and paraphrase detection.  
- **BERTScore:** Uses contextual embeddings for semantic similarity.  

## Further Reading

- **Papers:**  
  - [Lin, 2004: ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013)  
  - [Ganesan, 2018: ROUGE 2.0 - Improved Evaluation Measures](https://arxiv.org/abs/1803.01937)  

- **Blogs/Tutorials:**  
  - [ROUGE How-To](http://kavita-ganesan.com/rouge-howto)  
  - [ROUGE in Hugging Face](https://huggingface.co/docs/evaluate)  

## Metric Card Authors

## Citation

  ```
  @inproceedings{lin-2004-rouge,
      title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
      author = "Lin, Chin-Yew",
      booktitle = "Text Summarization Branches Out",
      month = jul,
      year = "2004",
      address = "Barcelona, Spain",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/W04-1013/",
      pages = "74--81"
  }
  ```

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  """

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 726.48828125  # in MB
    description: ClassVar[str] = "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a widely used evaluation metric for text summarization, machine translation, and text generation tasks. It measures the overlap between an automatically generated text and reference texts using various methods such as **n-gram overlap (ROUGE-1, ROUGE-2), longest common subsequence (ROUGE-L), and summary-level longest common subsequence (ROUGE-LSum)**."

    def __init__(self, **kwargs):
        name = "ROUGE"
        description = "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a widely used evaluation metric for text summarization, machine translation, and text generation tasks. It measures the overlap between an automatically generated text and reference texts using various methods such as **n-gram overlap (ROUGE-1, ROUGE-2), longest common subsequence (ROUGE-L), and summary-level longest common subsequence (ROUGE-LSum)**."

        submetrics = ["1", "2", "L", "Lsum"]
        measures = ['p', 'r', 'f1']

        self.combinations = [ 
            f"ROUGE-{submetric}-{measure}" for measure in measures for submetric in submetrics
        ] # ROUGE- + [1-p, 1-r, 1-f1, 2-p, 2-r, 2-f1, L-p, L-r, L-f1, Lsum-p, Lsum-r, Lsum-f1]
        
        super().__init__(name, description, self.combinations, **kwargs)
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Calculate ROUGE scores for the given input and output.
        """
        if references is None:
            references = []

        assert len(references) > 0, "ROUGE requires at least one reference"
        assert type(references) == list, "ROUGE requires a list of references"
        assert type(references[0]) == str, "ROUGE requires a list of strings"

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_stemmer=True)
        
        all_scores = [[] for _ in range(len(self.combinations))]

        for reference in references:
            scores = scorer.score(reference, output)
            parsed_scores = parse_rouge_dict(scores)

            for i, score in enumerate(parsed_scores):
                all_scores[i].append(score)

        # Calculate the max scores
        max_scores = [max(scores) for scores in all_scores]

        return max_scores

if __name__ == "__main__":
    # Example usage
    rouge = ROUGE()
    input = None
    output = "The cat is sitting on the mat."
    references = ["The cat is on the mat.", "A cat sits on the mat."]
    scores = rouge.calculate(input, output, references)
    print("ROUGE scores:", scores)