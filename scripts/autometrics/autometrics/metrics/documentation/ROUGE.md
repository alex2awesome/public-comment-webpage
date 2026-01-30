---
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

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  