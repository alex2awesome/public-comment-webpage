---
# Metric Card for SummaQA

SummaQA is a reference-free evaluation metric for summarization based on question answering (QA). It evaluates a generated summary by measuring its ability to correctly answer cloze-style questions derived from named entities in the source document. The metric leverages a BERT-based QA model to estimate both the probability of predicted answers and their F1 score against ground-truth answers masked from the original text.

## Metric Details

### Metric Description

SummaQA evaluates the content preservation of a generated summary without requiring a reference summary. It uses a two-step process:
1. **Question Generation (QG):** Named entities in the source text are masked to create cloze-style questions.
2. **Question Answering (QA):** A pretrained BERT QA model attempts to answer these questions using the generated summary.

The metric reports two sub-scores:
- **Average Answer Probability (avg_prob):** Likelihood of the predicted answer span under the QA model.
- **Average F1 Score (avg_fscore):** Overlap between predicted and ground-truth answers.

These scores reflect the factual consistency and informativeness of the summary with respect to the source.

- **Metric Type:** Faithfulness
- **Range:** 0 to 1 (for both submetrics)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Let $Q = \{(q_i, a_i)\}_{i=1}^N$ be a set of cloze questions $q_i$ and ground-truth answers $a_i$ extracted from the source document. Let $s$ be the generated summary. For each $(q_i, a_i)$:

- The QA model predicts answer $\hat{a}_i$ and confidence $p_i$ from $s$.
- Compute F1 score between $\hat{a}_i$ and $a_i$:
  
$$
\text{F1}_i = \frac{2 \cdot \text{precision}_i \cdot \text{recall}_i}{\text{precision}_i + \text{recall}_i}
$$

- The overall metric returns:

$$
\text{avg\_prob} = \frac{1}{N} \sum _{i=1}^{N} p_i, \quad \text{avg\_fscore} = \frac{1}{N} \sum _{i=1}^{N} \text{F1}_i
$$

### Inputs and Outputs

- **Inputs:**  
  - Input document (used for generating cloze questions)  
  - Generated summary (evaluated as a source of answers to cloze questions)

- **Outputs:**  
  - Two scalar values:
    - `avg_prob`: average answer probability under the QA model
    - `avg_fscore`: average F1 score between predicted and gold answers

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Summarization

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating factual consistency and information preservation in abstractive summarization.
  - Scenarios where no reference summaries are available.
  
- **Not Recommended For:**  
  - Non-extractive or highly abstractive generation without surface-level entity mentions.
  - Tasks involving creative generation or summarization of texts with sparse named entities.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Official GitHub Repository](https://github.com/ThomasScialom/summa-qa)  
  - [SummaQA Paper on ACL Anthology](https://aclanthology.org/D19-1320/)  

### Computational Complexity

- **Efficiency:**  
  The metric requires:
  - spaCy NER for question generation (linear in text length),
  - Transformer-based QA inference per question (expensive for large $N$).
  Overall cost scales with number of entity-based questions.

- **Scalability:**  
  Suitable for small to medium batch evaluation. Preloading models and disabling persistence can reduce memory usage. Less scalable for long documents or large corpora without parallelization.

## Known Limitations

- **Biases:**  
  - Relies on named entity recognition (NER); may neglect important non-entity content.
  - Biased toward facts recoverable via NER-based questions, overlooking stylistic or inferential aspects.

- **Task Misalignment Risks:**  
  - Poor alignment with abstractive summarization models that paraphrase or omit named entities.
  - Vulnerable to hallucinated but answerable spans in generated summaries.

- **Failure Cases:**  
  - Low F1 despite semantically correct paraphrases.
  - High scores if hallucinated content coincides with correct QA patterns.

## Related Metrics

- **QAGS:** Also uses QA to evaluate factual consistency in summaries but supports sentence-level scoring and manual questions.  
- **FEQA:** Focuses on factual consistency via QA pairs and calculates cosine similarity in embedding space.  
- **BERTScore:** Measures semantic similarity but not fact retention.  

## Further Reading

- **Papers:**  
  - [Scialom et al. (2019) "Answers Unite! Unsupervised Metrics for Reinforced Summarization Models"](https://aclanthology.org/D19-1320/)  
  - [arXiv version](https://arxiv.org/abs/1909.01610)

- **Blogs/Tutorials:**  
  - [SummaQA GitHub Quickstart Guide](https://github.com/ThomasScialom/summa-qa#quickstart)

## Citation

```
@inproceedings{scialom-etal-2019-answers,
    title = "Answers Unite! Unsupervised Metrics for Reinforced Summarization Models",
    author = "Scialom, Thomas  and
      Lamprier, Sylvain  and
      Piwowarski, Benjamin  and
      Staiano, Jacopo",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1320/",
    doi = "10.18653/v1/D19-1320",
    pages = "3246--3256",
    abstract = "Abstractive summarization approaches based on Reinforcement Learning (RL) have recently been proposed to overcome classical likelihood maximization. RL enables to consider complex, possibly non differentiable, metrics that globally assess the quality and relevance of the generated outputs. ROUGE, the most used summarization metric, is known to suffer from bias towards lexical similarity as well as from sub-optimal accounting for fluency and readability of the generated abstracts. We thus explore and propose alternative evaluation measures: the reported human-evaluation analysis shows that the proposed metrics, based on Question Answering, favorably compare to ROUGE {--} with the additional property of not requiring reference summaries. Training a RL-based model on these metrics leads to improvements (both in terms of human or automated metrics) over current approaches that use ROUGE as reward."
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu