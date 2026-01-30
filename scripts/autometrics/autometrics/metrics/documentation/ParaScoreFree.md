---
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
- **Contact:** mryan0@stanford.edu