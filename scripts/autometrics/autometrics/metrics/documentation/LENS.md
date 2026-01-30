---
# Metric Card for LENS

LENS (Learnable Evaluation Metric for Text Simplification) is a reference-based metric designed specifically to evaluate system outputs in the task of text simplification. It aligns with human judgments more closely than prior metrics by learning from human ratings using a mixture-of-experts (MoE) model, which captures multiple aspects of simplification quality, such as grammaticality, meaning preservation, and simplicity. LENS can be rescaled between 0 and 100 for interpretability.

## Metric Details

### Metric Description

LENS evaluates text simplification quality by comparing a system-generated simplification against both the complex source sentence and one or more human-written simplifications (references). It is trained to regress toward average human judgments across three dimensions: grammaticality, meaning preservation, and simplicity. 

To capture these aspects, LENS uses a mixture-of-experts model built atop sentence-level and word-level representations from a pre-trained encoder (T5 encoder). Each expert corresponds to a latent factor presumed to model a subset of simplification phenomena. LENS is trained on human-annotated ratings from multiple datasets, and the resulting model provides a scalar score aligned with holistic simplification quality.

- **Metric Type:** Semantic Similarity
- **Range:** $\mathbb{R}$ (rescaled to [0, 100] for interpretability)
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Given a source sentence $C$, a system output simplification $S$, and one or more reference simplifications $R = \{r_1, \dots, r_n\}$, the LENS score is computed as follows:

1. Encode the triplet $(C, S, R)$ using the T5 encoder to obtain sentence-level and word-level embeddings.
2. Pass these embeddings through $K$ expert scoring heads, each of which outputs a scalar.
3. Use a gating network to produce weights $w_1, \dots, w_K$ over the experts based on the input.
4. Compute the final score as the weighted combination of expert predictions:

$$
\text{LENS}(C, S, R) = \sum _{k=1}^K w_k \cdot f_k(C, S, R)
$$

where $f_k$ is the $k$-th expert head’s output.

### Inputs and Outputs

- **Inputs:**  
  - Complex sentence (source input)  
  - Simplified sentence (system output)  
  - Reference simplifications (1 or more)

- **Outputs:**  
  - Scalar LENS score, either in original form (unbounded real number) or rescaled between 0 and 100.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Text Simplification

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating English text simplification outputs, particularly when references are available and multiple quality dimensions (e.g., fluency, meaning, simplicity) are relevant.

- **Not Recommended For:**  
  Tasks that are not simplification-specific (e.g., translation, paraphrasing) or that lack appropriate reference simplifications. LENS is not designed for creative generation or tasks involving high lexical diversity.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [LENS GitHub](https://github.com/davidheineman/lens) (official implementation)  
  - [Hugging Face Model Hub - davidheineman/lens](https://huggingface.co/davidheineman/lens)  

### Computational Complexity

- **Efficiency:**  
  Moderate. Inference uses a pre-trained encoder and multiple expert heads, with computational cost comparable to standard encoder-forward passes in T5.

- **Scalability:**  
  Scales adequately with batching; suitable for GPU-based batched evaluation but may be slower than traditional lexical metrics.

## Known Limitations

- LENS was trained on English simplification datasets and may not generalize to other languages without retraining.
- It requires both source and reference inputs, limiting its use in reference-free or source-free settings.

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  Not suitable for tasks like summarization or paraphrasing without retraining or adaptation.

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- **SARI:** A lexical overlap-based metric for simplification, focusing on additions, deletions, and retention.
- **BLEU/ROUGE:** Often used but poorly aligned with human judgments in simplification tasks.
- **BERTScore:** Captures semantic similarity but is not simplification-specific.
- **QuestEval:** General-purpose learned evaluation, not optimized for simplification.

## Further Reading

- **Papers:**  
  - [LENS: A Learnable Evaluation Metric for Text Simplification (Maddela et al., 2023)](https://aclanthology.org/2023.acl-long.905)

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{maddela-etal-2023-lens,
  title = "{LENS}: A Learnable Evaluation Metric for Text Simplification",
  author = "Maddela, Mounica  and
    Dou, Yao  and
    Heineman, David  and
    Xu, Wei",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2023",
  address = "Toronto, Canada",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.acl-long.905",
  doi = "10.18653/v1/2023.acl-long.905",
  pages = "16383--16408",
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu