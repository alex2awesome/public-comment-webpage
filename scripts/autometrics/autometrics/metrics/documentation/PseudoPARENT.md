---
# Metric Card for PseudoPARENT

**PseudoPARENT** is a *custom adaptation* of the PARENT metric for evaluating text generation from structured inputs. Unlike the original PARENT metric, which operates on explicit tables of (attribute, value) or (head, relation, tail) triples, **PseudoPARENT treats the input text as a flat sequence of tokens, using these tokens as "pseudo-table values"**. This allows PseudoPARENT to simulate input-aware evaluation without requiring a structured table format. It effectively acts as a hybrid between BLEU-style surface matching and PARENT-style factual grounding.

## Metric Details

### Metric Description

PseudoPARENT is a reference-based and input-aware metric designed to evaluate whether generated outputs faithfully reflect the content of an input text. It adapts the structure of the original PARENT metric but with a crucial change: **the structured table is replaced with a bag-of-tokens from the input string**, which are treated as table values. The entailment functions (used to determine how well predicted content aligns with the source) are redefined as simple token overlaps with this pseudo-table.

This design allows PseudoPARENT to serve as an *input-aware variant of BLEU*, incorporating both:
- **Precision**: How well the prediction matches the reference, with token-based entailment fallback to the input.
- **Recall**: How well the prediction mentions content from the reference and input combined.

Multiple references are supported, and the final score is based on the maximum F1 across references.

- **Metric Type:** Faithfulness
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let:
- $P$ be the prediction (generated output),
- $R_i$ be each reference (tokenized),
- $I$ be the tokenized input text (used as a pseudo-table),
- $O(ngram, I)$ be the overlap probability of an $n$-gram with the input,
- $M(tok, P)$ be the mention probability of an input token in the prediction,
- $N$ be the maximum $n$-gram order (default: 4).

Then for each $(P, \{R_i\}, I)$ triple:

1. Compute precision:
$$
\text{Prec}_n = \frac{\sum_{ng \in P_n} c(ng) \cdot \left[\min\left(1, \frac{c_{R_i}(ng)}{c(ng)}\right) + \left(1 - \min\left(1, \frac{c_{R_i}(ng)}{c(ng)}\right)\right) \cdot O(ng, I)\right]}{\sum_{ng \in P_n} c(ng)}
$$

2. Compute recall (reference and input):
$$
\text{RefRec}_n = \frac{\sum_{ng \in R_i^n} c(ng) \cdot O(ng, I) \cdot \min\left(1, \frac{c_P(ng)}{c(ng)}\right)}{\sum_{ng \in R_i^n} c(ng) \cdot O(ng, I)}
$$

$$
\text{InputRec} = \frac{|\{tok \in I : tok \in P\}|}{|I|}
$$

3. Combine:
$$
\text{CPrec} = \left( \prod_{n=1}^N \text{Prec}_n \right)^{\frac{1}{N}}, \quad
\text{RefRec} = \left( \prod_{n=1}^N \text{RefRec}_n \right)^{\frac{1}{N}}, \quad
\text{CRec} = \text{RefRec}^{(1 - \lambda)} \cdot \text{InputRec}^\lambda
$$

4. Final F-score:
$$
F = \frac{2 \cdot \text{CPrec} \cdot \text{CRec}}{\text{CPrec} + \text{CRec}}
$$

The final score is the maximum $F$ across all references $R_i$.

### Inputs and Outputs

- **Inputs:**  
  - Generated text (candidate)  
  - One or more reference texts  
  - Input text (used as source "table")  

- **Outputs:**  
  - Scalar score representing F1 (can also return precision and recall)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Data-to-Text Generation, Summarization, Paraphrasing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluation setups where input text is available and should influence output generation, especially when explicit structured tables are unavailable.

- **Not Recommended For:**  
  Tasks lacking clear input-output alignment or where structured data is essential to proper factual grounding (e.g., original PARENT use cases). Not suitable for purely reference-free evaluation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Based on original [google-research/language](https://github.com/google-research/language/tree/master/language/table_text_eval)

### Computational Complexity

- **Efficiency:**  
  Moderate. Involves $n$-gram counting and token set operations. Significantly faster than original PARENT due to simplified overlap-based entailment and multiprocessing support.

- **Scalability:**  
  Scales well to medium-size datasets. Multiprocessing improves batch computation efficiency.

## Known Limitations

- **Biases:**  
  Penalizes synonyms and paraphrases that are not token-identical to input. Precision fallback to input overlap may reward hallucinated content that happens to share tokens with the input.

- **Task Misalignment Risks:**  
  Designed for input-sensitive evaluation; not appropriate for tasks where the input is irrelevant or open-ended.

- **Failure Cases:**  
  Performs poorly when input is noisy, overly long, or semantically distant from the output. Token-level matching can be brittle under complex morphology or low-resource tokenization.

## Related Metrics

- **PARENT:** Original version with structured table inputs and entailment probabilities.
- **BLEU:** Pure $n$-gram overlap; lacks input-awareness.
- **ROUGE:** Recall-based $n$-gram overlap.
- **BERTScore:** Embedding-based similarity; can be extended to input-aware settings.

## Further Reading

- **Papers:**  
  - [Original PARENT Paper (ACL 2019)](https://aclanthology.org/P19-1483/)

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{dhingra-etal-2019-handling,
    title = "Handling Divergent Reference Texts when Evaluating Table-to-Text Generation",
    author = "Dhingra, Bhuwan  and
      Faruqui, Manaal  and
      Parikh, Ankur  and
      Chang, Ming-Wei  and
      Das, Dipanjan  and
      Cohen, William",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\`a}rquez, Llu{\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1483/",
    doi = "10.18653/v1/P19-1483",
    pages = "4884--4895"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu