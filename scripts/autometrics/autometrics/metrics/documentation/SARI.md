---
# Metric Card for SARI

SARI (Sentence Adaptation for Readability Index) is a metric designed to evaluate the quality of text simplification by comparing the system-generated output against both the original input text and reference simplified texts. It measures how well words are added, deleted, and kept appropriately, rewarding edits that enhance readability while preserving meaning.

## Metric Details

### Metric Description

SARI evaluates text simplification by considering three types of operations: additions, deletions, and retention of n-grams. It computes precision and recall for these operations by comparing the system output to both the input and the reference simplified texts. SARI is particularly suited for simplification tasks as it explicitly rewards edits that improve readability while maintaining semantic correctness.

- **Metric Type:** Surface-Level Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** Yes  

### Formal Definition

SARI is computed as the arithmetic mean of F-scores for the addition and retention operations, along with the precision of the deletion operation:

$$
SARI = \frac{1}{3}(F_{add} + F_{keep} + P_{del})
$$

Where:
- $F_{add}$: F-score for addition operations
- $F_{keep}$: F-score for keeping relevant text
- $P_{del}$: Precision for deletion operations  

Each F-score or precision is computed based on the comparison of n-grams in the input, system output, and references.

### Inputs and Outputs

- **Inputs:**  
  - Source text (original, complex input text)  
  - Candidate text (simplified text from the system)  
  - Reference texts (simplified human-created texts)  

- **Outputs:**  
  - Scalar SARI score (range: 0 to 1)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:**  
  - Text Simplification  

### Applicability and Limitations

- **Best Suited For:**  
  - Text simplification tasks where changes to the text, such as paraphrasing, deletions, or additions, are expected to enhance readability.  

- **Not Recommended For:**  
  - Open-ended or creative text generation tasks where diversity and semantic similarity matter more than lexical transformation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [SARI Implementation in LENS Repository](https://github.com/Yao-Dou/LENS/blob/master/experiments/meta_evaluation/metrics/sari.py)  

### Computational Complexity

- **Efficiency:**  
  SARI is computationally efficient, with complexity similar to BLEU, as it involves n-gram extraction and comparison.  

- **Scalability:**  
  SARI scales well across datasets with multiple references, leveraging n-gram matching for simplicity evaluation.

## Known Limitations

- **Biases:**  
  - SARI may over-penalize outputs that do not align well with reference texts, particularly in cases where valid simplifications are not covered by references.  

- **Task Misalignment Risks:**  
  - SARI is unsuitable for tasks that emphasize semantic similarity over structural changes, such as summarization or machine translation.  

- **Failure Cases:**  
  - It can struggle with highly creative or diverse simplifications where multiple equally valid outputs are possible.

## Related Metrics

- **BLEU:** Measures surface similarity but does not compare outputs with the input text.  
- **FKBLEU:** Combines BLEU with the Flesch-Kincaid readability metric for simplification tasks.  
- **ROUGE:** Suitable for summarization but less relevant for simplification.  

## Further Reading

- **Papers:**  
  - [Optimizing Statistical Machine Translation for Text Simplification (Xu et al., 2016)](https://github.com/cocoxu/simplification/)

## Citation

```
@article{Xu-EtAl:2016:TACL,
  author = {Wei Xu and Courtney Napoles and Ellie Pavlick and Quanze Chen and Chris Callison-Burch},
  title = {Optimizing Statistical Machine Translation for Text Simplification},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {4},
  year = {2016},
  url = {https://cocoxu.github.io/publications/tacl2016-smt-simplification.pdf},
  pages = {401--415}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu