---
# Metric Card for Levenshtein Ratio

Levenshtein Ratio is a normalized similarity metric that computes the relative similarity between two sequences by evaluating the minimum number of insertions and deletions required to transform one sequence into the other. The result is expressed as a value between 0 and 1, where 1 indicates identical sequences.

## Metric Details

### Metric Description

Levenshtein Ratio calculates a normalized indel similarity score. It uses the indel distance (i.e., the minimum number of insertions and deletions required to change one sequence into the other) and normalizes this value by the total length of both sequences. This provides a score in the range [0, 1], where higher scores indicate greater similarity.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let $d(s_1, s_2)$ be the indel distance between sequences $s_1$ and $s_2$. The Levenshtein Ratio is defined as:

$$
\text{Levenshtein Ratio} = 1 - \frac{d(s_1, s_2)}{|s_1| + |s_2|}
$$

where $|s_1|$ and $|s_2|$ denote the lengths of the sequences $s_1$ and $s_2$, respectively.

### Inputs and Outputs

- **Inputs:**  
  - Two sequences (e.g., strings or lists of hashable elements) to compare.
  - Optional parameters:
    - `processor`: A callable to preprocess the inputs (default is None).
    - `score_cutoff`: A threshold for early termination, specified as a float between 0 and 1 (default is 0, which deactivates this behavior).
  
- **Outputs:**  
  - A float representing the normalized similarity between the two sequences, ranging from 0 (completely different) to 1 (identical).

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Spell Checking, Error Correction, Approximate String Matching, Quality Evaluation of Generated Text

### Applicability and Limitations

- **Best Suited For:**  
  - Situations requiring a normalized measure of string similarity.
  - Applications where the relative similarity (rather than the absolute number of edit operations) is more informative.
  
- **Not Recommended For:**  
  - Scenarios where semantic similarity is crucial, as the metric only considers literal character differences.
  - Tasks with high variability in acceptable outputs, such as creative text generation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Levenshtein Python module](https://rapidfuzz.github.io/Levenshtein/index.html)

### Computational Complexity

- **Efficiency:**  
  - The computation typically requires $O(m \times n)$ time, where $m$ and $n$ are the lengths of the input sequences.
  
- **Scalability:**  
  - Memory optimizations (e.g., using a two-row dynamic programming approach) allow the metric to scale for moderately sized sequences.

## Known Limitations

- **Biases:**  
  - The metric is sensitive only to literal character differences and does not account for semantic or contextual similarity.
  
- **Task Misalignment Risks:**  
  - May yield low similarity scores for strings that are semantically similar but differ significantly in character order or structure.
  
- **Failure Cases:**  
  - Not effective for comparing sequences where insertions and deletions are less indicative of overall similarity (e.g., when substitutions are more prevalent).

## Related Metrics

- **Levenshtein Distance:** The non-normalized version measuring the absolute number of edit operations.
- **Damerau-Levenshtein Ratio:** A variant that also considers transpositions.
- **BERTScore:** Evaluates semantic similarity using contextual embeddings.

## Further Reading

- **Papers:**  
  - V. I. Levenshtein, "Binary Codes with Correction of Deletions, Insertions, and Substitutions of Symbols", Doklady of the USSR Academy of Sciences, 1965. [Available here](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=31411&option_lang=rus)
  
- **Blogs/Tutorials:**  
  - [Levenshtein Python Module Documentation](https://rapidfuzz.github.io/Levenshtein/index.html)

## Citation

```
@article{levenshtein1965binary,
  author       = {В.~И.~Левенштейн},
  title        = {Двоичные коды с исправлением выпадений, вставок и замещений символов},
  journal      = {Доклады Академии Наук СССР},
  year         = {1965},
  volume       = {163},
  number       = {4},
  pages        = {845--848},
  url          = {http://mi.mathnet.ru/dan31411},
  note         = {English translation: \emph{Binary Codes Capable of Correcting Deletions, Insertions, and Reversals}, \emph{Soviet Physics Doklady}, vol. 10, pp. 707, 1966.}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu