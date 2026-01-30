---
# Metric Card for Levenshtein Distance

Levenshtein Distance measures the minimum number of single-character edits—insertions, deletions, or substitutions—required to transform one sequence into another. It is a fundamental metric in text processing, error correction, and approximate string matching.

## Metric Details

### Metric Description

Levenshtein Distance calculates the minimal cost of edit operations needed to convert one string into another. The computation is typically performed using a dynamic programming approach that considers insertions, deletions, and substitutions. Users can optionally assign custom weights to each operation (defaulting to 1 for all), allowing the metric to adapt to different application needs.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to $\infty$ (practically 0 to $\max(|s_1|, |s_2|)$ with unit costs)
- **Higher is Better?:** No (lower values indicate greater similarity)
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Given two sequences $s_1$ and $s_2$, and weights $w_{ins}$, $w_{del}$, $w_{sub}$, the Levenshtein Distance $D(i, j)$ is defined as:

$$
D(i, 0) = i \cdot w_{del}, \quad D(0, j) = j \cdot w_{ins}
$$

$$
D(i, j) = \min \begin{cases}
D(i-1, j) + w_{del}, \\
D(i, j-1) + w_{ins}, \\
D(i-1, j-1) + \begin{cases}
0, & \text{if } s_1[i] = s_2[j] \\
w_{sub}, & \text{otherwise}
\end{cases}
\end{cases}
$$

where $1 \leq i \leq |s_1|$, $1 \leq j \leq |s_2|$, and typically $w_{ins} = w_{del} = w_{sub} = 1$.

### Inputs and Outputs

- **Inputs:**  
  - Two sequences (e.g., strings or lists of hashable items) to compare.
  - Optional parameters:
    - `weights`: A tuple $(w_{ins}, w_{del}, w_{sub})$ specifying custom costs.
    - `processor`: A callable to preprocess the inputs.
    - `score_cutoff`: A threshold to limit computation for large distances.
  
- **Outputs:**  
  - An integer representing the computed Levenshtein Distance between the two inputs.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Machine Translation, Summarization, Paraphrasing, Spell Checking, Error Correction

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating character-level similarity between two sequences.
  - Applications in spell checking, optical character recognition, and error correction where precise, literal differences matter.
  
- **Not Recommended For:**  
  - Tasks requiring semantic or context-aware similarity measures (e.g., creative text generation, open-ended dialogue).
  - Scenarios where reordering or paraphrasing plays a significant role in perceived similarity.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Levenshtein Python module](https://rapidfuzz.github.io/Levenshtein/index.html) – A highly optimized C-based implementation for computing Levenshtein Distance.

### Computational Complexity

- **Efficiency:**  
  - The standard dynamic programming solution operates in $O(m \times n)$ time, where $m$ and $n$ are the lengths of the two input sequences.
  
- **Scalability:**  
  - Memory usage can be optimized to $O(\min(m, n))$ using a two-row technique, making it feasible for moderately sized inputs. However, performance may become an issue for extremely long sequences.

## Known Limitations

- **Biases:**  
  - Focuses solely on literal character differences and does not account for semantic or contextual similarity.
  
- **Task Misalignment Risks:**  
  - May not correlate with human judgments in cases where meaning is preserved despite significant character-level differences.
  
- **Failure Cases:**  
  - In tasks with high variability in acceptable outputs (e.g., creative generation), the metric may yield misleadingly high distances despite semantically similar content.

## Related Metrics

- **Damerau-Levenshtein Distance:** Considers transpositions in addition to insertions, deletions, and substitutions.
- **Hamming Distance:** Measures the number of differing characters for sequences of equal length.
- **ROUGE and BLEU:** Surface-level similarity metrics commonly used in text generation evaluation.
- **BERTScore:** Evaluates semantic similarity using contextual embeddings.

## Further Reading

- **Papers:**  
  - V. I. Levenshtein, "Binary Codes with Correction of Deletions, Insertions, and Substitutions of Symbols", Doklady of the USSR Academy of Sciences, 1965. [Available here](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=31411)
  
- **Blogs/Tutorials:**  
  - [Levenshtein Distance Documentation](https://rapidfuzz.github.io/Levenshtein/index.html)

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