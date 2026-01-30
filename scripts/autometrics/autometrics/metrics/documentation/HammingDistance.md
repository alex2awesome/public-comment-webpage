---
# Metric Card for Hamming Distance

Hamming Distance measures the number of positions at which two equal-length sequences differ. It is a fundamental metric in coding theory and information theory, commonly used for error detection and correction in digital communications.

## Metric Details

### Metric Description

Hamming Distance calculates the number of substitutions required to change one sequence into the other, or equivalently, the number of positions where the corresponding symbols differ. This metric is strictly defined for sequences of equal length.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to $n$ (where $n$ is the length of the sequences)
- **Higher is Better?:** No (lower values indicate greater similarity)
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

For two sequences $s$ and $t$ of equal length $n$, the Hamming Distance is defined as:

$$
H(s, t) = \sum_{i=1}^{n} \mathbf{1}\{ s_i \neq t_i \}
$$

where $\mathbf{1}\{ s_i \neq t_i \}$ is an indicator function that equals 1 if $s_i \neq t_i$ and 0 otherwise.

### Inputs and Outputs

- **Inputs:**  
  - Two equal-length sequences (e.g., binary strings, character arrays)
  
- **Outputs:**  
  - An integer representing the number of differing positions between the two sequences.

## Intended Use

### Domains and Tasks

- **Domain:**  
  - Coding Theory  
  - Information Theory
  
- **Tasks:**  
  - Error Detection  
  - Error Correction  
  - Code Evaluation

### Applicability and Limitations

- **Best Suited For:**  
  - Fixed-length sequences in digital communication systems and error correcting codes.
  
- **Not Recommended For:**  
  - Sequences of differing lengths or applications requiring semantic similarity evaluation (e.g., natural language text similarity).

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Custom implementations in Python (using simple loops or vectorized operations in NumPy)
  - Functions available in libraries such as SciPy.
  - For a conceptual overview, see the [Hamming Distance - Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance).

### Computational Complexity

- **Efficiency:**  
  - Operates in $O(n)$ time, where $n$ is the length of the sequences.
  
- **Scalability:**  
  - Highly efficient for moderate-length sequences; performance scales linearly with sequence length.

## Known Limitations

- **Biases:**  
  - Considers only literal character differences without accounting for semantic or contextual similarities.
  
- **Task Misalignment Risks:**  
  - Not applicable to sequences of unequal length.
  
- **Failure Cases:**  
  - Use with unequal-length inputs will result in errors or undefined behavior.

## Related Metrics

- **Levenshtein Distance:** Measures the minimum edit operations (insertions, deletions, substitutions) required to transform one sequence into another.
- **Damerau-Levenshtein Distance:** Extends Levenshtein Distance by considering transpositions as well.
- **Jaccard Index:** Evaluates similarity based on set overlap, rather than positional differences.

## Further Reading

- **Papers:**  
  - Richard W. Hamming, "Error detecting and error correcting codes", *The Bell System Technical Journal*, 1950. [Available here](https://hdl.handle.net/10945/64206)
  
- **Blogs/Tutorials:**  
  - [Hamming Distance - Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance)

## Citation

```
@article{hamming1950error,
  title={Error detecting and error correcting codes},
  author={Hamming, Richard W},
  journal={The Bell system technical journal},
  volume={29},
  number={2},
  pages={147--160},
  year={1950},
  publisher={Nokia Bell Labs}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu