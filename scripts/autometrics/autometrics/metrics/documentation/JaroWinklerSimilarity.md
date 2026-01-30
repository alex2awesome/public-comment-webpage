---
# Metric Card for Jaro-Winkler Similarity

Jaro-Winkler Similarity is a string metric that builds upon the Jaro Similarity by incorporating a prefix scale to give extra weight to common prefixes. It is widely used in record linkage, data deduplication, and other applications where matching similar strings (such as names) is critical.

## Metric Details

### Metric Description

Jaro-Winkler Similarity adjusts the base Jaro Similarity score by factoring in the length of the common prefix (up to 4 characters) between two strings. This enhancement increases the similarity score for strings that match from the beginning. The similarity score is normalized between 0 and 1, where 1 indicates an exact match.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let $J$ be the Jaro Similarity between two strings $s_1$ and $s_2$, defined as:

$$
J = \frac{1}{3} \left( \frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m - t}{m} \right)
$$

where:
- $m$ is the number of matching characters,
- $t$ is half the number of transpositions.

The Jaro-Winkler Similarity, $JW$, is then given by:

$$
JW = J + l \cdot p \cdot (1 - J)
$$

where:
- $l$ is the length of the common prefix (maximum 4),
- $p$ is a constant scaling factor (typically 0.1).

### Inputs and Outputs

- **Inputs:**  
  - Two strings to compare.
  - Optional parameters:
    - `prefix_weight` ($p$), default is 0.1.
    - Maximum prefix length, default is 4.
  
- **Outputs:**  
  - A float representing the similarity score in the range [0, 1].

## Intended Use

### Domains and Tasks

- **Domain:**  
  - Record Linkage  
  - Data Deduplication  
  - Information Retrieval
  
- **Tasks:**  
  - Matching names and addresses in databases.
  - Detecting duplicate records.
  - Evaluating string similarity in record linkage.

### Applicability and Limitations

- **Best Suited For:**  
  - Applications where the initial part of the string is particularly significant (e.g., surnames in record linkage).
  - Scenarios where slight typographical errors need to be tolerated.
  
- **Not Recommended For:**  
  - Cases where the common prefix is not indicative of overall similarity.
  - Applications that require a metric adhering to the triangle inequality (Jaro-Winkler does not satisfy this).

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Implementations are available in multiple programming languages (Python, C++, Java, etc.).
  - Detailed implementations and explanations can be found on [GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)
  - The [Levenshtein module](https://rapidfuzz.github.io/Levenshtein/levenshtein.html#jaro-winkler) documentation provides context on related edit-distance metrics.
  - The original paper by Winkler (1990), "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage", outlines the theoretical foundation. [here](https:files.eric.ed.gov:fulltext:ED325505.pdf)

### Computational Complexity

- **Efficiency:**  
  - The algorithm typically operates in $O(|s_1| \times |s_2|)$ time.
  
- **Scalability:**  
  - Suitable for moderate-length strings, with performance diminishing for very long strings.

## Known Limitations

- **Biases:**  
  - The metric heavily weights the common prefix, which might not be appropriate for all types of data.
  
- **Task Misalignment Risks:**  
  - It may overestimate similarity for strings with similar beginnings but divergent endings.
  
- **Failure Cases:**  
  - Less effective when the prefix does not carry significant meaning or when errors occur predominantly beyond the prefix.

## Related Metrics

- **Jaro Similarity:** The base metric without the prefix adjustment.
- **Levenshtein Distance:** Measures the number of edit operations required to transform one string into another.
- **Damerau-Levenshtein Distance:** Extends Levenshtein by also considering transpositions.
- **Hamming Distance:** Counts differences in fixed-length strings.

## Further Reading

- **Papers:**  
  - Winkler, W. E. (1990). "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage." [Available here](https://files.eric.ed.gov/fulltext/ED325505.pdf)
  - Jaro, M. A. (1989). "Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida." [Available here](https://www.jstor.org/stable/2289924)
  
- **Blogs/Tutorials:**  
  - [Jaro and Jaro-Winkler Similarity - GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)

## Citation

```
@article{winkler1990string,
  title={String comparator metrics and enhanced decision rules in the fellegi-sunter model of record linkage.},
  author={Winkler, William E},
  year={1990},
  publisher={ERIC}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu