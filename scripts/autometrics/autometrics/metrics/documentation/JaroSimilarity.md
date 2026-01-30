---
# Metric Card for Jaro Similarity

Jaro Similarity is a string metric used to measure the similarity between two strings based on the number of matching characters and the number of transpositions. It produces a score between 0 and 1, where 1 indicates an exact match and 0 indicates no similarity.

## Metric Details

### Metric Description

Jaro Similarity computes a score by comparing two strings as follows:
- **Matching Characters:** Two characters from the two strings are considered matching if they are the same and not farther than $\lfloor\max(|s_1|, |s_2|)/2\rfloor - 1$ positions apart.
- **Transpositions:** Transpositions are counted as half the number of matching characters that appear in a different order.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let $s_1$ and $s_2$ be two strings with lengths $|s_1|$ and $|s_2|$, respectively. Let $m$ denote the number of matching characters, and $t$ denote half the number of transpositions. The Jaro Similarity $J$ is defined as:

$$
J = \frac{1}{3} \left( \frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m - t}{m} \right)
$$

### Inputs and Outputs

- **Inputs:**  
  - Two strings (or sequences) to compare.
  
- **Outputs:**  
  - A float representing the similarity score in the range [0, 1].

## Intended Use

### Domains and Tasks

- **Domain:**  
  - Record Linkage
  
- **Tasks:**  
  - Record Linkage, Data Deduplication, Approximate String Matching

### Applicability and Limitations

- **Best Suited For:**  
  - Comparing strings in applications such as record linkage or data deduplication where a normalized similarity score is useful.
  
- **Not Recommended For:**  
  - Applications requiring semantic or context-aware similarity, as this metric considers only character-level differences.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Implementations are available in multiple programming languages (Python, C++, Java, etc.). For example, a detailed explanation and implementation can be found on [GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)

### Computational Complexity

- **Efficiency:**  
  - The algorithm operates in $O(|s_1| \times |s_2|)$ time.
  
- **Scalability:**  
  - Suitable for comparing moderate-length strings; performance may degrade with very long strings.

## Known Limitations

- **Biases:**  
  - The metric is sensitive to the chosen matching window, which may affect the similarity score for strings of varying lengths.
  
- **Task Misalignment Risks:**  
  - May not perform well in scenarios where higher-level semantic similarity is required.
  
- **Failure Cases:**  
  - Less effective when strings contain numerous transpositions or when small differences are critical.

## Related Metrics

- **Jaro-Winkler Similarity:** A variant that incorporates a prefix scale to give extra weight to common prefixes.
- **Levenshtein Distance:** Measures the number of edit operations (insertions, deletions, substitutions) needed to transform one string into another.
- **Damerau-Levenshtein Distance:** Extends Levenshtein by also considering transpositions as a valid edit operation.

## Further Reading

- **Papers:**  
  - Matthew A. Jaro, "Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida", *Journal of the American Statistical Association*, 1989. [Available here](https://www.jstor.org/stable/2289924) Jaro-AdvancesRecordLinkageMethodology-1989.pdf](https://www.jstor.org/stable/2289924)
  
- **Blogs/Tutorials:**  
  - [Jaro and Jaro-Winkler Similarity - GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)

## Citation

```
@article{3ebd50ca-85b6-3914-bf38-759fcad3ed72,
 ISSN = {01621459, 1537274X},
 URL = {http://www.jstor.org/stable/2289924},
 author = {Matthew A. Jaro},
 journal = {Journal of the American Statistical Association},
 number = {406},
 pages = {414--420},
 publisher = {[American Statistical Association, Taylor & Francis, Ltd.]},
 title = {Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida},
 urldate = {2025-05-07},
 volume = {84},
 year = {1989}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu