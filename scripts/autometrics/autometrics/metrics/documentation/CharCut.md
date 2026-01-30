---
# Metric Card for CharCut

CharCut is a character-based evaluation metric for machine translation that measures the similarity between candidate and reference translations using a human-targeted string difference algorithm. It identifies and scores "loose differences" through an iterative process of extracting long common substrings (LCSubstrs), designed to reduce noise and better align with human perception of meaningful edits. Unlike traditional edit distances, CharCut emphasizes user-aligned visualization and scoring of string differences, offering both evaluation and diagnostic utility.

## Metric Details

### Metric Description

CharCut compares generated text to a reference by iteratively identifying the longest common substrings (LCSubstrs) between the candidate and reference segments, under a threshold to avoid short, noisy matches. After extracting all LCSubstrs that meet a length-based threshold (typically ≥3 characters), the remaining non-matching substrings are categorized as "loose differences." CharCut also detects and handles shifts (reordered substrings) and assigns penalties for insertions, deletions, and shifts, yielding a normalized similarity score.

This metric is both interpretable and efficient, with results shown to correlate strongly with human judgments in WMT16 system- and segment-level evaluations. CharCut is especially useful when both automated scoring and human-readable highlighting of differences are needed.

- **Metric Type:** Surface-Level Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** No (lower scores indicate higher similarity)  
- **Reference-Based?:** Yes  
- **Input-Required?:** No

### Formal Definition

Let $C_0$ and $R_0$ be the candidate and reference segments, respectively. CharCut proceeds as follows:

1. **Segmentation Phase:**
   - Iteratively extract the longest common substrings (LCSubstrs) between $C_n$ and $R_n$, cutting them from both strings until no LCSubstr exceeds a predefined length threshold (typically 3 characters).
   - Add longest common prefix and suffix (if applicable) to the LCSubstr set.

2. **Shift Detection:**
   - Identify re-ordered substrings (shifts) among LCSubstrs not in the longest common subsequence (LCS) of the ordered match sets.

3. **Scoring Phase:**
   - Assign cost 1 to each inserted, deleted, or shifted character.
   - Compute normalized score using the following formula:

$$
\text{CharCut}(C_0, R_0) = \min \left(1, \frac{\#\text{deletions} + \#\text{insertions} + \#\text{shifts}}{2 \cdot |C_0|} \right)
$$

A score of 0 indicates perfect match; 1 indicates maximal divergence.

### Inputs and Outputs

- **Inputs:**  
  - Predictions (candidate translations): list of strings  
  - References (gold-standard translations): list of strings  

- **Outputs:**  
  - `charcut_mt`: a float score between 0 and 1 (lower is better)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation

### Applicability and Limitations

- **Best Suited For:**  
  - Machine translation evaluation, especially when visual inspection of differences is also required.  
  - Languages with alphabetic or subword representations (e.g., Byte Pair Encoding).  

- **Not Recommended For:**  
  - Evaluation of highly diverse or creative tasks (e.g., storytelling, open-ended generation) where character-level overlap is not informative.  
  - Scenarios requiring semantic similarity or meaning preservation beyond surface-level matches.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Evaluate: `charcut`](https://huggingface.co/spaces/evaluate-metric/CharCut)
  - [Original GitHub Repository (alardill/CharCut)](https://github.com/alardill/CharCut)
  - [Repackaged GitHub (BramVanroy/CharCut)](https://github.com/BramVanroy/CharCut)

### Computational Complexity

- **Efficiency:**  
  - Processes ~260 segment pairs/second on a 2.8 GHz processor using a Python implementation.  
  - More efficient than CharacTER and comparable in speed to chrF.

- **Scalability:**  
  - Suitable for medium to large-scale evaluations. May be slower than embedding-based methods for very large datasets.

## Known Limitations

- **Biases:**  
  - Sensitive to character-level surface variations (e.g., spelling differences, inflections), which may not reflect true semantic error.  

- **Task Misalignment Risks:**  
  - Inappropriate for evaluating tasks requiring semantic alignment or abstraction.  

- **Failure Cases:**  
  - May fail to identify semantic equivalence in paraphrased or reordered content when surface form diverges significantly.  
  - Shifts are detected without considering shift *distance*, so distant reordering incurs the same cost as nearby shifts.

## Related Metrics

- **chrF:** Another character-level metric, but based on precision and recall of character n-grams.  
- **CharacTER:** Related metric that penalizes edit distance on characters with additional weighting for shifts.  
- **TER:** Word-based edit distance metric.  
- **BLEU:** n-gram overlap metric with brevity penalty (much less sensitive to character-level variation).

## Further Reading

- **Papers:**  
  - [CharCut: Human-Targeted Character-Based MT Evaluation with Loose Differences (Lardilleux & Lepage, 2017)](https://aclanthology.org/2017.iwslt-1.20)  

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{lardilleux-lepage-2017-charcut,  
  title = "{CHARCUT}: Human-Targeted Character-Based {MT} Evaluation with Loose Differences",  
  author = "Lardilleux, Adrien  and  
    Lepage, Yves",  
  booktitle = "Proceedings of the 14th International Conference on Spoken Language Translation",  
  month = dec # " 14-15",  
  year = "2017",  
  address = "Tokyo, Japan",  
  publisher = "International Workshop on Spoken Language Translation",  
  url = "https://aclanthology.org/2017.iwslt-1.20",  
  pages = "146--153",  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu