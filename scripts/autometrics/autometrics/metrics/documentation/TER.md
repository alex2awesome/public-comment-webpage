---
# Metric Card for TER (Translation Edit Rate)

TER (Translation Edit Rate) is an automatic evaluation metric designed to measure the effort required to edit a machine translation output to match a reference translation. It computes the minimum number of edits (insertions, deletions, substitutions, and shifts of words or phrases) needed to make a hypothesis identical to a reference. TER provides a straightforward, intuitive measure of translation quality and correlates well with human judgments.

## Metric Details

### Metric Description

TER calculates the minimum number of edits needed to convert a translation hypothesis into one of the provided reference translations. Edits include:
- **Insertions**
- **Deletions**
- **Substitutions**
- **Shifts** (reordering of contiguous sequences of words)

The final score is normalized by the average length of the reference translations. All edits, including shifts, have a uniform cost of 1. 

TER can be computed in two modes:
1. **Untargeted TER (TER):** Compares the hypothesis to predefined reference translations.
2. **Human-targeted TER (HTER):** Compares the hypothesis to a targeted reference created by human annotators to maximize semantic equivalence.

- **Metric Type:** Surface-Level Similarity
- **Range:** $[0, \infty)$
- **Higher is Better?:** No
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

The TER score is calculated as:

$$
\text{TER} = \frac{\text{Number of edits}}{\text{Average reference length}}
$$

Where:
- **Number of edits:** The sum of insertions, deletions, substitutions, and shifts needed to make the hypothesis match the closest reference.
- **Average reference length:** The mean number of words in the reference translations.

### Inputs and Outputs

- **Inputs:**  
  - Hypothesis translation (generated text)
  - One or more reference translations (gold-standard texts)

- **Outputs:**  
  - Scalar TER score (lower values indicate better translations)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Machine Translation

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating structured tasks like machine translation where there is a well-defined correspondence between the hypothesis and reference translations.
  
- **Not Recommended For:**  
  Tasks requiring semantic equivalence or diversity, such as open-ended text generation or dialogue systems.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [SacreBLEU](https://github.com/mjpost/sacrebleu): Includes an implementation of TER.  

### Computational Complexity

- **Efficiency:**  
  The computation of TER involves finding the minimum edit distance, which is optimized using dynamic programming and a greedy beam search for shifts. The algorithm has a complexity of $O(n^2)$ for the edit distance computation and $O(n)$ for the beam search.

- **Scalability:**  
  TER is computationally efficient for sentence-level evaluation but may require optimizations for very large datasets.

## Known Limitations

- **Biases:**  
  - Assigns equal cost to all edits, which may not accurately reflect human perceptions of translation effort.
  - Penalizes valid translations that differ in structure but are semantically equivalent to the reference.
  
- **Task Misalignment Risks:**  
  - Not suited for tasks requiring high semantic understanding or creative language use, as it focuses on surface-level similarity.

- **Failure Cases:**  
  - TER scores may misrepresent translation quality when references are poorly constructed or insufficient in number.

## Related Metrics

- **BLEU:** Focuses on n-gram overlap and does not account for reordering.  
- **METEOR:** Incorporates stemming and synonymy to capture semantic similarity.  
- **HTER:** A variant of TER that uses human-targeted references for improved semantic alignment.

## Further Reading

- **Papers:**  
  - [A Study of Translation Edit Rate with Targeted Human Annotation (Snover et al., 2006)](https://aclanthology.org/2006.amta-papers.25.pdf)

## Citation

```
@inproceedings{snover-etal-2006-study,
    title = "A Study of Translation Edit Rate with Targeted Human Annotation",
    author = "Snover, Matthew  and
      Dorr, Bonnie  and
      Schwartz, Rich  and
      Micciulla, Linnea  and
      Makhoul, John",
    booktitle = "Proceedings of the 7th Conference of the Association for Machine Translation in the Americas: Technical Papers",
    month = aug # " 8-12",
    year = "2006",
    address = "Cambridge, Massachusetts, USA",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2006.amta-papers.25/",
    pages = "223--231"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu