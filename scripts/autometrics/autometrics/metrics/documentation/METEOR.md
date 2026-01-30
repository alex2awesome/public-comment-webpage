---
# Metric Card for METEOR

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a reference-based metric for evaluating machine translation and related tasks. It calculates a weighted harmonic mean of unigram precision and recall, with higher weight on recall, and includes a fragmentation penalty to assess fluency. METEOR supports exact, stemmed, and synonym-based word matching to improve correlation with human judgment compared to earlier metrics like BLEU.

## Metric Details

### Metric Description

METEOR evaluates a candidate sentence by aligning its unigrams with one or more reference sentences using a sequence of matching strategies: exact matches, stemmed matches (e.g., via the Porter Stemmer), and WordNet synonymy. It constructs the alignment to minimize crossing links and maximize matching chunks, thereby rewarding both content similarity and word order.

The score is computed as follows:
- Compute unigram precision and recall from matched unigrams.
- Calculate a harmonic mean ($F_{\text{mean}}$) that gives more weight to recall.
- Penalize fragmented matches using a penalty based on the number of chunks.
- Final score is $(1 - \text{Penalty}) \cdot F_{\text{mean}}$.

- **Metric Type:** Semantic Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** No

### Formal Definition

Given a hypothesis $h$ and a reference $r$, define:

- $P = \frac{\text{\# matched unigrams}}{\text{\# unigrams in } h}$  
- $R = \frac{\text{\# matched unigrams}}{\text{\# unigrams in } r}$

The $F_{\text{mean}}$ is calculated with recall weighted 9x over precision:

$$
F_{\text{mean}} = \frac{10 \cdot P \cdot R}{R + 9 \cdot P}
$$

Let $ch$ be the number of contiguous matching chunks, and $m$ the number of matched unigrams. The penalty is:

$$
\text{Penalty} = 0.5 \cdot \left(\frac{ch}{m}\right)^3
$$

The final METEOR score is:

$$
\text{Score} = (1 - \text{Penalty}) \cdot F_{\text{mean}}
$$

### Inputs and Outputs

- **Inputs:**  
  - Hypothesis sentence (tokenized)  
  - One or more reference sentences (tokenized)  

- **Outputs:**  
  - A scalar METEOR score between 0 and 1

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation, Summarization, Paraphrasing, Dialogue Generation, Style Transfer

### Applicability and Limitations

- **Best Suited For:**  
  Evaluation of system outputs in structured generation tasks where lexical overlap and word order are important. Especially suited for sentence-level evaluation where semantic equivalence is essential but not strictly literal.

- **Not Recommended For:**  
  Creative generation tasks with high lexical variance (e.g., storytelling, poetry), or domains where syntactic or multi-word phrase matching is more important than unigram similarity.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [NLTK](https://www.nltk.org/_modules/nltk/translate/meteor_score.html)
  - [Hugging Face Evaluate](https://huggingface.co/spaces/evaluate-metric/METEOR)

### Computational Complexity

- **Efficiency:**  
  Efficient at sentence-level but includes iterative alignment and synonym lookup, making it moderately heavier than BLEU. Includes configurable parameters for alignment strategies and penalties.

- **Scalability:**  
  Scales adequately for batch processing but may slow on large corpora due to synonym lookups (WordNet) and alignment crossing minimization.

## Known Limitations

- **Biases:**  
  May reflect biases present in WordNet or stemming algorithms.  
  Can favor surface-form similarity and penalize valid lexical variation not captured by synonyms.

- **Task Misalignment Risks:**  
  Underperforms in tasks where surface-level matching is not meaningful (e.g., open-domain dialogue).

- **Failure Cases:**  
  Produces low scores for valid paraphrases without lexical overlap or synonymy.  
  Cannot handle multi-word synonymy or deeper semantic equivalence.

## Related Metrics

- **BLEU:** N-gram precision-only metric with brevity penalty, less recall-sensitive.  
- **ROUGE:** Recall-oriented metric commonly used in summarization.  
- **BERTScore:** Embedding-based metric capturing contextual semantic similarity.  

## Further Reading

- **Papers:**  
  - [Banerjee & Lavie (2005)](https://aclanthology.org/W05-0909)  
  - [Lavie & Agarwal (2007)](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf)

- **Blogs/Tutorials:**  
  - [NLTK METEOR Documentation](https://www.nltk.org/_modules/nltk/translate/meteor_score.html)  
  - [Hugging Face Evaluate Metric Card](https://huggingface.co/spaces/evaluate-metric/METEOR)

## Citation

```
@inproceedings{banerjee-lavie-2005-meteor,
    title = "{METEOR}: An Automatic Metric for {MT} Evaluation with Improved Correlation with Human Judgments",
    author = "Banerjee, Satanjeev  and
      Lavie, Alon",
    editor = "Goldstein, Jade  and
      Lavie, Alon  and
      Lin, Chin-Yew  and
      Voss, Clare",
    booktitle = "Proceedings of the {ACL} Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization",
    month = jun,
    year = "2005",
    address = "Ann Arbor, Michigan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W05-0909/",
    pages = "65--72"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu