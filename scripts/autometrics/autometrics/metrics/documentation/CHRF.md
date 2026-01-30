---
# Metric Card for CHRF++

CHRF++ (Character n-gram F-score with word n-grams) is a metric for evaluating the quality of machine translation or other text generation tasks. It calculates the precision and recall of character-level n-grams and augments this by incorporating word n-grams, making it a versatile metric that balances lexical similarity and fluency. CHRF++ can handle languages with diverse morphological structures and is tokenization-independent.

## Metric Details

### Metric Description

CHRF++ evaluates text quality by calculating F-scores over character n-grams while also incorporating word-level n-grams. The metric averages precision and recall of n-grams to compute an overall F-score, parameterized by $\beta$ to adjust the importance of recall relative to precision. By capturing character-level n-gram overlaps, CHRF++ is well-suited for languages with rich morphology or those that lack clear tokenization conventions.

- **Metric Type:** Surface-Level Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** No  

### Formal Definition

The CHRF++ score is calculated as:

$$
CHRF_{\beta} = (1 + \beta^2) \cdot \frac{CHRP \cdot CHRR}{\beta^2 \cdot CHRP + CHRR}
$$

Where:

- $CHRP$ is the average precision of character and word n-grams:

$$
CHRP = \frac{1}{N} \sum_{n=1}^N \frac{\text{n-grams in hypothesis and reference}}{\text{total n-grams in hypothesis}}
$$

- $CHRR$ is the average recall of character and word n-grams:

$$
CHRR = \frac{1}{N} \sum_{n=1}^N \frac{\text{n-grams in hypothesis and reference}}{\text{total n-grams in reference}}
$$

- $N$ is the maximum n-gram length (typically $N=6$ for characters and $N=2$ for words).
- $\beta$ adjusts the weight of recall relative to precision (e.g., $\beta=3$ for CHRF3).

### Inputs and Outputs

- **Inputs:**  
  - Generated text (hypothesis)  
  - Reference text(s)  

- **Outputs:**  
  - Scalar CHRF++ score (range: 0 to 1)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation, Summarization, Paraphrasing, Data-to-Text Generation  

### Applicability and Limitations

- **Best Suited For:**  
  - Tasks where tokenization independence and morphological sensitivity are crucial, such as low-resource or morphologically rich languages.  

- **Not Recommended For:**  
  - Open-ended or creative generation tasks where diversity or higher-level semantics outweigh character or word n-gram similarity.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [SacreBLEU](https://github.com/mjpost/sacrebleu)

### Computational Complexity

- **Efficiency:**  
  CHRF++ is computationally efficient, with complexity proportional to the number of n-grams in the input texts.  

- **Scalability:**  
  Scales well for datasets of varying sizes due to its simple character and word n-gram matching.

## Known Limitations

- **Biases:**  
  - Can overemphasize lexical similarity, penalizing valid outputs with synonymous expressions or structural variations.
  
- **Task Misalignment Risks:**  
  - May not adequately capture semantic adequacy or fluency in tasks requiring understanding beyond n-gram overlaps.  

- **Failure Cases:**  
  - Struggles with evaluating translations into languages where character n-grams lack correspondence with meaningful units (e.g., ideographic languages without preprocessing).

## Related Metrics

- **BLEU:** A word-based n-gram precision metric, widely used but less sensitive to morphological differences.  
- **METEOR:** Incorporates synonym matching for better semantic similarity.  
- **ROUGE:** Focuses on recall, commonly used for summarization tasks.  

## Further Reading

- **Papers:**  
  - [Original CHRF Paper (Popović, 2015)](https://aclanthology.org/W15-3049)  

- **Blogs/Tutorials:**  
  - SacreBLEU Documentation: [CHRF++ details](https://github.com/mjpost/sacrebleu)

## Citation

```
@inproceedings{popovic-2015-chrf,
    title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",
    author = "Popovi{\'c}, Maja",
    editor = "Bojar, Ond{\v{r}}ej  and
      Chatterjee, Rajan  and
      Federmann, Christian  and
      Haddow, Barry  and
      Hokamp, Chris  and
      Huck, Matthias  and
      Logacheva, Varvara  and
      Pecina, Pavel",
    booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W15-3049/",
    doi = "10.18653/v1/W15-3049",
    pages = "392--395"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu