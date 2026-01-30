---
# Metric Card for BARTScore

BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task. It leverages the pre-trained BART model to compute the conditional likelihood of one text given another, enabling flexible evaluation of different aspects such as informativeness, fluency, factuality, and coherence. BARTScore outperforms existing metrics across multiple tasks and evaluation settings.

## Metric Details

### Metric Description

BARTScore conceptualizes evaluation as a text generation problem, assessing how likely a hypothesis (generated text) is given a reference text, source text, or both. This probability is computed using the log-likelihood of the hypothesis under a pre-trained BART model. Different evaluation perspectives can be achieved by modifying the generation direction:

- **Faithfulness ($s \to h$)**: Measures how well the generated text aligns with the source text.
- **Precision ($r \to h$)**: Evaluates the likelihood of generating the hypothesis given the reference text.
- **Recall ($h \to r$)**: Assesses how easily the reference could be generated from the hypothesis.
- **F-score ($r \leftrightarrow h$)**: Computes an average of Precision and Recall.

Fine-tuning on downstream tasks (e.g., summarization, paraphrasing) and prompt engineering further enhance BARTScore’s adaptability to different domains.

- **Metric Type:** Semantic Similarity  
- **Range:** $(-\infty, 0]$ (log-probabilities, higher is better)  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** Yes  

### Formal Definition

BARTScore is computed as:

$$
BARTScore = \sum _{t=1}^{m} \omega _{t} \log p ( y _{t} \mid y _{\text{<}t}, x, \theta )
$$

where:

- $p(y _{t} \mid y _{<t}, x, \theta)$ is the probability of the $t$-th token in the hypothesis $y$ given the preceding tokens and the source/reference text $x$ under the BART model parameters $\theta$.
- $\omega _{t}$ is an optional weighting factor (default: uniform).

The choice of $x$ and $y$ varies depending on the evaluation perspective (e.g., source-to-hypothesis for faithfulness, reference-to-hypothesis for precision).

### Inputs and Outputs

- **Inputs:**  
  - Source text (optional, for faithfulness evaluation)
  - Generated text (hypothesis)
  - Reference text(s) (for precision, recall, and F-score)

- **Outputs:**  
  - Scalar log-likelihood score (higher indicates better alignment)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:**  
  - Machine Translation  
  - Summarization  
  - Paraphrasing  
  - Data-to-Text Generation  
  - Dialogue Generation  

### Applicability and Limitations

- **Best Suited For:**  
  - Tasks where reference-based evaluation is appropriate (e.g., machine translation, summarization).
  - Evaluating generated text from multiple perspectives (e.g., factuality, coherence, fluency).
  - Cases where fine-tuning and prompt-based customization are beneficial.

- **Not Recommended For:**  
  - Fully reference-free evaluation tasks.
  - Open-ended generation tasks where diversity matters more than similarity to references.
  - Evaluating highly extractive summaries, where performance may degrade.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [BARTScore GitHub Repository](https://github.com/neulab/BARTScore)  
  - Available in Hugging Face `evaluate` module  

### Computational Complexity

- **Efficiency:**  
  - Requires forward passes through BART, making it more computationally expensive than n-gram-based metrics.  
  - Can be optimized using batch processing.  

- **Scalability:**  
  - Suitable for large-scale evaluations but requires GPU acceleration for efficiency.  
  - Performance depends on the pre-trained model size and dataset length.

## Known Limitations

- **Biases:**  
  - BARTScore tends to favor abstractive over extractive summaries.  
  - May be sensitive to the domain of the pre-trained BART model used.  

- **Task Misalignment Risks:**  
  - May not fully capture factual correctness despite faithfulness scoring.  
  - Sensitive to tokenization and domain shift effects.  

- **Failure Cases:**  
  - Performance degrades when evaluating extractive summarization models.  
  - Prompt engineering impacts results significantly, requiring careful selection.  

## Related Metrics

- **ROUGE:** Measures lexical overlap, whereas BARTScore captures semantic similarity.  
- **BERTScore:** Also embeds text using pre-trained models but computes cosine similarity instead of generation probabilities.  
- **BLEU:** Focuses on n-gram precision, lacking semantic alignment capabilities.  

## Further Reading

- **Papers:**  
  - [BARTScore: Evaluating Generated Text as Text Generation (Yuan et al., 2021)](https://arxiv.org/abs/2106.11520)  

- **Blogs/Tutorials:**  
  - [BARTScore GitHub Documentation](https://github.com/neulab/BARTScore)  

## Citation

```
@inproceedings{10.5555/3540261.3542349,
   author = {Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
   title = {BARTSCORE: evaluating generated text as text generation},
   year = {2021},
   isbn = {9781713845393},
   publisher = {Curran Associates Inc.},
   address = {Red Hook, NY, USA},
   booktitle = {Proceedings of the 35th International Conference on Neural Information Processing Systems},
   articleno = {2088},
   numpages = {15},
   series = {NIPS '21}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  