---
# Metric Card for Self-BLEU

Self-BLEU is a reference-free diversity metric used in text generation tasks. It quantifies the similarity of each generated sentence to the rest of the generated outputs using the BLEU score, treating each sentence in turn as the hypothesis and the others as the reference. Lower Self-BLEU scores indicate higher diversity, making it useful for detecting mode collapse in generative models.

## Metric Details

### Metric Description

Self-BLEU evaluates the diversity of a set of generated sentences by measuring the average BLEU score of each sentence against the rest. BLEU is typically used to measure similarity between a generated sentence and reference(s), but in Self-BLEU, the generated samples themselves act as both hypotheses and references. If generated texts are overly similar (as in mode collapse), Self-BLEU will be high; if texts are diverse, Self-BLEU will be low.

- **Metric Type:** Diversity  
- **Range:** 0 to 1  
- **Higher is Better?:** No  
- **Reference-Based?:** No  
- **Input-Required?:** No  

### Formal Definition

Let $X = \{x_1, x_2, \dots, x_n\}$ be the set of generated sentences. For each $x_i \in X$, compute BLEU with $x_i$ as the hypothesis and $X \setminus \{x_i\}$ as references:

$$
\text{Self-BLEU}(X) = \frac{1}{n} \sum_{i=1}^{n} \text{BLEU}(x_i, X \setminus \{x_i\})
$$

Each BLEU computation typically uses uniform $n$-gram weights (e.g., unigram to 5-gram), along with a smoothing function such as `method1` from NLTK’s `SmoothingFunction`.

### Inputs and Outputs

- **Inputs:**  
  - A set of generated sentences (no external references required)  
  - $n$-gram order (commonly 2 to 5)  
  - Tokenization (e.g., NLTK word tokenizer)  

- **Outputs:**  
  - A scalar Self-BLEU score between 0 and 1

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Dialogue Generation, Storytelling, Paraphrasing, Creative Writing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating diversity in unconditional generation or detecting mode collapse in GANs or autoregressive LMs.

- **Not Recommended For:**  
  Tasks where high inter-output similarity is desirable, or settings where quality needs to be measured jointly with diversity.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Texygen](https://github.com/geek-ai/Texygen): Official implementation includes multiprocessing support and sampling optimizations.  
  - Python NLTK (`nltk.translate.bleu_score`) is used internally to compute BLEU for each hypothesis-reference pair, but does **not** support Self-BLEU directly.

### Computational Complexity

- **Efficiency:**  
  $O(n^2 \cdot m)$ where $n$ is the number of generated samples and $m$ is the average sentence length. Optimizations like subsampling and multiprocessing are commonly applied in practice.

- **Scalability:**  
  Scales poorly with large output sets due to pairwise computation; practical use often limits evaluation to 500–1,000 samples.

## Known Limitations

- **Biases:**  
  Sensitive to tokenization and $n$-gram overlap. Penalizes semantically diverse but lexically similar outputs.

- **Task Misalignment Risks:**  
  Ignores output quality entirely. A diverse set of incoherent outputs may score well.

- **Failure Cases:**  
  - Semantically equivalent but lexically different outputs (e.g., paraphrases) may appear overly diverse.  
  - Cannot distinguish diversity from incoherence.

## Related Metrics

- **BLEU:** Used internally within Self-BLEU to compute similarity between sentences.  
- **MS-Jaccard:** Joint quality-diversity metric that measures overlap with real data.  
- **Frechet BERT Distance (FBD):** Embedding-based diversity-quality measure.  
- **Entropy:** Reference-free diversity measure.

## Further Reading

- **Papers:**  
  - Zhu et al. (2018): *Texygen: A Benchmarking Platform for Text Generation Models*  
  - Montahaei et al. (2019): *Jointly Measuring Diversity and Quality in Text Generation Models*  

- **Blogs/Tutorials:**  
  - Needs more information

## Citation

```
@article{zhu2018texygen,  
  title={Texygen: A Benchmarking Platform for Text Generation Models},  
  author={Zhu, Yaoming and Lu, Sidi and Zheng, Lei and Guo, Jiaxian and Zhang, Weinan and Wang, Jun and Yu, Yong},  
  journal={SIGIR},  
  year={2018}  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu