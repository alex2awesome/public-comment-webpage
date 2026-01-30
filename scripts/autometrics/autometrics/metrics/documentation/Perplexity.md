---
# Metric Card for Perplexity

Perplexity (PPL) is a widely used metric for evaluating the fluency of language models. It measures how well a probabilistic model predicts a sequence of tokens, with lower values indicating better predictions. Specifically, it computes the exponentiated average negative log-likelihood of a sequence. Perplexity is only applicable to autoregressive language models (e.g., GPT-2) and **cannot** be used with masked language models like BERT.

## Metric Details

### Metric Description

Perplexity assesses the predictive capability of a language model by computing the exponentiated average negative log-likelihood of a given sequence. It quantifies how uncertain the model is when predicting the next token. A lower perplexity score indicates better model performance, as it suggests the model assigns higher probabilities to the correct tokens.

- **Metric Type:** Fluency
- **Range:** $(1, \infty)$
- **Higher is Better?:** No
- **Reference-Based?:** No
- **Input-Required?:** No (Perplexity can be computed on output tokens alone)

### Formal Definition

Given a sequence of tokens $X = (x_1, x_2, ..., x_T)$, the perplexity of $X$ under a language model with parameters $\theta$ is defined as:

$$
PPL(X) = \exp \left( -\frac{1}{T} \sum_{i=1}^{T} \log p_{\theta}(x_i \mid x_{\text{<}i}) \right)
$$

where:
- $p_{\theta}(x_i \mid x_{\text{<}i})$ is the probability assigned by the model to token $x_i$ given the preceding tokens.
- $T$ is the length of the sequence.

A lower perplexity value indicates that the model assigns higher probabilities to observed sequences, meaning it better predicts the given data.

### Sliding-Window Perplexity

For models with a fixed context size (e.g., GPT-2, LLaMA), perplexity cannot be computed over arbitrarily long sequences directly. Instead, a **sliding-window** approach is used, as described in the [Hugging Face blog on perplexity](https://huggingface.co/docs/transformers/en/perplexity):

- The input sequence is broken into overlapping **windows** of a fixed length.
- Each window is passed through the model, and **only the log-likelihood of the newly introduced tokens** (not the entire window) is used in the perplexity calculation.
- This approach better approximates full-sequence perplexity compared to naïve chunking (which can overestimate perplexity due to loss of context).

Using this method, perplexity is calculated as:

$$
PPL(X) = \exp \left( -\frac{1}{T} \sum_{i=1}^{T} \log p_{\theta}(x_i \mid x_{\max(1, i-k):i-1}) \right)
$$

where:
- $k$ is the model’s fixed context size,
- The probability of each token $x_i$ is conditioned on a **sliding context of at most $k$ tokens**.

This method provides a **more realistic** evaluation of model fluency while efficiently handling long sequences.

### Inputs and Outputs

- **Inputs:**  
  - A sequence of text tokens (typically output from a model)
  - A trained language model (e.g., GPT-2)
  - Tokenizer for processing input text

- **Outputs:**  
  - A scalar value representing the perplexity score of the input text

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Language Modeling, Dialogue Generation, Storytelling, Code Completion

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating the fluency of language models, especially autoregressive models
  - Comparing the relative performance of different language models on the same dataset
  
- **Not Recommended For:**  
  - Evaluating masked language models (e.g., BERT) since perplexity is undefined for non-autoregressive architectures
  - Assessing high-level semantic coherence, factual consistency, or diversity in generated text

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/perplexity)

### Computational Complexity

- **Efficiency:**  
  - Perplexity calculation involves computing log-likelihoods for each token, making it computationally intensive for large datasets.
  
- **Scalability:**  
  - Efficient when used with GPU acceleration but may become expensive for long sequences due to the need for multiple forward passes.

## Known Limitations

- **Biases:**  
  - Sensitive to tokenization choices; different tokenization schemes can yield different perplexity values.
  - Models trained on specific domains may yield artificially low perplexity scores on similar datasets while failing on out-of-domain data.
  
- **Task Misalignment Risks:**  
  - Perplexity measures token-level fluency but does not assess semantic correctness or factuality.
  
- **Failure Cases:**  
  - Does not distinguish between grammatically correct but nonsensical text and genuinely coherent text.
  - Perplexity values are not always comparable across different models due to differences in vocabulary and tokenization.

## Related Metrics

- **Cross-Entropy Loss:** Closely related to perplexity, as perplexity is the exponentiated cross-entropy loss.
- **BERTScore:** Evaluates semantic similarity rather than fluency.
- **ROUGE/BLEU:** Measure lexical overlap rather than model uncertainty.

## Further Reading

- **Papers:**  
  - Jelinek et al. (1977) - [Perplexity: A Measure of the Difficulty of Speech Recognition Tasks](https://doi.org/10.1121/1.2016299)
  - Hugging Face Documentation - [Perplexity of Fixed-Length Models](https://huggingface.co/docs/transformers/en/perplexity)

- **Blogs/Tutorials:**  
  - [Understanding Evaluation Metrics for Language Models](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)
  - [Hugging Face's Guide to Perplexity](https://huggingface.co/docs/transformers/en/perplexity)

## Citation

```
@article{10.1121/1.2016299,
    author = {Jelinek, F. and Mercer, R. L. and Bahl, L. R. and Baker, J. K.},
    title = {Perplexity—a measure of the difficulty of speech recognition tasks},
    journal = {The Journal of the Acoustical Society of America},
    volume = {62},
    number = {S1},
    pages = {S63-S63},
    year = {2005},
    month = {08},
    issn = {0001-4966},
    doi = {10.1121/1.2016299},
    url = {https://doi.org/10.1121/1.2016299},
    eprint = {https://pubs.aip.org/asa/jasa/article-pdf/62/S1/S63/11558910/s63\_5\_online.pdf},
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu