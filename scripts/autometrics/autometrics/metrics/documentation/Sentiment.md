---
# Metric Card for Sentiment

The Sentiment metric quantifies the sentiment polarity of generated text using a pretrained model fine-tuned on Twitter data: `cardiffnlp/twitter-roberta-base-sentiment-latest`. It converts the model’s categorical output (negative, neutral, positive) into a continuous regression score by computing the normalized difference between positive and negative class probabilities. This allows sentiment to be used as a reference-free, scalar evaluation metric for generation tasks.

## Metric Details

### Metric Description

This metric uses the Cardiff NLP Twitter RoBERTa-based model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to classify input text into three categories: negative, neutral, and positive. Instead of returning a discrete label, it converts the model’s output probabilities into a scalar sentiment score by computing:

$$
\text{Score} = \frac{p_{\text{positive}} - p_{\text{negative}}}{2}
$$

This results in a range of $[-0.5, 0.5]$, where higher values indicate more positive sentiment and lower values indicate more negative sentiment.

- **Metric Type:** Reference-Free  
- **Range:** $[-0.5, 0.5]$  
- **Higher is Better?:** Yes (when positive sentiment is desirable)  
- **Reference-Based?:** No  
- **Input-Required?:** No

### Formal Definition

Given a generated output string $\hat{y}$, the model predicts class probabilities:
- $p_{\text{neg}}$: probability of negative sentiment
- $p_{\text{neu}}$: probability of neutral sentiment
- $p_{\text{pos}}$: probability of positive sentiment

The sentiment score is computed as:

$$
\text{Sentiment}(\hat{y}) = \frac{p_{\text{pos}} - p_{\text{neg}}}{2}
$$

### Inputs and Outputs

- **Inputs:**  
  - Generated text (output string)

- **Outputs:**  
  - Scalar sentiment score in $[-0.5, 0.5]$

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems  
- **Tasks:** Dialogue Generation, Response Generation, Creative Writing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating sentiment polarity of model-generated outputs, particularly in tasks where emotional tone or user alignment matters (e.g., empathetic chatbots, personalized generation).

- **Not Recommended For:**  
  Tasks where neutrality or factuality is the goal (e.g., summarization, translation). The metric does not account for semantic correctness or fluency.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`)

### Computational Complexity

- **Efficiency:**  
  The model performs a forward pass through a transformer for each input, with complexity roughly $O(n)$ in sequence length. Inference is efficient but not suitable for extremely high-throughput settings without batching.

- **Scalability:**  
  Supports batched inference for moderate-scale evaluation (e.g., thousands of generations).

## Known Limitations

- **Biases:**  
  The model is trained on Twitter data and may reflect social media-specific biases, such as overemphasis on sarcasm, slang, or cultural expressions common to Twitter users.

- **Task Misalignment Risks:**  
  Applying this metric to domains outside Twitter or to tasks where sentiment is irrelevant can result in misleading evaluations.

- **Failure Cases:**  
  - Neutral content may still yield non-zero scores.  
  - The model can misinterpret ambiguous or sarcastic statements.  
  - Sentiment extremes may be underrepresented or flattened due to the regression mapping.

## Related Metrics

- **BERTScore:** Measures semantic similarity, not sentiment.  
- **Toxicity (e.g., FastTextToxicity):** Related in spirit but designed to identify harmful rather than emotionally valenced content.  
- **PPLM Sentiment Control Metrics:** Used in controlled generation evaluation but not standardized.

## Further Reading

- **Papers:**  
  - [TweetNLP: Cutting-Edge Natural Language Processing for Social Media](https://aclanthology.org/2022.emnlp-demos.5)  
  - [TimeLMs: Diachronic Language Models from Twitter](https://aclanthology.org/2022.acl-demo.25)

- **Blogs/Tutorials:**  
  - [CardiffNLP Model Hub (Hugging Face)](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

## Citation

```
@inproceedings{camacho-collados-etal-2022-tweetnlp,  
    title = "{T}weet{NLP}: Cutting-Edge Natural Language Processing for Social Media",  
    author = "Camacho-collados, Jose  and  
      Rezaee, Kiamehr  and  
      Riahi, Talayeh  and  
      Ushio, Asahi  and  
      Loureiro, Daniel  and  
      Antypas, Dimosthenis  and  
      Boisson, Joanne  and  
      Espinosa Anke, Luis  and  
      Liu, Fangyu  and  
      Mart{\'\i}nez C{\'a}mara, Eugenio",  
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",  
    month = dec,  
    year = "2022",  
    address = "Abu Dhabi, UAE",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2022.emnlp-demos.5",  
    pages = "38--49"  
}
```

```
@inproceedings{loureiro-etal-2022-timelms,  
    title = "{T}ime{LM}s: Diachronic Language Models from {T}witter",  
    author = "Loureiro, Daniel  and  
      Barbieri, Francesco  and  
      Neves, Leonardo  and  
      Espinosa Anke, Luis  and  
      Camacho-collados, Jose",  
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",  
    month = may,  
    year = "2022",  
    address = "Dublin, Ireland",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2022.acl-demo.25",  
    doi = "10.18653/v1/2022.acl-demo.25",  
    pages = "251--260"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu