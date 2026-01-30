---
# Metric Card for InfoLM

InfoLM is a reference-based metric for evaluating the similarity between a generated and a reference text by computing a divergence between their distributions in embedding space, as derived from a pre-trained masked language model (MLM). Unlike surface-level metrics, InfoLM supports a family of information-theoretic distances (e.g., KL divergence, Rényi divergence, L1/L2 distances) and aims to better capture semantic equivalence while also offering configurability to adjust sensitivity and robustness. It was introduced to address limitations of standard string-based metrics, particularly in summarization and data-to-text generation tasks.

## Metric Details

### Metric Description

InfoLM computes a divergence between the distributions of the hypothesis and reference texts using the output of a pre-trained masked language model (e.g., BERT). It operates in embedding space, evaluating how the probability distributions over tokens (conditioned by context) differ between the generated and reference sequences. The metric can be instantiated with different divergence measures such as KL divergence, Rényi divergence, Alpha/Beta/AB divergences, and common distance functions (L1, L2, L∞, Fisher-Rao).

Each divergence formulation captures different sensitivity profiles; for instance, KL divergence is asymmetric and more sensitive to underestimation of support, while Rényi divergence can interpolate between various behaviors depending on the alpha parameter. This configurability makes InfoLM adaptable to different evaluation priorities (e.g., penalizing hallucinations, encouraging coverage).

- **Metric Type:** Semantic Similarity
- **Range:** Depends on selected divergence (e.g., [0, ∞) for KL; [0, 2] for L1)
- **Higher is Better?:** Depends on selected divergence (e.g., lower is better for L1, L2, KL)
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

Let $P$ and $Q$ be the distributions over tokens derived from the masked language model for the reference and hypothesis texts, respectively. InfoLM measures the divergence $D(P \| Q)$ using a chosen information-theoretic formulation, such as:

- KL Divergence:
$$
D _{KL}(P \| Q) = \sum _{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
$$

- $\alpha$-divergence:
$$
D _{\alpha}(P \| Q) = \frac{1}{\alpha(\alpha - 1)} \left( \sum _{x \in \mathcal{X}} P(x)^{\alpha} Q(x)^{1 - \alpha} - 1 \right)
$$

- Rényi Divergence:
$$
D _{\alpha}^{\text{Rényi}}(P \| Q) = \frac{1}{\alpha - 1} \log \left( \sum _{x \in \mathcal{X}} P(x)^{\alpha} Q(x)^{1 - \alpha} \right)
$$

Where $\alpha$ is a hyperparameter, and $\mathcal{X}$ is the vocabulary space.

These divergences are computed over distributions derived from MLM logits, optionally scaled by temperature and normalized via IDF weighting.

### Inputs and Outputs

- **Inputs:**  
  - `preds`: A list of generated/hypothesis texts  
  - `target`: A list of reference texts  
  - Optional configuration:
    - `model_name_or_path`: Pretrained masked language model (e.g., `"bert-base-uncased"`)
    - `information_measure`: Divergence type (e.g., `'kl_divergence'`, `'renyi_divergence'`)
    - `temperature`: Calibration factor for smoothing
    - `idf`: Whether to apply IDF weighting
    - `alpha`, `beta`: Parameters for respective divergences

- **Outputs:**  
  - A scalar InfoLM score (or a tuple of corpus-level and sentence-level scores if configured)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Summarization, Data-to-Text Generation

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating semantic similarity between model outputs and references where strict lexical overlap is insufficient (e.g., abstractive summarization)
  - Scenarios requiring configurable evaluation sensitivity via divergence parameters

- **Not Recommended For:**  
  - Tasks where interpretability or faithfulness to input (e.g., hallucination detection) is paramount without further adaptation  
  - Cases lacking appropriate pre-trained MLMs for the domain or language

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/) (`torchmetrics.text.infolm.InfoLM`)
  - [Hugging Face Transformers](https://github.com/huggingface/transformers) (backend model support)

### Computational Complexity

- **Efficiency:**  
  Requires multiple forward passes through a masked language model, making it moderately expensive. Complexity depends on sequence length and model size.

- **Scalability:**  
  Batched processing is supported. Performance can be tuned via `batch_size`, `num_threads`, and `max_length` configuration options. May be limited by GPU memory for large models.

## Known Limitations

- **Biases:**  
  May inherit representational biases from the underlying masked language model (e.g., BERT), especially in domain-specific or low-resource contexts.

- **Task Misalignment Risks:**  
  If the task requires strict factual consistency or groundedness, InfoLM may not penalize hallucinations adequately.

- **Failure Cases:**  
  - Poor calibration when temperature or divergence parameters are not tuned appropriately  
  - Suboptimal behavior on short texts or uncommon vocabulary if IDF is not enabled

## Related Metrics

- **BERTScore:** Embedding-based similarity metric that compares contextual token embeddings using cosine similarity  
- **MoverScore:** Uses Earth Mover’s Distance over contextual embeddings  
- **BLEURT:** Supervised metric leveraging BERT with task-specific fine-tuning

## Further Reading

- **Papers:**  
  - [InfoLM: A New Metric to Evaluate Summarization & Data2Text Generation (Colombo et al., 2021)](https://api.semanticscholar.org/CorpusID:244896426)

- **Blogs/Tutorials:**  
  - [TorchMetrics Documentation on InfoLM](https://lightning.ai/docs/torchmetrics/stable/text/infolm/)

## Citation

```
@inproceedings{Colombo2021InfoLMAN,
  title={InfoLM: A New Metric to Evaluate Summarization \& Data2Text Generation},
  author={Pierre Colombo and Chloe Clave and Pablo Piantanida},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2021},
  url={https://api.semanticscholar.org/CorpusID:244896426}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu