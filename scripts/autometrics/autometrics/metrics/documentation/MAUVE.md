---
# Metric Card for MAUVE

MAUVE (Measuring the Alignment of Unconditional VErsions) quantifies the similarity between two text distributions (e.g., generated vs. human-written) by computing divergence frontiers based on KL divergences between their quantized representations in the embedding space of a large language model. MAUVE can capture nuanced differences due to model size, decoding strategy, or topic shift, and is especially useful in evaluating open-ended text generation tasks.

## Metric Details

### Metric Description

MAUVE evaluates how close the distribution of generated text is to that of human-written text. It operates by:
1. Embedding text samples using a large pretrained language model (e.g., GPT-2).
2. Applying PCA for dimensionality reduction and k-means for quantization to discretize the feature space.
3. Estimating histograms over quantized regions for both distributions.
4. Computing divergence curves using KL divergences of mixtures between these histograms.
5. Calculating the area under the divergence frontier curve to yield a scalar MAUVE score.

MAUVE can distinguish between generations of different quality, decoding strategies, and model sizes. The smoothed variant, MAUVE*, incorporates Krichevsky-Trofimov smoothing for improved robustness.

- **Metric Type:** Semantic Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

Let $p$ and $q$ be quantized histograms of two text distributions (e.g., reference and generated text). MAUVE is defined using divergence frontiers derived from KL divergences between convex combinations of $p$ and $q$.

Let $r = w p + (1 - w) q$ for $w \in (0,1)$, and define:

$$
\text{KL}(p \| r) = \sum _{i} p_i \log \frac{p_i}{r_i}, \quad \text{KL}(q \| r) = \sum _{i} q_i \log \frac{q_i}{r_i}
$$

The divergence curve consists of points:

$$
\left( \exp(-c \cdot \text{KL}(q \| r)), \exp(-c \cdot \text{KL}(p \| r)) \right)
$$

The MAUVE score is the symmetric area under the divergence curve:

$$
\text{MAUVE}(p, q) = \frac{1}{2} \left[ \text{AUC}(x, y) + \text{AUC}(y, x) \right]
$$

where $c$ is a scaling factor (default: 5), and AUC is the area under the curve computed using sorted points.

### Inputs and Outputs

- **Inputs:**  
  - `p_text`: list of generated texts (or `p_tokens`, `p_features`)
  - `q_text`: list of human/reference texts (or `q_tokens`, `q_features`)

- **Outputs:**  
  - `mauve`: scalar score ∈ [0, 1]  
  - `frontier_integral`: alternative distance measure (lower is better)  
  - `mauve_star`: smoothed variant of MAUVE  
  - `p_hist`, `q_hist`: quantized histograms  
  - `divergence_curve`: array of $(\text{KL}_q, \text{KL}_p)$ points

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Open-Ended Text Generation, Dialogue Generation, Storytelling, Summarization, Style Transfer

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating large-scale text generation systems where distributional similarity to human writing is important (e.g., language modeling, creative writing, summarization at scale).

- **Not Recommended For:**  
  Fine-grained sentence-level comparison or evaluation settings where token-level matching (e.g., translation) is prioritized.

- ⚠️ **Note on Current Usage in Metric Bank:**  
  In this metric bank, MAUVE is currently used in a setting with **one generated output and multiple references**, despite the original paper recommending **≥ 1000 samples per distribution** for reliable estimation. This usage is therefore **out-of-distribution for the intended application of MAUVE**. A future refactor may support true distributional evaluation, but users should interpret results with caution in the current setup.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [mauve-text (official)](https://github.com/krishnap25/mauve)
  - [HuggingFace Evaluate](https://huggingface.co/spaces/evaluate-metric/mauve)

### Computational Complexity

- **Efficiency:**  
  Moderately expensive due to LLM featurization and clustering; PCA and k-means add computational cost.

- **Scalability:**  
  Scales well to large datasets when using pre-computed features; authors recommend ≥1000 samples per distribution.

## Known Limitations

- **Biases:**  
  Embedding bias: MAUVE inherits any inductive biases present in the embedding model (e.g., GPT-2), which can affect distributional similarity estimates.

- **Task Misalignment Risks:**  
  May underestimate quality when high-quality text differs stylistically, topically, or length-wise from the human reference corpus.

- **Failure Cases:**  
  Cannot reliably detect subtle decoding changes (e.g., $p=0.95$ vs $p=0.96$ in top-p sampling); sensitive to seed and stochasticity in generation/clustering.

## Related Metrics

- **Fréchet Distance (FVD/TVD):** Similar in spirit for generative image evaluation.
- **BERTScore:** Evaluates semantic similarity using contextual embeddings but is sentence-level.
- **Self-BLEU / Distinct-n:** Capture diversity but not quality.
- **Perplexity:** Captures fluency but not alignment with a reference distribution.

## Further Reading

- **Papers:**  
  - [Pillutla et al., 2023 (JMLR)](https://www.jmlr.org/papers/v24/21-1214.html)  
  - [Pillutla et al., 2021 (NeurIPS)](https://arxiv.org/abs/2102.01454)  
  - [Liu et al., 2021 (NeurIPS)](https://arxiv.org/abs/2102.04130)

- **Blogs/Tutorials:**  
  - [MAUVE GitHub](https://github.com/krishnap25/mauve)  
  - [HuggingFace Metric Card](https://huggingface.co/spaces/evaluate-metric/mauve)

## Citation

```
@article{pillutla-etal:mauve:jmlr2023,  
  title={{MAUVE Scores for Generative Models: Theory and Practice}},  
  author={Pillutla, Krishna and Liu, Lang and Thickstun, John and Welleck, Sean and Swayamdipta, Swabha and Zellers, Rowan and Oh, Sewoong and Choi, Yejin and Harchaoui, Zaid},  
  journal={JMLR},  
  year={2023}  
}
```

```
@inproceedings{pillutla-etal:mauve:neurips2021,  
  title={MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers},  
  author={Pillutla, Krishna and Swayamdipta, Swabha and Zellers, Rowan and Thickstun, John and Welleck, Sean and Choi, Yejin and Harchaoui, Zaid},  
  booktitle = {NeurIPS},  
  year      = {2021}  
}
```

```
@inproceedings{liu-etal:mauve-theory:neurips2021,  
  title={{Divergence Frontiers for Generative Models: Sample Complexity, Quantization Effects, and Frontier Integrals}},  
  author={Liu, Lang and Pillutla, Krishna and Welleck, Sean and Oh, Sewoong and Choi, Yejin and Harchaoui, Zaid},  
  booktitle={NeurIPS},  
  year={2021}  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu