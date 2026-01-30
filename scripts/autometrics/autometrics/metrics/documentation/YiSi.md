---
# Metric Card for YiSi-2

YiSi-2 is a semantic machine translation evaluation and quality estimation metric designed for cross-lingual assessment. It measures the semantic similarity between the source sentence and the machine-translated output using bilingual word embeddings, optionally incorporating shallow semantic structures through semantic role labeling.

## Metric Details

### Metric Description

YiSi-2 is the bilingual, reference-less variant of the YiSi metric family. It evaluates machine translation output quality by comparing the translated sentence to the original input using cross-lingual semantic similarity. The metric relies on a shared bilingual embedding space (e.g., from multilingual BERT) to align and compare words across languages. It optionally incorporates shallow semantic parsing to capture structural meaning via aligned semantic frames and roles.

Lexical similarity between tokens is computed using cosine similarity of their embeddings. These similarities are aggregated using inverse document frequency (IDF)-weighted pooling to compute precision and recall scores. YiSi-2 then combines these using an $F_\alpha$-like formulation.

- **Metric Type:** Semantic Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let $E$ be the machine translation (MT) output and $F$ the source input sentence. Each word or subword token $e_i \in E$ and $f_j \in F$ is embedded into a shared bilingual embedding space using a cross-lingual language model. Define:

- $v(e_i)$: embedding of token $e_i$
- $s(e_i, f_j) = \cos(v(e_i), v(f_j))$: cosine similarity
- $w(e_i), w(f_j)$: IDF weights for $e_i$ and $f_j$

Let $\max_{f_j} s(e_i, f_j)$ be the maximum similarity for $e_i$ to any token in $F$, and analogously for $f_j$.

Define weighted precision and recall as:

$$
\text{Precision} = \frac{\sum _{e_i \in E} w(e_i) \cdot \max _{f_j \in F} s(e_i, f_j)}{\sum _{e_i \in E} w(e_i)}
$$

$$
\text{Recall} = \frac{\sum _{f_j \in F} w(f_j) \cdot \max _{e_i \in E} s(f_j, e_i)}{\sum _{f_j \in F} w(f_j)}
$$

Finally, the YiSi-2 score is computed as:

$$
\text{YiSi-2} = \frac{\text{Precision} \cdot \text{Recall}}{\alpha \cdot \text{Precision} + (1 - \alpha) \cdot \text{Recall}}
$$

with typical $\alpha = 0.8$ for MT evaluation, or $\alpha = 0.5$ for MT optimization.

### Inputs and Outputs

- **Inputs:**  
  - Input sentence (original source text)  
  - Generated text (machine translation output)  

- **Outputs:**  
  - Scalar YiSi-2 score (range: 0 to 1)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Machine Translation, Quality Estimation

### Applicability and Limitations

- **Best Suited For:**  
  - MT quality estimation
  - Cross-lingual evaluation between languages with shared multilingual embeddings

- **Not Recommended For:**  
  - Fluency evaluation (due to lack of monolingual reference)  
  - Evaluation of tasks that require generation in the same language as the input

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Original YiSi GitHub (C++)](https://github.com/chikiulo/yisi)
  - [Metametrics Python adaptation](https://github.com/meta-metrics/metametrics/blob/main/src/metametrics/metrics/yisi_metric.py)

### Computational Complexity

- **Efficiency:**  
  YiSi-2 requires forward passes through a cross-lingual encoder and pairwise cosine similarity computation, making it more computationally intensive than surface-level metrics. Complexity scales with sentence length and batch size.

- **Scalability:**  
  Batch inference and IDF precomputation are used to improve scalability. Still, performance may degrade on long or high-throughput datasets if GPU memory is limited.

## Known Limitations

- **Biases:**  
  - Inherits biases from the underlying multilingual embedding model (e.g., mBERT).
  - IDF weights computed from the input/MT corpus may not generalize across domains.

- **Task Misalignment Risks:**  
  - YiSi-2 may not capture fluency or grammar issues due to its reliance on embedding-level semantics.
  - It may reward semantically related outputs that are poorly formed linguistically.

- **Failure Cases:**  
  - Poor performance in language pairs with weak cross-lingual alignment in embeddings.
  - Degraded reliability if the embedding model is not aligned well across input and output languages.

## Related Metrics

- **YiSi-1:** Monolingual, reference-based variant using contextual embeddings  
- **BLEU / chrF:** Surface-level n-gram overlap metrics  
- **BERTScore:** Reference-based metric using contextual embeddings (same-language only)  
- **COMET-QE:** Reference-free MT quality estimation using a learned model

## Further Reading

- **Papers:**  
  - [Lo (2019) – YiSi: A Unified Semantic MT Metric (WMT)](https://aclanthology.org/W19-5358.pdf)

- **Blogs/Tutorials:**  
  - [MachineTranslate YiSi Overview](https://machinetranslate.org/metrics#yisi)

## Citation

```
@inproceedings{lo-2019-yisi,
    title = "{Y}i{S}i - a Unified Semantic {MT} Quality Evaluation and Estimation Metric for Languages with Different Levels of Available Resources",
    author = "Lo, Chi-kiu",
    editor = "Bojar, Ond{\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\'e}  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\'e}v{\'e}ol, Aur{\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Turchi, Marco  and
      Verspoor, Karin",
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5358/",
    doi = "10.18653/v1/W19-5358",
    pages = "507--513"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu