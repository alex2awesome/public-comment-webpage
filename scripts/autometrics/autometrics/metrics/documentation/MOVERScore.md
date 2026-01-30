---
# Metric Card for MoverScore

MoverScore is a semantic similarity metric for evaluating generated text, leveraging contextualized embeddings (such as BERT) and Earth Mover’s Distance (EMD) to measure the alignment between system outputs and reference texts. It is designed to capture semantic similarity beyond lexical overlap and has been shown to achieve a high correlation with human judgments across tasks like machine translation, summarization, image captioning, and data-to-text generation.

## Metric Details

### Metric Description

MoverScore measures text similarity by computing the minimum cost required to move the distributed representations of words from the generated text to the reference text. It uses contextualized word embeddings (from models like BERT) and optimizes a transport cost matrix to determine the most efficient mapping between words. Unlike surface-level metrics such as BLEU and ROUGE, MoverScore accounts for semantic equivalence even when lexical forms differ.

- **Metric Type:** Semantic Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No (evaluates only system output and reference)
  
### Formal Definition

MoverScore extends **Word Mover’s Distance (WMD)** by incorporating contextualized embeddings. Given a generated sentence $x$ and a reference sentence $y$, let $x_n$ and $y_n$ represent their n-grams. The distance between these sentences is computed as:

$$
\text{WMD}(x _{n}, y _{n}) = \min _{F} \sum _{i,j} C _{ij} F _{ij}
$$

subject to:

$$
F 1 = f _{x _{n}}, \quad F^T 1 = f _{y _{n}}
$$

where:
- $F$ is the transportation flow matrix,
- $C _{ij}$ is the Euclidean distance between the embeddings of n-grams $x _{i}^{n}$ and $y _{j}^{n}$,
- $f _{x _{n}}$ and $f _{y _{n}}$ represent n-gram weight distributions, computed using inverse document frequency (IDF).

MoverScore supports multiple variations, including **Word Mover Distance (WMD) on unigrams/bigrams** and **Sentence Mover Distance (SMD)**.

### Inputs and Outputs

- **Inputs:**  
  - Generated text (system output)  
  - Reference text(s) (gold-standard text)

- **Outputs:**  
  - A scalar similarity score in the range [0,1], where higher values indicate better semantic alignment.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Machine Translation, Summarization, Image Captioning, Data-to-Text Generation

### Applicability and Limitations

- **Best Suited For:**  
  - Tasks where capturing semantic similarity is critical, such as summarization, paraphrasing, and machine translation.  
  - Cases where lexical overlap is insufficient to judge text quality (e.g., abstractive summarization).  

- **Not Recommended For:**  
  - Evaluating grammatical correctness or fluency in isolation.  
  - Tasks where exact lexical matching is the primary evaluation criterion (e.g., extractive summarization).  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [MoverScore GitHub Repository](https://github.com/AIPHES/emnlp19-moverscore)  
  - Available in `moverscore.py` (original) and `moverscore_v2.py` (faster but less accurate version).

### Computational Complexity

- **Efficiency:**  
  - MoverScore is computationally more expensive than n-gram-based metrics due to the need for contextualized embeddings and solving an optimal transport problem.
  
- **Scalability:**  
  - Can be slow for large datasets but optimized versions exist (`moverscore_v2.py` uses DistilBERT for speed).

## Known Limitations

- **Biases:**  
  - Performance depends on the pre-trained language model used. Models trained on large-scale English corpora may not generalize well to low-resource languages.
  
- **Task Misalignment Risks:**  
  - May not accurately reflect human preferences when lexical precision is crucial.
  
- **Failure Cases:**  
  - Contextual embeddings may not effectively capture domain-specific terminology or named entity variations.
  
## Related Metrics

- **BERTScore:** Similar to MoverScore but relies on cosine similarity rather than Earth Mover’s Distance.
- **BLEU, ROUGE:** Traditional surface-level n-gram overlap metrics, which MoverScore seeks to improve upon.
- **CIDEr, METEOR:** Alternative semantic similarity-based evaluation metrics.

## Further Reading

- **Papers:**  
  - Zhao et al., 2019. *MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance.* [EMNLP 2019](https://arxiv.org/abs/1909.02622)  
  - Peyrard et al., 2019. *Supervised and Unsupervised Metrics for Machine Translation and Summarization.*

- **Blogs/Tutorials:**  
  - [MoverScore GitHub README](https://github.com/AIPHES/emnlp19-moverscore)  
  - [Evaluating Text Generation with MoverScore](https://arxiv.org/pdf/1909.02622.pdf)

## Citation

```
@inproceedings{zhao-etal-2019-moverscore,
    title = "{M}over{S}core: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance",
    author = "Zhao, Wei  and
      Peyrard, Maxime  and
      Liu, Fei  and
      Gao, Yang  and
      Meyer, Christian M.  and
      Eger, Steffen",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1053/",
    doi = "10.18653/v1/D19-1053",
    pages = "563--578",
    abstract = "A robust evaluation metric has a profound impact on the development of text generation systems. A desirable metric compares system output against references based on their semantics rather than surface forms. In this paper we investigate strategies to encode system and reference texts to devise a metric that shows a high correlation with human judgment of text quality. We validate our new metric, namely MoverScore, on a number of text generation tasks including summarization, machine translation, image captioning, and data-to-text generation, where the outputs are produced by a variety of neural and non-neural systems. Our findings suggest that metrics combining contextualized representations with a distance measure perform the best. Such metrics also demonstrate strong generalization capability across tasks. For ease-of-use we make our metrics available as web service."
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  