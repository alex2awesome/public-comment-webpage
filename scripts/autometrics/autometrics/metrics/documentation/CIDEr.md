---
# Metric Card for CIDEr

CIDEr (Consensus-based Image Description Evaluation) measures the similarity between a candidate image caption and a set of human-generated reference captions. It leverages TF-IDF weighted n-gram matching to capture consensus in both content and linguistic style, making it especially useful for benchmarking image captioning systems.

## Metric Details

### Metric Description

CIDEr evaluates how well a candidate caption matches the consensus of human descriptions by comparing n-gram counts weighted by TF-IDF. In this process, each sentence is decomposed into n-grams (typically from unigrams up to four-grams), and each n-gram is weighted according to its term frequency in the sentence and its inverse document frequency across a dataset of images. The final score is the weighted average of cosine similarities computed for each n-gram length. An extended version, CIDEr-D, incorporates additional mechanisms (e.g., clipping and a Gaussian length penalty) to mitigate potential gaming of the metric.

- **Metric Type:** Surface-Level Similarity
- **Range:** Typically 0 to approximately 10 (higher values indicate better consensus)
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

For each n-gram $\omega_k$ in a reference sentence $s_{ij}$, the TF-IDF weighting is computed as:

$$
g_k(s_{ij}) = \frac{h_k(s_{ij})}{\sum_{\omega_l \in \Omega} h_l(s_{ij})} \log \left( \frac{|I|}{\sum_{p \in I} \min \left(1, \sum_q h_k(s_{pq}) \right)} \right)
$$

where:
- $h_k(s_{ij})$ is the count of n-gram $\omega_k$ in $s_{ij}$,
- $\Omega$ is the vocabulary of all n-grams,
- $|I|$ is the total number of images in the dataset, and
- the denominator in the log computes the document frequency of $\omega_k$.

The n-gram level CIDEr score is defined as:

$$
CIDEr_n(c_i, S_i) = \frac{1}{m} \sum_j \frac{g_n(c_i) \cdot g_n(s_{ij})}{\| g_n(c_i) \| \, \| g_n(s_{ij}) \|}
$$

where:
- $c_i$ is the candidate caption,
- $S_i = \{ s_{i1}, s_{i2}, \dots, s_{im} \}$ is the set of reference captions,
- $g_n(c_i)$ is the TF-IDF vector for n-grams of length $n$ from $c_i$, and
- $\| \cdot \|$ denotes the Euclidean norm.

The overall CIDEr score aggregates the n-gram scores:

$$
CIDEr(c_i, S_i) = \sum_{n=1}^N w_n \, CIDEr_n(c_i, S_i)
$$

with uniform weights $w_n = \frac{1}{N}$ (typically $N=4$).

### Inputs and Outputs

- **Inputs:**  
  - Candidate caption (a string)  
  - A set of reference captions (a list of strings)

- **Outputs:**  
  - A scalar CIDEr score representing the degree of consensus between the candidate caption and the references

## Intended Use

### Domains and Tasks

- **Domain:** Image Captioning, Multimodal Generation
- **Tasks:** Image Description Evaluation

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating image captioning systems where a rich set of human reference captions (e.g., 50 per image) is available. CIDEr is particularly effective for assessing how well a candidate caption captures the consensus of human descriptions.

- **Not Recommended For:**  
  Open-ended generation tasks (e.g., creative storytelling or dialogue) where high lexical diversity is expected, or scenarios lacking sufficient reference data.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [coco-caption repository](https://github.com/tylin/coco-caption) (reference implementation)  

### Computational Complexity

- **Efficiency:**  
  CIDEr is efficient at the sentence level; it primarily involves n-gram extraction, TF-IDF weighting, and cosine similarity calculations.

- **Scalability:**  
  While the computation scales linearly with the number of reference captions, performance benefits from additional references up to a saturation point.

## Known Limitations

- **Biases:**  
  - May favor candidate captions that closely mimic the majority of reference texts, potentially penalizing valid paraphrases or creative expressions.
  
- **Task Misalignment Risks:**  
  - Less effective for tasks with high variability in acceptable outputs, such as creative writing or dialogue generation.
  
- **Failure Cases:**  
  - When candidate captions use synonyms or alternative phrasings not reflected in the reference set, CIDEr may under-score these valid variations.

## Related Metrics

- **BLEU:** Focuses on n-gram precision without TF-IDF weighting.  
- **ROUGE:** Emphasizes n-gram recall.  
- **METEOR:** Incorporates stemming and synonym matching for enhanced semantic alignment.  
- **CIDEr-D:** A variant of CIDEr with added clipping and length penalty to reduce susceptibility to metric gaming.

## Further Reading

- **Papers:**  
  - Vedantam, R., Zitnick, C. L., & Parikh, D. (2015). *CIDEr: Consensus-based Image Description Evaluation*. [Original paper, arXiv:1411.5726](https://arxiv.org/abs/1411.5726)
  
- **Blogs/Tutorials:**  
  - Documentation and tutorials available on the coco-caption GitHub repository

## Citation

```
@INPROCEEDINGS{7299087,
  author={Vedantam, Ramakrishna and Zitnick, C. Lawrence and Parikh, Devi},
  booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={CIDEr: Consensus-based image description evaluation}, 
  year={2015},
  volume={},
  number={},
  pages={4566-4575},
  keywords={Measurement;Protocols;Accuracy;Training;Testing;Silicon;Correlation},
  doi={10.1109/CVPR.2015.7299087}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:** Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high), based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu