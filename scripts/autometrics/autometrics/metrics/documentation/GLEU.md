---
# Metric Card for GLEU

GLEU (Google-BLEU) is a metric introduced to address limitations of BLEU for sentence-level evaluation. It is designed to compute recall and precision over n-grams for hypotheses and references, taking the minimum of these two values as the final score. The metric is symmetrical and ranges from 0 (no match) to 1 (perfect match). It was initially proposed in Google's Neural Machine Translation (GNMT) system for reinforcement learning experiments.

## Metric Details

### Metric Description

GLEU computes sentence-level evaluation scores by comparing n-grams (of lengths 1 to 4) in the hypothesis and reference sentences. It calculates the precision (matching n-grams over total n-grams in the hypothesis) and recall (matching n-grams over total n-grams in the reference) and uses the minimum of the two values to determine the GLEU score. This approach avoids issues with BLEU's sentence-level evaluation while maintaining a high correlation with corpus-level BLEU scores.

- **Metric Type:** Surface-Level Similarity  
- **Range:** 0 to 1  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** No  

### Formal Definition

The GLEU score for a hypothesis $h$ and a set of reference sentences $\{r_1, r_2, \ldots, r_n\}$ is defined as:

$$
GLEU(h, R) = \min \left( \text{precision}, \text{recall} \right)
$$

Where:
- **Precision:** $\frac{\text{Number of matching n-grams}}{\text{Total n-grams in hypothesis}}$  
- **Recall:** $\frac{\text{Number of matching n-grams}}{\text{Total n-grams in reference}}$  

The final score is symmetrical with respect to hypothesis and reference, making it robust for single-sentence evaluation.

### Inputs and Outputs

- **Inputs:**  
  - Hypothesis sentence (generated text)  
  - Reference sentence(s) (gold-standard text)  

- **Outputs:**  
  - A scalar score in the range [0, 1], where 1 indicates a perfect match.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation, Summarization  

### Applicability and Limitations

- **Best Suited For:**  
  Sentence-level evaluation in structured tasks where precision and recall over n-grams are meaningful indicators of quality, such as translation.  

- **Not Recommended For:**  
  Creative or open-ended text generation tasks where semantic similarity or diversity is more relevant than surface-level n-gram overlap.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [NLTK GLEU Implementation](https://github.com/nltk/nltk/blob/develop/nltk/translate/gleu_score.py)  

### Computational Complexity

- **Efficiency:**  
  Efficient for sentence-level evaluation, as it requires simple n-gram matching and aggregation.

- **Scalability:**  
  Scales well for batch evaluations but may be computationally expensive for larger corpora due to repeated n-gram matching.

## Known Limitations

- **Biases:**  
  Penalizes valid paraphrases or semantically equivalent outputs that differ in n-gram overlap.  

- **Task Misalignment Risks:**  
  Designed for tasks with a single correct output structure; performs poorly for evaluating diverse or creative responses.  

- **Failure Cases:**  
  - GLEU may not adequately evaluate cases where semantic preservation is more important than lexical overlap.

## Related Metrics

- **BLEU:** GLEU is inspired by BLEU but designed for sentence-level evaluation.  
- **METEOR:** Aims to improve on BLEU by incorporating synonym matching.  
- **BERTScore:** Evaluates semantic similarity using contextual embeddings.  

## Further Reading

- **Papers:**  
  - [Google’s Neural Machine Translation System (Wu et al., 2016)](https://arxiv.org/pdf/1609.08144v2.pdf)  

- **Blogs/Tutorials:**  
  - Needs more information  

## Citation

```
@misc{wu2016googlesneuralmachinetranslation,
      title={Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation}, 
      author={Yonghui Wu and Mike Schuster and Zhifeng Chen and Quoc V. Le and Mohammad Norouzi and Wolfgang Macherey and Maxim Krikun and Yuan Cao and Qin Gao and Klaus Macherey and Jeff Klingner and Apurva Shah and Melvin Johnson and Xiaobing Liu and Łukasz Kaiser and Stephan Gouws and Yoshikiyo Kato and Taku Kudo and Hideto Kazawa and Keith Stevens and George Kurian and Nishant Patil and Wei Wang and Cliff Young and Jason Smith and Jason Riesa and Alex Rudnick and Oriol Vinyals and Greg Corrado and Macduff Hughes and Jeffrey Dean},
      year={2016},
      eprint={1609.08144},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1609.08144}, 
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu