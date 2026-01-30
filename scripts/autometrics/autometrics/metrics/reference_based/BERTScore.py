import bert_score
import torch
import warnings
from typing import ClassVar
from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import ReferenceBasedMultiMetric

def compute_bertscore(original, output, references, model="roberta-large", type="all", compare_to_original=False):

    all_origs, all_refs, all_cands = [], [], []
    for orig, hyp, refs in zip(original, output, references):
        for ref in refs:
            all_refs.append(ref.lower())
            all_cands.append(hyp.lower())
            all_origs.append(orig.lower())

    def _score(cands, refs):
        return bert_score.score(
            cands, refs, lang="en", return_hash=True, verbose=False, idf=False, model_type=model
        )

    try:
        if compare_to_original:
            (P, R, F), _ = _score(all_cands, all_origs)
        else:
            (P, R, F), _ = _score(all_cands, all_refs)
    except RuntimeError as e:
        if 'device-side assert' in str(e).lower():
            warnings.warn("[BERTScore] CUDA device-side assert; retrying BERTScore on CPU.")
            # Best-effort CPU fallback by temporarily disabling CUDA
            prior = torch.cuda.is_available()
            try:
                if prior:
                    torch.cuda.empty_cache()
                if compare_to_original:
                    (P, R, F), _ = _score(all_cands, all_origs)
                else:
                    (P, R, F), _ = _score(all_cands, all_refs)
            except Exception:
                raise
        else:
            raise

    ind = 0
    pscores = []
    rscores = []
    fscores = []
    for orig, out, refs in zip(original, output, references):
        sub_pscores = []
        sub_rscores = []
        sub_fscores = []
        for _ in refs:
            sub_fscores.append(F[ind].item())
            sub_pscores.append(P[ind].item())
            sub_rscores.append(R[ind].item())
            ind += 1
        pscores.append(max(sub_pscores))
        rscores.append(max(sub_rscores))
        fscores.append(max(sub_fscores))

    assert len(pscores) == len(rscores) == len(fscores) == len(output) == len(references) == len(original)
    
    if type == "precision":
        return pscores
    elif type == "recall":
        return rscores
    elif type == "f1":
        return fscores
    elif type == "all":
        return pscores, rscores, fscores

class BERTScore(ReferenceBasedMultiMetric):
    """---
# Metric Card for BERTScore

BERTScore is a semantic similarity metric for evaluating generated text against reference text. It leverages pre-trained contextual embeddings (e.g., BERT, RoBERTa) to compute token-level cosine similarity, measuring precision, recall, and F1 scores. BERTScore is particularly effective in capturing semantic equivalence and correlates well with human judgments, making it a versatile metric for various text generation tasks.

## Metric Details

### Metric Description

BERTScore evaluates the semantic similarity between a generated text and a reference text using contextual embeddings. Unlike traditional n-gram-based metrics (e.g., BLEU), which rely on surface-level token overlap, BERTScore uses pre-trained embeddings to capture the contextual meaning of tokens. 

The metric computes cosine similarity for each token pair between the reference and generated text, with optional inverse document frequency (IDF) weighting to emphasize rare tokens. The precision, recall, and F1 scores are calculated by aggregating the maximum similarity scores for each token, and an optional baseline rescaling makes the scores more interpretable.

- **Metric Type:** Semantic Similarity
- **Range:** Typically [0, 1] after rescaling
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

For a reference sentence $x = \langle x_1, \dots, x_k \rangle$ and a candidate sentence $\hat{x} = \langle \hat{x}_1, \dots, \hat{x}_l \rangle$, the BERTScore components are defined as:

$$
R_{\text{BERT}} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} x_i^\top \hat{x}_j
$$

$$
P_{\text{BERT}} = \frac{1}{|x|} \sum_{x_j \in \hat{x}_{j} } \max _{x_i \in x} x_i^\top \hat{x}_j
$$

$$
F_{\text{BERT}} = \frac{2 \cdot P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
$$

Here, $x_i$ and $\hat{x}_j$ represent the contextual embeddings of the tokens, and the similarity is computed using cosine similarity.

With IDF weighting, recall is modified as:

Recall Modified:

$$
R_{\text{BERT}} = \frac{\sum _{x_i \in x} \text{idf}(x_i) \cdot \max _{\hat{x}_j \in \hat{x}} x_i^\top \hat{x}_j}{\sum _{x_i \in x} \text{idf}(x_i)}
$$

Baseline rescaling adjusts scores to lie within [0, 1].

### Inputs and Outputs

- **Inputs:**  
  - Generated text (candidate)  
  - Reference text(s)  
  - Optional: IDF weights for importance weighting  

- **Outputs:**  
  - Scalar precision, recall, and F1 scores  

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems, Image Captioning
- **Tasks:** Machine Translation, Summarization, Paraphrasing, Image-to-Text Generation

### Applicability and Limitations

- **Best Suited For:**  
  - Tasks requiring semantic similarity evaluation between generated and reference texts.  
  - Use cases where semantic correctness is prioritized over lexical overlap.  

- **Not Recommended For:**  
  - Open-ended or highly creative generation tasks with diverse acceptable outputs (e.g., storytelling).  
  - Domains with very low-resource or out-of-domain embeddings.  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [BERTScore GitHub Repository](https://github.com/Tiiiger/bert_score)  
  - [Hugging Face `evaluate`](https://huggingface.co/docs/evaluate)  

### Computational Complexity

- **Efficiency:**  
  BERTScore is computationally intensive due to the use of contextual embeddings. A GPU is recommended for large-scale evaluations.

- **Scalability:**  
  Supports multiple languages and embeddings. Processing speed varies based on embedding size and sentence length.

## Known Limitations

- **Biases:**  
  - Performance may degrade for low-resource languages.  
  - Contextual embeddings may reflect biases present in the pre-trained models.  

- **Task Misalignment Risks:**  
  - Poor performance on tasks emphasizing diversity or creativity.  

- **Failure Cases:**  
  - Struggles with very long sentences due to truncation in transformer models.  
  - Sensitivity to embedding model choice and layer selection.  

## Related Metrics

- **BLEU:** Focuses on surface-level similarity using n-grams.  
- **ROUGE:** Often used for summarization but lacks semantic understanding.  
- **METEOR:** Incorporates synonyms but is limited in language coverage.  
- **CHRF:** Uses character-level n-grams for lexical similarity.  

## Further Reading

- **Papers:**  
  - [BERTScore: Evaluating Text Generation with BERT (Zhang et al., 2020)](https://arxiv.org/abs/1904.09675)  

- **Blogs/Tutorials:**  
  - [BERTScore GitHub Documentation](https://github.com/Tiiiger/bert_score) 
  
## Citation

```
@inproceedings{Zhang2020BERTScore,
   title={BERTScore: Evaluating Text Generation with BERT},
   author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
   booktitle={International Conference on Learning Representations},
   year={2020},
   url={https://openreview.net/forum?id=SkeHuCVFDr}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 8.125 # in MB
    cpu_mem: ClassVar[float] = 1512.303487270343 # in MB
    description: ClassVar[str] = "BERTScore is a semantic similarity metric for evaluating generated text against reference text. It leverages pre-trained contextual embeddings (e.g., BERT, RoBERTa) to compute token-level cosine similarity, measuring precision, recall, and F1 scores. BERTScore is particularly effective in capturing semantic equivalence and correlates well with human judgments, making it a versatile metric for various text generation tasks."

    def __init__(self, model="roberta-large", **kwargs):
        name = "BERTScore_" + model
        description = "BERTScore is a semantic similarity metric for evaluating generated text against reference text. It leverages pre-trained contextual embeddings (e.g., BERT, RoBERTa) to compute token-level cosine similarity, measuring precision, recall, and F1 scores. BERTScore is particularly effective in capturing semantic equivalence and correlates well with human judgments, making it a versatile metric for various text generation tasks."
        self.model = model

        submetrics = ["P", "R", "F"]
        
        # Pass model parameter explicitly to parent constructor for caching
        super().__init__(
            name=name, 
            description=description, 
            model=model,  # Pass model param explicitly for caching
            submetric_names=[f"BERTScore{submetric}_{model}" for submetric in submetrics], 
            **kwargs
        )
        
        # No need to explicitly register parameters anymore - they're automatically included in cache key
        
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Actual implementation of BERTScore batch calculation
        """
        if references is None:
            references = [None] * len(inputs)

        return compute_bertscore(inputs, outputs, references, model=self.model, type="all")
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Actual implementation of BERTScore for a single input/output pair
        """
        if references is None:
            references = []

        p, r, f = compute_bertscore([input], [output], [references], model=self.model, type="all")
        return p[0], r[0], f[0]