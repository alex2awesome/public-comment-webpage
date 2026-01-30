import torch
from typing import List, Union, ClassVar
import warnings
from lens import download_model
from lens import LENS as LENS_original
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class LENS(ReferenceBasedMetric):
    """---
# Metric Card for LENS

LENS (Learnable Evaluation Metric for Text Simplification) is a reference-based metric designed specifically to evaluate system outputs in the task of text simplification. It aligns with human judgments more closely than prior metrics by learning from human ratings using a mixture-of-experts (MoE) model, which captures multiple aspects of simplification quality, such as grammaticality, meaning preservation, and simplicity. LENS can be rescaled between 0 and 100 for interpretability.

## Metric Details

### Metric Description

LENS evaluates text simplification quality by comparing a system-generated simplification against both the complex source sentence and one or more human-written simplifications (references). It is trained to regress toward average human judgments across three dimensions: grammaticality, meaning preservation, and simplicity. 

To capture these aspects, LENS uses a mixture-of-experts model built atop sentence-level and word-level representations from a pre-trained encoder (T5 encoder). Each expert corresponds to a latent factor presumed to model a subset of simplification phenomena. LENS is trained on human-annotated ratings from multiple datasets, and the resulting model provides a scalar score aligned with holistic simplification quality.

- **Metric Type:** Semantic Similarity
- **Range:** $\mathbb{R}$ (rescaled to [0, 100] for interpretability)
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Given a source sentence $C$, a system output simplification $S$, and one or more reference simplifications $R = \{r_1, \dots, r_n\}$, the LENS score is computed as follows:

1. Encode the triplet $(C, S, R)$ using the T5 encoder to obtain sentence-level and word-level embeddings.
2. Pass these embeddings through $K$ expert scoring heads, each of which outputs a scalar.
3. Use a gating network to produce weights $w_1, \dots, w_K$ over the experts based on the input.
4. Compute the final score as the weighted combination of expert predictions:

$$
\text{LENS}(C, S, R) = \sum _{k=1}^K w_k \cdot f_k(C, S, R)
$$

where $f_k$ is the $k$-th expert head's output.

### Inputs and Outputs

- **Inputs:**  
  - Complex sentence (source input)  
  - Simplified sentence (system output)  
  - Reference simplifications (1 or more)

- **Outputs:**  
  - Scalar LENS score, either in original form (unbounded real number) or rescaled between 0 and 100.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Text Simplification

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating English text simplification outputs, particularly when references are available and multiple quality dimensions (e.g., fluency, meaning, simplicity) are relevant.

- **Not Recommended For:**  
  Tasks that are not simplification-specific (e.g., translation, paraphrasing) or that lack appropriate reference simplifications. LENS is not designed for creative generation or tasks involving high lexical diversity.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [LENS GitHub](https://github.com/davidheineman/lens) (official implementation)  
  - [Hugging Face Model Hub - davidheineman/lens](https://huggingface.co/davidheineman/lens)  

### Computational Complexity

- **Efficiency:**  
  Moderate. Inference uses a pre-trained encoder and multiple expert heads, with computational cost comparable to standard encoder-forward passes in T5.

- **Scalability:**  
  Scales adequately with batching; suitable for GPU-based batched evaluation but may be slower than traditional lexical metrics.

## Known Limitations

- LENS was trained on English simplification datasets and may not generalize to other languages without retraining.
- It requires both source and reference inputs, limiting its use in reference-free or source-free settings.

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  Not suitable for tasks like summarization or paraphrasing without retraining or adaptation.

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- **SARI:** A lexical overlap-based metric for simplification, focusing on additions, deletions, and retention.
- **BLEU/ROUGE:** Often used but poorly aligned with human judgments in simplification tasks.
- **BERTScore:** Captures semantic similarity but is not simplification-specific.
- **QuestEval:** General-purpose learned evaluation, not optimized for simplification.

## Further Reading

- **Papers:**  
  - [LENS: A Learnable Evaluation Metric for Text Simplification (Maddela et al., 2023)](https://aclanthology.org/2023.acl-long.905)

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{maddela-etal-2023-lens,
  title = "{LENS}: A Learnable Evaluation Metric for Text Simplification",
  author = "Maddela, Mounica  and
    Dou, Yao  and
    Heineman, David  and
    Xu, Wei",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2023",
  address = "Toronto, Canada",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.acl-long.905",
  doi = "10.18653/v1/2023.acl-long.905",
  pages = "16383--16408",
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    # TODO: Check this, because gpu memory being zero for a model is suspicious
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 3330.29296875  # in MB
    description: ClassVar[str] = "LENS (Learnable Evaluation Metric for Text Simplification) is a reference-based metric designed specifically to evaluate system outputs in the task of text simplification. It aligns with human judgments more closely than prior metrics by learning from human ratings using a mixture-of-experts (MoE) model, which captures multiple aspects of simplification quality, such as grammaticality, meaning preservation, and simplicity. LENS can be rescaled between 0 and 100 for interpretability."

    def __init__(
        self,
        name: str = "LENS",
        description: str = "LENS (Learnable Evaluation Metric for Text Simplification) is a reference-based metric designed specifically to evaluate system outputs in the task of text simplification. It aligns with human judgments more closely than prior metrics by learning from human ratings using a mixture-of-experts (MoE) model, which captures multiple aspects of simplification quality, such as grammaticality, meaning preservation, and simplicity. LENS can be rescaled between 0 and 100 for interpretability.",
        model_id: str = "davidheineman/lens",
        rescale: bool = True,
        batch_size: int = 16,
        devices: List[int] = None,
        persistent: bool = True,
        **kwargs
    ):
        super().__init__(name, description, model_id=model_id, rescale=rescale, batch_size=batch_size, devices=devices, persistent=persistent, **kwargs)
        self.model_id = model_id
        self.rescale = rescale
        self.batch_size = batch_size
        self.devices = devices
        self.persistent = persistent
        self.model = None

        self.exclude_from_cache_key("batch_size", "devices", "persistent")

    def _load_model(self):
        """Download checkpoint and load the LENS model."""
        if self.model is None:
            ckpt_path = download_model(self.model_id)
            try:
                self.model = LENS_original(ckpt_path, rescale=self.rescale)
            except RuntimeError as e:
                # Guard against device-side assertions during checkpoint load
                if 'device-side assert' in str(e).lower():
                    warnings.warn("[LENS] CUDA device-side assert during checkpoint load; retrying with CPU map_location.")
                    # Monkeypatch torch.load to force CPU map_location just for this load
                    import torch as _torch
                    orig_load = _torch.load
                    def _forced_cpu_load(*args, **kwargs):
                        kwargs.setdefault('map_location', 'cpu')
                        return orig_load(*args, **kwargs)
                    _torch.load = _forced_cpu_load
                    try:
                        self.model = LENS_original(ckpt_path, rescale=self.rescale)
                    finally:
                        _torch.load = orig_load
                else:
                    raise

    def _unload_model(self):
        """Unload model to free resources."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def _calculate_impl(self, input: str, output: str, references: List[str], **kwargs) -> float:
        """
        Compute LENS score for one example.
        """
        if self.model is None:
            self._load_model()
        # wrap inputs for single sample
        try:
            score = self.model.score(
                [input], [output], [references],
                batch_size=self.batch_size,
                devices=self.devices
            )
        except RuntimeError as e:
            if 'device-side assert' in str(e).lower():
                warnings.warn("[LENS] CUDA device-side assert during scoring; retrying with batch_size=1 and default devices.")
                # Retry with smaller batch and default device handling
                try:
                    score = self.model.score(
                        [input], [output], [references],
                        batch_size=1,
                        devices=None
                    )
                except RuntimeError:
                    warnings.warn("[LENS] Fallback scoring also failed; propagating error.")
                    raise
            else:
                raise
        result = float(score[0])
        if not self.persistent:
            self._unload_model()
        return result

    def _calculate_batched_impl(
        self,
        inputs: List[str],
        outputs: List[str],
        references: List[List[str]],
        **kwargs
    ) -> List[float]:
        """
        Compute LENS scores for a batch of samples.
        """
        if self.model is None:
            self._load_model()
        try:
            scores = self.model.score(
                inputs, outputs, references,
                batch_size=self.batch_size,
                devices=self.devices
            )
        except RuntimeError as e:
            if 'device-side assert' in str(e).lower():
                warnings.warn("[LENS] CUDA device-side assert during batched scoring; retrying with batch_size=1 and default devices.")
                try:
                    scores = self.model.score(
                        inputs, outputs, references,
                        batch_size=1,
                        devices=None
                    )
                except RuntimeError:
                    warnings.warn("[LENS] Batched fallback also failed; propagating error.")
                    raise
            else:
                raise
        results = [float(s) for s in scores]
        if not self.persistent:
            self._unload_model()
        return results 