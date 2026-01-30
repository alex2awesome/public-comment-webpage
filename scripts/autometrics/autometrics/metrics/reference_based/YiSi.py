import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Union, ClassVar
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric


# Credit: Adapted from metametrics (https://github.com/meta-metrics/metametrics/blob/main/src/metametrics/metrics/yisi_metric.py)
# and original implementation by David Anugraha

class YiSiModel(nn.Module):
    """
    Underlying model for YiSi-2: computes contextual cosine similarity and IDF-weighted pooling.
    """
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', idf_weights: Dict[int, float] = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.idf_weights = idf_weights or {}

    def get_token_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def compute_cosine_similarity_matrix(self, emb1, emb2):
        # emb1, emb2: [batch, seq_len, dim]
        norm1 = emb1.norm(dim=2, keepdim=True)
        norm2 = emb2.norm(dim=2, keepdim=True)
        return torch.bmm(emb1, emb2.transpose(1,2)) / (norm1 * norm2.transpose(1,2))

    def compute_weighted_pool(self, similarities, input_ids):
        # similarities: [batch, seq_len]
        # Build IDF weight tensor
        device = input_ids.device
        idf_matrix = torch.tensor(
            [[self.idf_weights.get(tok.item(), 1.0) for tok in seq] for seq in input_ids],
            device=device,
            dtype=similarities.dtype  # Explicitly match the data type of similarities
        )
        weighted_sum = torch.bmm(similarities.unsqueeze(1), idf_matrix.unsqueeze(2)).squeeze(-1)
        total_weight = idf_matrix.sum(dim=1, keepdim=True)
        return weighted_sum / total_weight

    def forward(self,
                pred_input_ids, pred_attention_mask,
                ref_input_ids, ref_attention_mask):
        # Get embeddings
        pred_emb = self.get_token_embeddings(pred_input_ids, pred_attention_mask)
        ref_emb = self.get_token_embeddings(ref_input_ids, ref_attention_mask)
        # Cosine similarity
        cos_sim = self.compute_cosine_similarity_matrix(pred_emb, ref_emb)
        # Max similarities
        max_sim_pred, _ = cos_sim.max(dim=2)
        max_sim_ref, _ = cos_sim.max(dim=1)
        # IDF-weighted pooling
        weighted_pred = self.compute_weighted_pool(max_sim_pred, pred_input_ids)
        weighted_ref = self.compute_weighted_pool(max_sim_ref, ref_input_ids)
        return weighted_pred, weighted_ref

class YiSi(ReferenceBasedMetric):
    """---
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
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 686.8544921875  # in MB
    cpu_mem: ClassVar[float] = 1420.12890625  # in MB
    description: ClassVar[str] = """YiSi-2 is a semantic machine translation evaluation and quality estimation metric designed for cross-lingual assessment. It measures the semantic similarity between the source sentence and the machine-translated output using bilingual word embeddings, optionally incorporating shallow semantic structures through semantic role labeling."""

    def __init__(
        self,
        name: str = 'YiSi',
        description: str = 'YiSi-2 is a semantic machine translation evaluation and quality estimation metric designed for cross-lingual assessment. It measures the semantic similarity between the source sentence and the machine-translated output using bilingual word embeddings, optionally incorporating shallow semantic structures through semantic role labeling.',
        model_name: str = 'bert-base-multilingual-cased',
        alpha: float = 0.8,
        batch_size: int = 64,
        max_input_length: int = 512,
        device: str = 'cuda',
        persistent: bool = True,
        **kwargs
    ):
        super().__init__(name, description, model_name=model_name, alpha=alpha, batch_size=batch_size, max_input_length=max_input_length, device=device, persistent=persistent, **kwargs)
        self.model_name = model_name
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.persistent = persistent
        self.tokenizer = None
        self.model = None

        self.exclude_from_cache_key('model_name', 'batch_size', 'device', 'persistent')

    def _initialize_metric(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = YiSiModel(model_name=self.model_name).to(self.device)
        self.model.eval()

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None

    def _compute_idf(self, docs: List[str]) -> Dict[int, float]:
        vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1,1),
            tokenizer=self.tokenizer.tokenize, token_pattern=None
        )
        tfidf = vectorizer.fit(docs)
        idf_dict: Dict[int, float] = {}
        for tok, idx in vectorizer.vocabulary_.items():
            token_id = self.tokenizer.convert_tokens_to_ids(tok)
            idf_dict[token_id] = tfidf.idf_[idx]
        return idf_dict

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts, padding=True, truncation=True,
            return_tensors='pt', max_length=self.max_input_length
        ).to(self.device)

    def _calculate_impl(self, input: str, output: str, references: Union[List[str], str], **kwargs) -> float:
        # Delegate to batched
        refs = references if isinstance(references, list) else [references]
        score = self._calculate_batched_impl([input], [output], [refs], **kwargs)[0]
        if not self.persistent:
            self._unload_model()
        return score

    def _calculate_batched_impl(self,
                          inputs: List[str],
                          outputs: List[str],
                          references: List[Union[List[str], str]],
                          **kwargs
    ) -> List[float]:
        # Flatten references to single string per example
        flat_refs: List[str] = []
        for ref in references:
            if isinstance(ref, (list, tuple)):
                flat_refs.append(ref[0])
            else:
                flat_refs.append(ref)
        preds = outputs
        # Initialize model/tokenizer if needed
        if self.model is None:
            self._initialize_metric()
        # Compute IDF over all texts
        idf_weights = self._compute_idf(preds + flat_refs)
        self.model.idf_weights = idf_weights
        results: List[float] = []
        # Batch inference
        for start in range(0, len(preds), self.batch_size):
            end = min(start + self.batch_size, len(preds))
            batch_preds = preds[start:end]
            batch_refs = flat_refs[start:end]
            pred_inputs = self._tokenize(batch_preds)
            ref_inputs = self._tokenize(batch_refs)
            with torch.no_grad():
                weighted_pred, weighted_ref = self.model(
                    pred_input_ids=pred_inputs['input_ids'],
                    pred_attention_mask=pred_inputs['attention_mask'],
                    ref_input_ids=ref_inputs['input_ids'],
                    ref_attention_mask=ref_inputs['attention_mask']
                )
            for p, r in zip(weighted_pred, weighted_ref):
                prec = p.item()
                rec = r.item()
                denom = self.alpha * prec + (1 - self.alpha) * rec
                # avoid division by zero
                score = (prec * rec) / denom if denom > 0 else 0.0
                results.append(score)
        if not self.persistent:
            self._unload_model()
        return results