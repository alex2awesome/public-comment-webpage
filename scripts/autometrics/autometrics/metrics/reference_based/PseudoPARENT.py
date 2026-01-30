import collections
import math
import multiprocessing as mp
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from typing import ClassVar

def _parent_helper(args):
    metric, inp, out, refs = args
    return metric.parent(inp, out, refs)

class PseudoPARENT(ReferenceBasedMetric):
    """---
# Metric Card for PseudoPARENT

**PseudoPARENT** is a *custom adaptation* of the PARENT metric for evaluating text generation from structured inputs. Unlike the original PARENT metric, which operates on explicit tables of (attribute, value) or (head, relation, tail) triples, **PseudoPARENT treats the input text as a flat sequence of tokens, using these tokens as "pseudo-table values"**. This allows PseudoPARENT to simulate input-aware evaluation without requiring a structured table format. It effectively acts as a hybrid between BLEU-style surface matching and PARENT-style factual grounding.

## Metric Details

### Metric Description

PseudoPARENT is a reference-based and input-aware metric designed to evaluate whether generated outputs faithfully reflect the content of an input text. It adapts the structure of the original PARENT metric but with a crucial change: **the structured table is replaced with a bag-of-tokens from the input string**, which are treated as table values. The entailment functions (used to determine how well predicted content aligns with the source) are redefined as simple token overlaps with this pseudo-table.

This design allows PseudoPARENT to serve as an *input-aware variant of BLEU*, incorporating both:
- **Precision**: How well the prediction matches the reference, with token-based entailment fallback to the input.
- **Recall**: How well the prediction mentions content from the reference and input combined.

Multiple references are supported, and the final score is based on the maximum F1 across references.

- **Metric Type:** Faithfulness
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let:
- $P$ be the prediction (generated output),
- $R_i$ be each reference (tokenized),
- $I$ be the tokenized input text (used as a pseudo-table),
- $O(ngram, I)$ be the overlap probability of an $n$-gram with the input,
- $M(tok, P)$ be the mention probability of an input token in the prediction,
- $N$ be the maximum $n$-gram order (default: 4).

Then for each $(P, \{R_i\}, I)$ triple:

1. Compute precision:
$$
\text{Prec}_n = \frac{\sum_{ng \in P_n} c(ng) \cdot \left[\min\left(1, \frac{c_{R_i}(ng)}{c(ng)}\right) + \left(1 - \min\left(1, \frac{c_{R_i}(ng)}{c(ng)}\right)\right) \cdot O(ng, I)\right]}{\sum_{ng \in P_n} c(ng)}
$$

2. Compute recall (reference and input):
$$
\text{RefRec}_n = \frac{\sum_{ng \in R_i^n} c(ng) \cdot O(ng, I) \cdot \min\left(1, \frac{c_P(ng)}{c(ng)}\right)}{\sum_{ng \in R_i^n} c(ng) \cdot O(ng, I)}
$$

$$
\text{InputRec} = \frac{|\{tok \in I : tok \in P\}|}{|I|}
$$

3. Combine:
$$
\text{CPrec} = \left( \prod_{n=1}^N \text{Prec}_n \right)^{\frac{1}{N}}, \quad
\text{RefRec} = \left( \prod_{n=1}^N \text{RefRec}_n \right)^{\frac{1}{N}}, \quad
\text{CRec} = \text{RefRec}^{(1 - \lambda)} \cdot \text{InputRec}^\lambda
$$

4. Final F-score:
$$
F = \frac{2 \cdot \text{CPrec} \cdot \text{CRec}}{\text{CPrec} + \text{CRec}}
$$

The final score is the maximum $F$ across all references $R_i$.

### Inputs and Outputs

- **Inputs:**  
  - Generated text (candidate)  
  - One or more reference texts  
  - Input text (used as source "table")  

- **Outputs:**  
  - Scalar score representing F1 (can also return precision and recall)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Data-to-Text Generation, Summarization, Paraphrasing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluation setups where input text is available and should influence output generation, especially when explicit structured tables are unavailable.

- **Not Recommended For:**  
  Tasks lacking clear input-output alignment or where structured data is essential to proper factual grounding (e.g., original PARENT use cases). Not suitable for purely reference-free evaluation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Based on original [google-research/language](https://github.com/google-research/language/tree/master/language/table_text_eval)

### Computational Complexity

- **Efficiency:**  
  Moderate. Involves $n$-gram counting and token set operations. Significantly faster than original PARENT due to simplified overlap-based entailment and multiprocessing support.

- **Scalability:**  
  Scales well to medium-size datasets. Multiprocessing improves batch computation efficiency.

## Known Limitations

- **Biases:**  
  Penalizes synonyms and paraphrases that are not token-identical to input. Precision fallback to input overlap may reward hallucinated content that happens to share tokens with the input.

- **Task Misalignment Risks:**  
  Designed for input-sensitive evaluation; not appropriate for tasks where the input is irrelevant or open-ended.

- **Failure Cases:**  
  Performs poorly when input is noisy, overly long, or semantically distant from the output. Token-level matching can be brittle under complex morphology or low-resource tokenization.

## Related Metrics

- **PARENT:** Original version with structured table inputs and entailment probabilities.
- **BLEU:** Pure $n$-gram overlap; lacks input-awareness.
- **ROUGE:** Recall-based $n$-gram overlap.
- **BERTScore:** Embedding-based similarity; can be extended to input-aware settings.

## Further Reading

- **Papers:**  
  - [Original PARENT Paper (ACL 2019)](https://aclanthology.org/P19-1483/)

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{dhingra-etal-2019-handling,
    title = "Handling Divergent Reference Texts when Evaluating Table-to-Text Generation",
    author = "Dhingra, Bhuwan  and
      Faruqui, Manaal  and
      Parikh, Ankur  and
      Chang, Ming-Wei  and
      Das, Dipanjan  and
      Cohen, William",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\`a}rquez, Llu{\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1483/",
    doi = "10.18653/v1/P19-1483",
    pages = "4884--4895"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 725.58984375  # in MB
    description: ClassVar[str] = "**PseudoPARENT** is a *custom adaptation* of the PARENT metric for evaluating text generation from structured inputs. Unlike the original PARENT metric, which operates on explicit tables of (attribute, value) or (head, relation, tail) triples, **PseudoPARENT treats the input text as a flat sequence of tokens, using these tokens as \"pseudo-table values\"**. This allows PseudoPARENT to simulate input-aware evaluation without requiring a structured table format. It effectively acts as a hybrid between BLEU-style surface matching and PARENT-style factual grounding."

    def __init__(
        self,
        name="PseudoPARENT",
        description="**PseudoPARENT** is a *custom adaptation* of the PARENT metric for evaluating text generation from structured inputs. Unlike the original PARENT metric, which operates on explicit tables of (attribute, value) or (head, relation, tail) triples, **PseudoPARENT treats the input text as a flat sequence of tokens, using these tokens as \"pseudo-table values\"**. This allows PseudoPARENT to simulate input-aware evaluation without requiring a structured table format. It effectively acts as a hybrid between BLEU-style surface matching and PARENT-style factual grounding.",
        lambda_weight=0.5,
        smoothing=1e-5,
        max_order=4,
        n_jobs=1,
        **kwargs
    ):
        super().__init__(name, description, lambda_weight=lambda_weight, smoothing=smoothing, max_order=max_order, **kwargs)
        self.lambda_weight = lambda_weight
        self.smoothing = smoothing
        self.max_order = max_order
        self.n_jobs = mp.cpu_count() if n_jobs and n_jobs < 0 else n_jobs
        
        self.exclude_from_cache_key("n_jobs")
        
    def tokenize(self, text: str):
        if not text:
            return []
        return text.split()
    
    def safe_log(self, x, eps=1e-12):
        return math.log(x if x > 0 else eps)

    def _ngram_counts(self, sequence, n):
        if len(sequence) < n:
            return collections.Counter()
        return collections.Counter(
            tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)
        )

    def overlap_probability(self, ngram, table_values):
        # fraction of tokens in ngram that appear in table_values
        overlap = sum(1 for token in ngram if token in table_values)
        return float(overlap + self.smoothing) / float(len(ngram) + self.smoothing)

    def _precision(self, pred_tokens, ref_tokens_list, table_values):
        ng_prec = []
        for n in range(1, self.max_order + 1):
            pred_counts = self._ngram_counts(pred_tokens, n)
            if not pred_counts:
                ng_prec.append(0.0)
                continue
            num = 0.0
            den = 0
            # precompute reference ngram sets
            ref_sets = [set(self._ngram_counts(ref, n).keys()) for ref in ref_tokens_list]
            for ngram, count in pred_counts.items():
                den += count
                in_ref = any(ngram in s for s in ref_sets)
                w = self.overlap_probability(ngram, table_values)
                num += count * (1.0 if in_ref else w)
            ng_prec.append(num / den)
        # geometric mean with smoothing
        rates = [p if p > 0 else self.smoothing for p in ng_prec]
        return math.exp(sum(self.safe_log(p) for p in rates) / self.max_order)

    def _recall(self, pred_tokens, ref_tokens_list, table_values):
        # recall against references
        ng_rec = []
        for n in range(1, self.max_order + 1):
            # build max-count reference counter
            ref_counts = collections.Counter()
            for ref in ref_tokens_list:
                for ng, c in self._ngram_counts(ref, n).items():
                    ref_counts[ng] = max(ref_counts.get(ng, 0), c)
            if not ref_counts:
                ng_rec.append(1.0)
                continue
            num = 0.0
            den = 0
            pred_counts = self._ngram_counts(pred_tokens, n)
            for ng, c in ref_counts.items():
                den += c
                if pred_counts.get(ng, 0) >= c:
                    num += c * self.overlap_probability(ng, table_values)
            ng_rec.append(num / den)
        # geometric mean
        rec_ref = math.exp(sum(self.safe_log(r if r > 0 else self.smoothing) for r in ng_rec) / self.max_order)
        # recall against input tokens (table_values)
        if table_values:
            rec_inp = sum(1 for tok in table_values if tok in pred_tokens) / len(table_values)
        else:
            rec_inp = 0.0
        # combine
        lw = self.lambda_weight
        return (rec_ref ** (1 - lw)) * (rec_inp ** lw)

    def parent(self, input, output, references):
        inp_toks = self.tokenize(input)
        table_values = set(inp_toks)
        pred_toks = self.tokenize(output)
        refs = references or []
        ref_tok_lists = [self.tokenize(r) for r in refs] if refs else [[]]

        p = self._precision(pred_toks, ref_tok_lists, table_values)
        r = self._recall(pred_toks, ref_tok_lists, table_values)
        return 2 * p * r / (p + r + 1e-8)

    def _calculate_impl(self, input, output, references=None, **kwargs):
        return self.parent(input, output, references or [])

    def calculate(self, input, output, references=None, **kwargs):
        return self.parent(input, output, references or [])

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        refs = references or [[] for _ in inputs]

        # Sequential fallback when multiprocessing is disabled (n_jobs<=1)
        if not self.n_jobs or self.n_jobs <= 1:
            return [self.parent(i, o, r) for i, o, r in zip(inputs, outputs, refs)]

        tasks = [(self, i, o, r) for i, o, r in zip(inputs, outputs, refs)]
        with mp.Pool(processes=self.n_jobs) as pool:
            return pool.map(_parent_helper, tasks)

    def predict(self, dataset, update_dataset=True, **kwargs):
        return super().predict(dataset, update_dataset=update_dataset, **kwargs)
