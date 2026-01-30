# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizerFast, BartForConditionalGeneration
from typing import List, ClassVar
import numpy as np
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class BARTScorer:
    def __init__(
        self,
        device: str = None,
        max_length: int = None,
        checkpoint: str = "facebook/bart-large-cnn"
    ):
        self.checkpoint = checkpoint
        # Use the fast tokenizer to get correct model_max_length
        self.tokenizer = BartTokenizerFast.from_pretrained(checkpoint, use_fast=True)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        
        # Pick device
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        
        # Load on CPU first (HF default), then move safely to desired device if possible
        try:
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                self.model = self.model.to(self.device)
        except Exception as e:
            print(f"    🔧 BARTScore: keeping model on CPU due to device move error: {e}")

        # Cap max_length to the model's own limit
        limit = self.tokenizer.model_max_length
        if limit is None or limit > 1024:
            # most BART checkpoints report 1024
            limit = 1024
        self.max_length = limit if max_length is None else min(max_length, limit)

        # Loss & log-softmax
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.tokenizer.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=-1)

    def _rebuild_model_on(self, device: torch.device):
        """Recreate the model cleanly on the requested device, avoiding meta tensors."""
        try:
            # Load fresh on CPU first to avoid meta
            fresh = BartForConditionalGeneration.from_pretrained(self.checkpoint)
            fresh.eval()
            # Move to target device with meta-safe fallback
            if isinstance(device, str):
                device = torch.device(device)
            if str(device).startswith('cuda') and torch.cuda.is_available():
                fresh = fresh.to(device)
            self.model = fresh
            self.device = device
        except Exception as e:
            print(f"    ❌ Failed to rebuild BART model on {device}: {e}")
            raise

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def _truncate_text(self, text: str) -> str:
        """Return `text` shortened so that it tokenizes to <= max_length."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_length:
            return text
        ids = ids[: self.max_length]
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # -------------------------------------------------
    # main scoring routine
    # -------------------------------------------------
    def score(self, srcs: List[str], tgts: List[str], batch_size: int = 4):
        """Score (src, tgt) pairs, guarding against overly‑long inputs."""
        assert len(srcs) == len(tgts), "srcs and tgts must be same length"

        # pre‑truncate texts to avoid the tokenizer overflow bug
        srcs_trunc = [self._truncate_text(s) for s in srcs]
        tgts_trunc = [self._truncate_text(t) for t in tgts]

        scores: List[float] = []
        for i in range(0, len(srcs_trunc), batch_size):
            sb = srcs_trunc[i : i + batch_size]
            tb = tgts_trunc[i : i + batch_size]
            try:
                with torch.no_grad():
                    enc_src = self.tokenizer(
                        sb,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    enc_tgt = self.tokenizer(
                        tb,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )

                    # Move tensors to device; on failure, we will run this batch on CPU
                    batch_on_cpu = False
                    try:
                        src_ids = enc_src.input_ids.to(self.device, non_blocking=True)
                        src_mask = enc_src.attention_mask.to(self.device, non_blocking=True)
                        tgt_ids = enc_tgt.input_ids.to(self.device, non_blocking=True)
                        tgt_mask = enc_tgt.attention_mask.to(self.device, non_blocking=True)
                    except Exception:
                        # Fall back to CPU tensors
                        src_ids = enc_src.input_ids.cpu()
                        src_mask = enc_src.attention_mask.cpu()
                        tgt_ids = enc_tgt.input_ids.cpu()
                        tgt_mask = enc_tgt.attention_mask.cpu()
                        batch_on_cpu = True
                    
                    tgt_lens = tgt_mask.sum(dim=1)

                    # Ensure model is on the correct device before forward pass
                    try:
                        # If batch tensors are on CPU but model is on CUDA, run this batch on CPU
                        if batch_on_cpu:
                            model_device = next(self.model.parameters()).device
                            if str(model_device).startswith('cuda'):
                                cpu_model = self.model.to('cpu')
                                out = cpu_model(
                                    input_ids=src_ids,
                                    attention_mask=src_mask,
                                    decoder_attention_mask=tgt_mask,
                                    labels=tgt_ids,
                                )
                                # Optionally move back for next batch
                                if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                                    try:
                                        self.model = self.model.to(self.device)
                                    except Exception:
                                        pass
                            else:
                                out = self.model(
                                    input_ids=src_ids,
                                    attention_mask=src_mask,
                                    decoder_attention_mask=tgt_mask,
                                    labels=tgt_ids,
                                )
                        else:
                            out = self.model(
                                input_ids=src_ids,
                                attention_mask=src_mask,
                                decoder_attention_mask=tgt_mask,
                                labels=tgt_ids,
                            )
                    except Exception as e:
                        msg = str(e).lower()
                        # Rebuild fresh on CPU and run this batch on CPU for all device/meta/assert issues
                        if (
                            'expected device meta' in msg
                            or 'meta tensor' in msg
                            or 'device-side assert' in msg
                            or 'is not on the expected device' in msg
                            or 'expected all tensors to be on the same device' in msg
                        ):
                            print("    🔧 BARTScore device/meta issue; rebuilding on CPU and retrying batch…")
                            self._rebuild_model_on(torch.device('cpu'))
                            out = self.model(
                                input_ids=src_ids.cpu(),
                                attention_mask=src_mask.cpu(),
                                decoder_attention_mask=tgt_mask.cpu(),
                                labels=tgt_ids.cpu(),
                            )
                            # Optionally rebuild back on original device for subsequent batches
                            if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                                try:
                                    self._rebuild_model_on(self.device)
                                except Exception:
                                    pass
                        else:
                            raise

                    # Compute per-example scores using manual gather to avoid NLLLoss/meta issues
                    try:
                        logits = out.logits
                        # Guard: if logits are on meta device (fake tensors), force CPU path
                        if str(logits.device).startswith('meta') or getattr(logits, 'is_meta', False):
                            raise RuntimeError('logits_on_meta_device')
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = tgt_ids[:, 1:].contiguous()
                        pad_id = self.tokenizer.pad_token_id
                        # Mask out padding positions
                        valid_mask = (shift_labels != pad_id)
                        # Replace invalid labels with zero to allow safe gather
                        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
                        log_probs = self.lsm(shift_logits)
                        gathered = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
                        # Zero-out invalid positions
                        gathered = gathered * valid_mask.to(gathered.dtype)
                        token_counts = valid_mask.sum(dim=1).clamp_min(1)
                        per_ex_loss = -gathered.sum(dim=1) / token_counts
                        scores.extend([float(x.item()) * -1.0 for x in per_ex_loss])
                    except Exception:
                        # Full CPU fallback: re-tokenize and forward on CPU, then compute gather on CPU
                        try:
                            self._rebuild_model_on(torch.device('cpu'))
                            enc_src_cpu = self.tokenizer(
                                sb,
                                max_length=self.max_length,
                                truncation=True,
                                padding=True,
                                return_tensors="pt",
                            )
                            enc_tgt_cpu = self.tokenizer(
                                tb,
                                max_length=self.max_length,
                                truncation=True,
                                padding=True,
                                return_tensors="pt",
                            )
                            src_ids_cpu = enc_src_cpu.input_ids
                            src_mask_cpu = enc_src_cpu.attention_mask
                            tgt_ids_cpu = enc_tgt_cpu.input_ids
                            tgt_mask_cpu = enc_tgt_cpu.attention_mask
                            with torch.no_grad():
                                out_cpu = self.model(
                                    input_ids=src_ids_cpu,
                                    attention_mask=src_mask_cpu,
                                    decoder_attention_mask=tgt_mask_cpu,
                                    labels=tgt_ids_cpu,
                                )
                            logits_cpu = out_cpu.logits[:, :-1, :].contiguous()
                            labels_cpu = tgt_ids_cpu[:, 1:].contiguous()
                            pad_id = self.tokenizer.pad_token_id
                            valid_mask = (labels_cpu != pad_id)
                            safe_labels = labels_cpu.masked_fill(~valid_mask, 0)
                            log_probs = nn.LogSoftmax(dim=-1)(logits_cpu)
                            gathered = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
                            gathered = gathered * valid_mask.to(gathered.dtype)
                            token_counts = valid_mask.sum(dim=1).clamp_min(1)
                            per_ex_loss = -gathered.sum(dim=1) / token_counts
                            scores.extend([float(x.item()) * -1.0 for x in per_ex_loss])
                        finally:
                            # Try to rebuild back on original device for subsequent batches (best-effort)
                            if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                                try:
                                    self._rebuild_model_on(self.device)
                                except Exception:
                                    pass

            except (RuntimeError, OverflowError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Batch starting at {i} failed: {e}") from e

        return scores

    def multi_ref_score(
        self,
        srcs: List[str],
        tgts: List[List[str]],
        agg: str = "mean",
        batch_size: int = 4,
    ):
        # Check uniform references per sample
        ref_counts = [len(r) for r in tgts]
        if len(set(ref_counts)) > 1:
            raise ValueError("All examples must have the same number of references.")
        num_refs = ref_counts[0]

        # Score each reference column
        matrix = []
        for idx in range(num_refs):
            column = [refs[idx] for refs in tgts]
            matrix.append(self.score(srcs, column, batch_size))
        arr = np.array(matrix)
        if agg == "mean":
            return list(arr.mean(axis=0))
        elif agg == "max":
            return list(arr.max(axis=0))
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    def test(self, batch_size: int = 3):
        srcs = [
            "This is a very good idea. Although simple, but very insightful.",
            "Can I take a look?",
            "Do not trust him, he is a liar.",
        ]
        tgts = [
            "That's stupid.",
            "What's the problem?",
            "He is trustworthy.",
        ]
        print(self.score(srcs, tgts, batch_size))


class BARTScore(ReferenceBasedMetric):
    """---
# Metric Card for BARTScore

BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task. It leverages the pre-trained BART model to compute the conditional likelihood of one text given another, enabling flexible evaluation of different aspects such as informativeness, fluency, factuality, and coherence. BARTScore outperforms existing metrics across multiple tasks and evaluation settings.

## Metric Details

### Metric Description

BARTScore conceptualizes evaluation as a text generation problem, assessing how likely a hypothesis (generated text) is given a reference text, source text, or both. This probability is computed using the log-likelihood of the hypothesis under a pre-trained BART model. Different evaluation perspectives can be achieved by modifying the generation direction:

- **Faithfulness ($s \to h$)**: Measures how well the generated text aligns with the source text.
- **Precision ($r \to h$)**: Evaluates the likelihood of generating the hypothesis given the reference text.
- **Recall ($h \to r$)**: Assesses how easily the reference could be generated from the hypothesis.
- **F-score ($r \leftrightarrow h$)**: Computes an average of Precision and Recall.

Fine-tuning on downstream tasks (e.g., summarization, paraphrasing) and prompt engineering further enhance BARTScore's adaptability to different domains.

- **Metric Type:** Semantic Similarity  
- **Range:** $(-\infty, 0]$ (log-probabilities, higher is better)  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** Yes  

### Formal Definition

BARTScore is computed as:

$$
BARTScore = \sum _{t=1}^{m} \omega _{t} \log p ( y _{t} \mid y _{\text{<}t}, x, \theta )
$$

where:

- $p(y _{t} \mid y _{<t}, x, \theta)$ is the probability of the $t$-th token in the hypothesis $y$ given the preceding tokens and the source/reference text $x$ under the BART model parameters $\theta$.
- $\omega _{t}$ is an optional weighting factor (default: uniform).

The choice of $x$ and $y$ varies depending on the evaluation perspective (e.g., source-to-hypothesis for faithfulness, reference-to-hypothesis for precision).

### Inputs and Outputs

- **Inputs:**  
  - Source text (optional, for faithfulness evaluation)
  - Generated text (hypothesis)
  - Reference text(s) (for precision, recall, and F-score)

- **Outputs:**  
  - Scalar log-likelihood score (higher indicates better alignment)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:**  
  - Machine Translation  
  - Summarization  
  - Paraphrasing  
  - Data-to-Text Generation  
  - Dialogue Generation  

### Applicability and Limitations

- **Best Suited For:**  
  - Tasks where reference-based evaluation is appropriate (e.g., machine translation, summarization).
  - Evaluating generated text from multiple perspectives (e.g., factuality, coherence, fluency).
  - Cases where fine-tuning and prompt-based customization are beneficial.

- **Not Recommended For:**  
  - Fully reference-free evaluation tasks.
  - Open-ended generation tasks where diversity matters more than similarity to references.
  - Evaluating highly extractive summaries, where performance may degrade.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [BARTScore GitHub Repository](https://github.com/neulab/BARTScore)  
  - Available in Hugging Face `evaluate` module  

### Computational Complexity

- **Efficiency:**  
  - Requires forward passes through BART, making it more computationally expensive than n-gram-based metrics.  
  - Can be optimized using batch processing.  

- **Scalability:**  
  - Suitable for large-scale evaluations but requires GPU acceleration for efficiency.  
  - Performance depends on the pre-trained model size and dataset length.

## Known Limitations

- **Biases:**  
  - BARTScore tends to favor abstractive over extractive summaries.  
  - May be sensitive to the domain of the pre-trained BART model used.  

- **Task Misalignment Risks:**  
  - May not fully capture factual correctness despite faithfulness scoring.  
  - Sensitive to tokenization and domain shift effects.  

- **Failure Cases:**  
  - Performance degrades when evaluating extractive summarization models.  
  - Prompt engineering impacts results significantly, requiring careful selection.  

## Related Metrics

- **ROUGE:** Measures lexical overlap, whereas BARTScore captures semantic similarity.  
- **BERTScore:** Also embeds text using pre-trained models but computes cosine similarity instead of generation probabilities.  
- **BLEU:** Focuses on n-gram precision, lacking semantic alignment capabilities.  

## Further Reading

- **Papers:**  
  - [BARTScore: Evaluating Generated Text as Text Generation (Yuan et al., 2021)](https://arxiv.org/abs/2106.11520)  

- **Blogs/Tutorials:**  
  - [BARTScore GitHub Documentation](https://github.com/neulab/BARTScore)  

## Citation

```
@inproceedings{10.5555/3540261.3542349,
   author = {Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
   title = {BARTSCORE: evaluating generated text as text generation},
   year = {2021},
   isbn = {9781713845393},
   publisher = {Curran Associates Inc.},
   address = {Red Hook, NY, USA},
   booktitle = {Proceedings of the 35th International Conference on Neural Information Processing Systems},
   articleno = {2088},
   numpages = {15},
   series = {NIPS '21}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 1558.19189453125  # in MB
    cpu_mem: ClassVar[float] = 1370.58984375  # in MB
    description: ClassVar[str] = "BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task. It leverages the pre-trained BART model to compute the conditional likelihood of one text given another, enabling flexible evaluation of different aspects such as informativeness, fluency, factuality, and coherence. BARTScore outperforms existing metrics across multiple tasks and evaluation settings."

    def __init__(
        self, batch_size: int = 4, model: str = "facebook/bart-large-cnn", persistent: bool = True, **kwargs
    ):
        # Get device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Pass all parameters explicitly to parent constructor for caching
        super().__init__(
            name=f"BARTScore_{model.split('/')[-1]}",
            description="BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task. It leverages the pre-trained BART model to compute the conditional likelihood of one text given another, enabling flexible evaluation of different aspects such as informativeness, fluency, factuality, and coherence. BARTScore outperforms existing metrics across multiple tasks and evaluation settings.",
            batch_size=batch_size,
            model=model,
            device=str(self.device),  # Include device as string
            persistent=persistent,
            **kwargs
        )
        
        self.model = model
        self.batch_size = batch_size
        self.persistent = persistent
        self.bart_scorer = None
        
        # Exclude from cache key as these don't affect results, just computation method
        self.exclude_from_cache_key('device', 'batch_size', 'persistent')

    def _load_model(self):
        """Load the BART model if not already loaded."""
        if self.bart_scorer is None:
            self.bart_scorer = BARTScorer(
                device=str(self.device), max_length=None, checkpoint=self.model
            )

    def _unload_model(self):
        """Unload model to free resources."""
        if self.bart_scorer is not None:
            del self.bart_scorer
            torch.cuda.empty_cache()
            self.bart_scorer = None

    def _calculate_impl(self, input: str, output: str, references=None, **kwargs):
        if self.bart_scorer is None:
            self._load_model()
            
        refs = references or []
        if len(refs) > 1:
            scores = self.bart_scorer.multi_ref_score(
                [input], [refs], agg="max", batch_size=self.batch_size
            )
        else:
            scores = self.bart_scorer.score(
                [input], [refs[0] if refs else ""], batch_size=self.batch_size
            )
            
        result = scores[0]
        
        if not self.persistent:
            self._unload_model()
            
        return result

    def _calculate_batched_impl(
        self, inputs: List[str], outputs: List[str], references=None, **kwargs
    ):
        if self.bart_scorer is None:
            self._load_model()
            
        refs = references or [[] for _ in inputs]
        groups = {}
        for i, r in enumerate(refs):
            groups.setdefault(len(r), []).append(i)

        all_scores = [0] * len(outputs)
        for ref_count, idxs in groups.items():
            cur_inputs = [inputs[i] for i in idxs]
            cur_refs   = [refs[i] for i in idxs]
            if ref_count > 1:
                sc = self.bart_scorer.multi_ref_score(
                    cur_inputs, cur_refs, agg="max", batch_size=self.batch_size
                )
            else:
                single_refs = [r[0] if r else "" for r in cur_refs]
                sc = self.bart_scorer.score(
                    cur_inputs, single_refs, batch_size=self.batch_size
                )
            for idx, score in zip(idxs, sc):
                all_scores[idx] = score
                
        if not self.persistent:
            self._unload_model()

        return all_scores


if __name__ == "__main__":
    metric = BARTScore()

    # single example
    inp = (
        "Peter and Elizabeth took a taxi to attend the night party in the city. "
        "While in the party, Elizabeth collapsed and was rushed to the hospital."
    )
    out = "Peter and Elizabeth attend party city. Elizabeth rushed hospital."
    refs = ["Elizabeth was hospitalized after attending a party with Peter."]
    print("BARTScore:", metric.calculate(inp, out, references=refs))

    # batched examples
    inputs = [inp, "The cat sat on the mat."]
    outputs = [out, "The cat is on the mat."]
    references = [
        ["Elizabeth was hospitalized after attending a party with Peter."],
        ["The cat sat on the mat.", "The cat is on the mat.", "The cat is on the rug."],
    ]
    print("BARTScore batch scores:", metric.calculate_batched(inputs, outputs, references=references))
