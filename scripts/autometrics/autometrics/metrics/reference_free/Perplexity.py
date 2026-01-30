"""
This file is based on the reference implementation of Fixed-Length Perplexity from:
https://huggingface.co/docs/transformers/en/perplexity
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from tqdm import tqdm
from accelerate.test_utils.testing import get_backend
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
from typing import ClassVar


# -------------------------------
# Helper Functions
# -------------------------------

def maybe_tqdm(iterable, progress_bar=True, **tqdm_kwargs):
    """Wrap iterable with tqdm if progress_bar is True."""
    return tqdm(iterable, **tqdm_kwargs) if progress_bar else iterable


def tokenize_texts(texts, tokenizer):
    """
    Tokenize each document (string) using the provided tokenizer.
    Returns a list of 1D tensors (input_ids).
    """
    tokenized_texts = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)  # shape: (L,)
        tokenized_texts.append(input_ids)
    return tokenized_texts


def get_windows(input_ids, max_length, stride):
    """
    Given a 1D tensor of token ids, break it into overlapping windows.
    Each window is a dict containing:
      - "window": the slice of tokens,
      - "trg_len": number of new tokens (i.e. tokens not seen in the previous window),
      - "window_length": the length of the window.
    """
    windows = []
    L = input_ids.size(0)
    prev_end_loc = 0
    for begin in range(0, L, stride):
        end = min(begin + max_length, L)
        trg_len = end - prev_end_loc  # tokens not covered in the previous iteration
        window = input_ids[begin:end]
        windows.append({"window": window, "trg_len": trg_len, "window_length": end - begin})
        prev_end_loc = end
        if end == L:
            break
    return windows


def group_texts_by_windows(tokenized_texts, max_length, stride):
    """
    For each tokenized text, compute its sliding windows and group texts by the number of windows.
    Returns:
      - grouped: a dict mapping number-of-windows to a list of tuples (doc_id, windows)
      - texts_windows: a list of tuples (doc_id, windows) in original order.
    """
    texts_windows = []
    for idx, input_ids in enumerate(tokenized_texts):
        windows = get_windows(input_ids, max_length, stride)
        texts_windows.append((idx, windows))
    grouped = defaultdict(list)
    for doc_id, windows in texts_windows:
        grouped[len(windows)].append((doc_id, windows))
    return grouped, texts_windows


def compute_per_document_perplexities(grouped, model, tokenizer, device, batch_size, progress_bar):
    """
    Process groups of documents (each with the same number of sliding-window iterations)
    in batches, computing the cumulative negative log likelihood and token counts for each document.
    Returns a dictionary mapping document ID to its perplexity.
    """
    doc_nll = {}
    doc_tokens = {}
    # Initialize all document ids found in the groups.
    for group in grouped.values():
        for doc_id, _ in group:
            doc_nll[doc_id] = 0.0
            doc_tokens[doc_id] = 0

    for num_iters, group in maybe_tqdm(grouped.items(), progress_bar=progress_bar, desc="Groups"):
        for batch_start in range(0, len(group), batch_size):
            batch = group[batch_start: batch_start + batch_size]
            for iter_idx in range(num_iters):
                # Extract document ids and window information for the current iteration.
                batch_doc_ids = [item[0] for item in batch]
                windows_batch = [item[1][iter_idx] for item in batch]
                # Get raw token tensors for each window.
                window_tensors = [w["window"] for w in windows_batch]
                # Pad to the maximum length in this batch.
                padded_inputs = pad_sequence(window_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
                target_ids = padded_inputs.clone()
                # Mask tokens not part of the current sliding-window chunk.
                for i, w in enumerate(windows_batch):
                    valid_length = w["window_length"]
                    trg_len = w["trg_len"]
                    valid_start = valid_length - trg_len  # only the last trg_len tokens are new
                    target_ids[i, :valid_start] = -100
                    if valid_length < padded_inputs.size(1):
                        target_ids[i, valid_length:] = -100

                # Clamp any out-of-range token IDs to avoid device-side asserts in Embedding/CE
                vocab_size_inputs = getattr(model.config, 'vocab_size', None)
                if vocab_size_inputs is not None:
                    padded_inputs = torch.clamp(padded_inputs, min=0, max=vocab_size_inputs - 1)
                padded_inputs = padded_inputs.to(device)
                target_ids = target_ids.to(device)
                # Derive attention mask to avoid attending over pad tokens
                attention_mask = (padded_inputs != tokenizer.pad_token_id).long().to(device)
                # Robust forward: try with attention_mask; on CUDA/meta issues, retry without it on same device
                with torch.no_grad():
                    try:
                        outputs = model(padded_inputs, attention_mask=attention_mask, labels=None)
                    except RuntimeError as e:
                        emsg = str(e).lower()
                        if ('cuda' in emsg or 'device-side assert' in emsg or 'meta' in emsg or 'cublas' in emsg):
                            outputs = model(padded_inputs, attention_mask=None, labels=None)
                        else:
                            raise
                logits = outputs.logits
                # Shift logits and targets for computing cross-entropy loss.
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_targets = target_ids[:, 1:].contiguous()
                # Sanitize targets: any id >= vocab_size is invalid => mark as ignore_index
                vocab_size = shifted_logits.size(-1)
                flat_targets = shifted_targets.view(-1)
                invalid_mask = (flat_targets >= vocab_size) & (flat_targets != -100)
                if invalid_mask.any():
                    flat_targets = flat_targets.masked_fill(invalid_mask, -100)
                    shifted_targets = flat_targets.view_as(shifted_targets)
                # Use ignore_index to safely skip masked/invalid targets (-100)
                loss_tensor = F.cross_entropy(
                    shifted_logits.view(-1, vocab_size),
                    shifted_targets.view(-1),
                    reduction='none',
                    ignore_index=-100,
                )
                loss_tensor = loss_tensor.view(shifted_targets.size())
                # For each example in the batch, sum the loss over valid tokens.
                for i in range(loss_tensor.size(0)):
                    sample_losses = loss_tensor[i]
                    sample_targets = shifted_targets[i]
                    valid_mask = sample_targets != -100
                    token_loss_sum = sample_losses[valid_mask].sum().item()
                    num_valid_tokens = valid_mask.sum().item()
                    doc_id = batch_doc_ids[i]
                    doc_nll[doc_id] += token_loss_sum
                    doc_tokens[doc_id] += num_valid_tokens

    # Compute perplexity for each document.
    doc_perplexities = {}
    for doc_id in doc_nll:
        if doc_tokens[doc_id] > 0:
            avg_nll = doc_nll[doc_id] / doc_tokens[doc_id]
            perplexity = torch.exp(torch.tensor(avg_nll)).item()
            doc_perplexities[doc_id] = perplexity
        else:
            doc_perplexities[doc_id] = float('inf')
    return doc_perplexities


def calculate_perplexities(documents, model, tokenizer, device, batch_size=8, stride=512, progress_bar=True):
    """
    Given a list of document strings, compute and return a list of perplexities,
    one per document (in the same order as the input list).
    """
    tokenized_texts = tokenize_texts(documents, tokenizer)
    max_length = model.config.n_positions
    grouped, texts_windows = group_texts_by_windows(tokenized_texts, max_length, stride)
    doc_perplexities = compute_per_document_perplexities(grouped, model, tokenizer, device, batch_size, progress_bar)
    # Return perplexities in original document order.
    ordered = [doc_perplexities[i] for i in range(len(texts_windows))]
    return ordered


# -------------------------------
# Perplexity Class
# -------------------------------

class Perplexity(ReferenceFreeMetric):
    """---
# Metric Card for Perplexity

Perplexity (PPL) is a widely used metric for evaluating the fluency of language models. It measures how well a probabilistic model predicts a sequence of tokens, with lower values indicating better predictions. Specifically, it computes the exponentiated average negative log-likelihood of a sequence. Perplexity is only applicable to autoregressive language models (e.g., GPT-2) and **cannot** be used with masked language models like BERT.

## Metric Details

### Metric Description

Perplexity assesses the predictive capability of a language model by computing the exponentiated average negative log-likelihood of a given sequence. It quantifies how uncertain the model is when predicting the next token. A lower perplexity score indicates better model performance, as it suggests the model assigns higher probabilities to the correct tokens.

- **Metric Type:** Fluency
- **Range:** $(1, \infty)$
- **Higher is Better?:** No
- **Reference-Based?:** No
- **Input-Required?:** No (Perplexity can be computed on output tokens alone)

### Formal Definition

Given a sequence of tokens $X = (x_1, x_2, ..., x_T)$, the perplexity of $X$ under a language model with parameters $\theta$ is defined as:

$$
PPL(X) = \exp \left( -\frac{1}{T} \sum_{i=1}^{T} \log p_{\theta}(x_i \mid x_{\text{<}i}) \right)
$$

where:
- $p_{\theta}(x_i \mid x_{\text{<}i})$ is the probability assigned by the model to token $x_i$ given the preceding tokens.
- $T$ is the length of the sequence.

A lower perplexity value indicates that the model assigns higher probabilities to observed sequences, meaning it better predicts the given data.

### Sliding-Window Perplexity

For models with a fixed context size (e.g., GPT-2, LLaMA), perplexity cannot be computed over arbitrarily long sequences directly. Instead, a **sliding-window** approach is used, as described in the [Hugging Face blog on perplexity](https://huggingface.co/docs/transformers/en/perplexity):

- The input sequence is broken into overlapping **windows** of a fixed length.
- Each window is passed through the model, and **only the log-likelihood of the newly introduced tokens** (not the entire window) is used in the perplexity calculation.
- This approach better approximates full-sequence perplexity compared to naïve chunking (which can overestimate perplexity due to loss of context).

Using this method, perplexity is calculated as:

$$
PPL(X) = \exp \left( -\frac{1}{T} \sum_{i=1}^{T} \log p_{\theta}(x_i \mid x_{\max(1, i-k):i-1}) \right)
$$

where:
- $k$ is the model's fixed context size,
- The probability of each token $x_i$ is conditioned on a **sliding context of at most $k$ tokens**.

This method provides a **more realistic** evaluation of model fluency while efficiently handling long sequences.

### Inputs and Outputs

- **Inputs:**  
  - A sequence of text tokens (typically output from a model)
  - A trained language model (e.g., GPT-2)
  - Tokenizer for processing input text

- **Outputs:**  
  - A scalar value representing the perplexity score of the input text

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Language Modeling, Dialogue Generation, Storytelling, Code Completion

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating the fluency of language models, especially autoregressive models
  - Comparing the relative performance of different language models on the same dataset
  
- **Not Recommended For:**  
  - Evaluating masked language models (e.g., BERT) since perplexity is undefined for non-autoregressive architectures
  - Assessing high-level semantic coherence, factual consistency, or diversity in generated text

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/perplexity)

### Computational Complexity

- **Efficiency:**  
  - Perplexity calculation involves computing log-likelihoods for each token, making it computationally intensive for large datasets.
  
- **Scalability:**  
  - Efficient when used with GPU acceleration but may become expensive for long sequences due to the need for multiple forward passes.

## Known Limitations

- **Biases:**  
  - Sensitive to tokenization choices; different tokenization schemes can yield different perplexity values.
  - Models trained on specific domains may yield artificially low perplexity scores on similar datasets while failing on out-of-domain data.
  
- **Task Misalignment Risks:**  
  - Perplexity measures token-level fluency but does not assess semantic correctness or factuality.
  
- **Failure Cases:**  
  - Does not distinguish between grammatically correct but nonsensical text and genuinely coherent text.
  - Perplexity values are not always comparable across different models due to differences in vocabulary and tokenization.

## Related Metrics

- **Cross-Entropy Loss:** Closely related to perplexity, as perplexity is the exponentiated cross-entropy loss.
- **BERTScore:** Evaluates semantic similarity rather than fluency.
- **ROUGE/BLEU:** Measure lexical overlap rather than model uncertainty.

## Further Reading

- **Papers:**  
  - Jelinek et al. (1977) - [Perplexity: A Measure of the Difficulty of Speech Recognition Tasks](https://doi.org/10.1121/1.2016299)
  - Hugging Face Documentation - [Perplexity of Fixed-Length Models](https://huggingface.co/docs/transformers/en/perplexity)

- **Blogs/Tutorials:**  
  - [Understanding Evaluation Metrics for Language Models](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)
  - [Hugging Face's Guide to Perplexity](https://huggingface.co/docs/transformers/en/perplexity)

## Citation

```
@article{10.1121/1.2016299,
    author = {Jelinek, F. and Mercer, R. L. and Bahl, L. R. and Baker, J. K.},
    title = {Perplexity—a measure of the difficulty of speech recognition tasks},
    journal = {The Journal of the Acoustical Society of America},
    volume = {62},
    number = {S1},
    pages = {S63-S63},
    year = {2005},
    month = {08},
    abstract = {Using counterexamples, we show that vocabulary size and static and dynamic branching factors are all inadequate as measures of speech recognition complexity of finite state grammars. Information theoretic arguments show that perplexity (the logarithm of which is the familiar entropy) is a more appropriate measure of equivalent choice. It too has certain weaknesses which we discuss. We show that perplexity can also be applied to languages having no obvious statistical description, since an entropy‐maximizing probability assignment can be found for any finite‐state grammar. Table I shows perplexity values for some well‐known speech recognition tasks. Perplexity Vocabulary Dynamic Phone Word size branching factorIBM‐Lasers 2.14 21.11 1000 1000IBM‐Raleigh 1.69 7.74 250 7.32CMU‐AIX05 1.52 6.41 1011 35},
    issn = {0001-4966},
    doi = {10.1121/1.2016299},
    url = {https://doi.org/10.1121/1.2016299},
    eprint = {https://pubs.aip.org/asa/jasa/article-pdf/62/S1/S63/11558910/s63\_5\_online.pdf},
}
```
  
## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu
    """

    # Resource usage statistics (in megabytes) based on GPT-2-large
    gpu_mem: ClassVar[float] = 3069.4375  # in MB
    cpu_mem: ClassVar[float] = 1504.29296875  # in MB
    description: ClassVar[str] = "Perplexity (PPL) is a widely used metric for evaluating the fluency of language models. It measures how well a probabilistic model predicts a sequence of tokens, with lower values indicating better predictions. Specifically, it computes the exponentiated average negative log-likelihood of a sequence. Perplexity is only applicable to autoregressive language models (e.g., GPT-2) and **cannot** be used with masked language models like BERT."

    def __init__(self, model="gpt2-large", batch_size=8, stride=512, progress_bar=True, persistent=True, **kwargs):
        name = "Perplexity_" + model
        description = "Perplexity (PPL) is a widely used metric for evaluating the fluency of language models. It measures how well a probabilistic model predicts a sequence of tokens, with lower values indicating better predictions. Specifically, it computes the exponentiated average negative log-likelihood of a sequence. Perplexity is only applicable to autoregressive language models (e.g., GPT-2) and **cannot** be used with masked language models like BERT."
        super().__init__(name, description, model=model, batch_size=batch_size, stride=stride, 
                         progress_bar=progress_bar, persistent=persistent, **kwargs)
        
        self.model_name = model
        self.batch_size = batch_size
        self.stride = stride
        self.progress_bar = progress_bar
        self.persistent = persistent

        device, _, _ = get_backend()
        # Prefer CPU for robustness unless explicitly CUDA
        self.device = device if str(device).startswith('cuda') and torch.cuda.is_available() else torch.device('cpu')
        
        self.model = None
        self.tokenizer = None
        
        if self.persistent:
            self._load_model()

        self.exclude_from_cache_key('batch_size', 'progress_bar', 'persistent')

    def _load_model(self):
        """Load model and tokenizer if not already loaded."""
        if getattr(self, 'model', None) is None:
            # Load on CPU first, then move to target device if CUDA and safe
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.eval()
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                try:
                    self.model = self.model.to(self.device)
                except Exception:
                    # Keep on CPU if move fails
                    self.device = torch.device('cpu')
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _unload_model(self):
        """Unload model and tokenizer to free resources."""
        # Make idempotent and resilient to partial deletion
        if hasattr(self, 'model') or hasattr(self, 'tokenizer'):
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Calculate the perplexity for a single document.
        Assumes `input` is a string.
        """
        if self.model is None:
            self._load_model()
            
        perplexities = calculate_perplexities(
            [output],
            self.model,
            self.tokenizer,
            self.device,
            batch_size=self.batch_size,
            stride=self.stride,
            progress_bar=False
        )
        
        result = perplexities[0]
        
        if not self.persistent:
            self._unload_model()
            
        return result

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate perplexities for a batch of documents.
        Assumes `inputs` is a list of strings.
        """
        if self.model is None:
            self._load_model()

        try:
            results = calculate_perplexities(
                outputs,
                self.model,
                self.tokenizer,
                self.device,
                batch_size=self.batch_size,
                stride=self.stride,
                progress_bar=self.progress_bar
            )
        except RuntimeError as e:
            # Retry on CPU for device-side assert / CUDA issues
            msg = str(e).lower()
            if 'device-side assert' in msg or 'cuda' in msg or 'cublas' in msg:
                cpu_device = torch.device('cpu')
                # Fully unload and reload on CPU to avoid touching a corrupted CUDA context
                try:
                    self._unload_model()
                except Exception:
                    pass
                self.device = cpu_device
                self._load_model()
                results = calculate_perplexities(
                    outputs,
                    self.model,
                    self.tokenizer,
                    cpu_device,
                    batch_size=self.batch_size,
                    stride=self.stride,
                    progress_bar=self.progress_bar
                )
            else:
                raise
        
        if not self.persistent:
            self._unload_model()
            
        return results

# -------------------------------
# Main Execution for Benchmarking
# -------------------------------
if __name__ == "__main__":
    # Load the wikitext dataset and join the texts with "\n\n"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    joined_text = "\n\n".join(dataset["text"])

    # Create the Perplexity metric instance
    metric = Perplexity(model="gpt2-large", batch_size=8, stride=512, progress_bar=True)

    # Compute perplexity using the calculate method on the joined text
    ppl = metric.calculate(None, joined_text)
    print(f"Perplexity: {ppl}")
    print("Expected perplexity (from Hugging Face report): ~16.44")

    # RESULTS:
    # Perplexity: 16.44437599182129
    # Expected perplexity (from Hugging Face report): ~16.44