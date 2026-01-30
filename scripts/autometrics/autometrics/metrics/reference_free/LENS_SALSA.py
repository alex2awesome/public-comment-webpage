import torch
from typing import List, Union, Tuple, ClassVar
from lens import download_model, LENS_SALSA as _LENS_SALSA_Model
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
from types import MethodType

class LENS_SALSA(ReferenceFreeMetric):
    """---
# Metric Card for LENS_SALSA

LENS_SALSA is a reference-free metric designed to evaluate the overall quality of text simplification outputs. It leverages the SALSA (Simplification Analysis via Lexical and Structural Alignment) framework introduced by Heineman et al. (2023), which analyzes edits at the word level to assess whether a simplification succeeds or fails. The LENS_SALSA model uses these insights to produce a scalar simplification quality score based on input-output pairs, with no need for reference texts. This makes it particularly useful in settings where reference simplifications are unavailable or unreliable.

## Metric Details

### Metric Description

LENS_SALSA is a neural, reference-free metric for evaluating sentence-level simplification quality. It builds on the SALSA framework, which identifies and categorizes the types of edits performed when transforming a complex sentence into a simplified one. SALSA aligns input and output tokens using an alignment algorithm and labels each word-level edit with one of several tags—e.g., deletion, substitution, or addition—and further classifies the edit as a *success* or *failure* based on its impact on fluency, adequacy, and simplicity. These labels are derived from a manually annotated corpus.

The LENS_SALSA model is trained using these edit-level annotations. It learns to aggregate the local edit patterns into a global simplification quality score using a supervised regression objective. Crucially, this scoring process does not require reference simplifications at inference time, making LENS_SALSA a practical tool in real-world simplification pipelines.

- **Metric Type:** Reference-Free
- **Range:** Unbounded (empirically observed in [0, 100] scale)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

The LENS_SALSA score is generated via a neural model trained on edit-labeled simplification data. The core of the system is the SALSA framework, which performs alignment and tagging of edits between a complex input sentence $x$ and a simplified candidate $\hat{x}$.

Let:
- $A = \text{Align}(x, \hat{x})$ be the alignment between input and output tokens,
- $E(x, \hat{x}, A)$ be the set of word-level edits extracted from the alignment,
- $T(e)$ be the success/failure tag for an edit $e$ (as determined by the SALSA labeling scheme),
- $f(E)$ be the feature vector summarizing the counts and types of edit tags in $E$.

Then, LENS_SALSA computes the final score using a regression model (MLP):

$$
\text{LENS SALSA}(x, \hat{x}) = \text{MLP}(f(E(x, \hat{x}, A)))
$$

The model is trained using human-annotated quality scores from simplification corpora.

### Inputs and Outputs

- **Inputs:**  
  - Input text (original complex sentence)  
  - Output text (candidate simplified sentence)  
  - (Optional) Reference text(s), used only during training or secondary analysis  
  
- **Outputs:**  
  - Scalar simplification score (float, typically between 0 and 100)  
  - (Optional) Word-level edit tags indicating success/failure for interpretability

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Text Simplification

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating text simplification outputs where reference simplifications are unavailable, unreliable, or highly variable.  
  Particularly effective for sentence-level simplification tasks focused on fluency, adequacy, and simplicity.

- **Not Recommended For:**  
  - Tasks outside of simplification, such as summarization or paraphrasing  
  - Long-form or document-level generation  
  - Settings where simplification quality depends heavily on context beyond a single sentence

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Hugging Face: davidheineman/lens-salsa](https://huggingface.co/davidheineman/lens-salsa)  
  - `autometrics` (custom wrapper around LENS_SALSA model for reference-free evaluation)

### Computational Complexity

- **Efficiency:**  
  Relatively efficient inference via PyTorch-based model. Overhead comes from computing alignment-based features and scoring.

- **Scalability:**  
  Scales to batched input using the `calculate_batched` method. Memory usage depends on model size and batch configuration.

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  Designed specifically for simplification; using it for other tasks may result in misleading evaluations.

- **Failure Cases:**  
  - Very long input texts may cause padding errors in the model
  - For best results, texts should be sentence-level rather than long passages

## Related Metrics

- **SARI:** Reference-based simplification metric often used alongside LENS_SALSA.  
- **BERTScore (adapted to simplification):** Captures semantic similarity between input and output.  
- **LENS Framework:** Edit-level analysis from which LENS_SALSA is derived.

## Further Reading

- **Papers:**  
  - [Dancing Between Success and Failure: Edit-level Simplification Evaluation using SALSA](https://aclanthology.org/2023.emnlp-main.211/)  

- **Blogs/Tutorials:**  
  Needs more information.

## Citation

```
@inproceedings{heineman-etal-2023-dancing,
    title = "Dancing Between Success and Failure: Edit-level Simplification Evaluation using {SALSA}",
    author = "Heineman, David  and
      Dou, Yao  and
      Maddela, Mounica  and
      Xu, Wei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.211/",
    doi = "10.18653/v1/2023.emnlp-main.211",
    pages = "3466--3495"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    # TODO: Check this, because gpu memory being zero is suspicious
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 2909.66796875  # in MB
    description: ClassVar[str] = "LENS_SALSA is a reference-free metric designed to evaluate the overall quality of text simplification outputs. It leverages the SALSA (Simplification Analysis via Lexical and Structural Alignment) framework introduced by Heineman et al. (2023), which analyzes edits at the word level to assess whether a simplification succeeds or fails. The LENS_SALSA model uses these insights to produce a scalar simplification quality score based on input-output pairs, with no need for reference texts. This makes it particularly useful in settings where reference simplifications are unavailable or unreliable."

    def __init__(
        self,
        name: str = "LENS_SALSA",
        description: str = "LENS_SALSA is a reference-free metric designed to evaluate the overall quality of text simplification outputs. It leverages the SALSA (Simplification Analysis via Lexical and Structural Alignment) framework introduced by Heineman et al. (2023), which analyzes edits at the word level to assess whether a simplification succeeds or fails. The LENS_SALSA model uses these insights to produce a scalar simplification quality score based on input-output pairs, with no need for reference texts. This makes it particularly useful in settings where reference simplifications are unavailable or unreliable.",
        model_id: str = "davidheineman/lens-salsa",
        batch_size: int = 16,
        devices: List[int] = None,
        persistent: bool = True,
        max_length: int = 512,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(name, description, model_id=model_id, batch_size=batch_size, devices=devices, persistent=persistent, max_length=max_length, max_retries=max_retries, **kwargs)
        self.model_id = model_id
        self.batch_size = batch_size
        self.devices = devices
        self.persistent = persistent
        self.model = None
        self.input_column = "src"
        self.output_column = "edit_id_simplified"
        self.max_length = max_length
        self.max_retries = max_retries
        
        # Standard XLM-RoBERTa model has 512 token limit (LENS uses this model)
        self.model_token_limit = 512
        # We conservatively use only a fraction of the limit so that when the
        # model later concatenates special tokens / source / target sequences
        # we stay well within bounds.
        self.truncation_fraction = 0.85  # keep ≤85 % of limit as a safeguard
        
        self.exclude_from_cache_key('batch_size', 'devices', 'persistent', 'max_retries')

    def _load_model(self):
        """Download SALSA checkpoint and load the LENS_SALSA model."""
        if self.model is None:
            ckpt_path = download_model(self.model_id)
            
            try:
                self.model = _LENS_SALSA_Model(ckpt_path)
            except RuntimeError as e:
                # Handle CUDA assert during checkpoint load by retrying on CPU
                if "device-side assert" in str(e) or "CUDA error" in str(e):
                    import os
                    old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        self.model = _LENS_SALSA_Model(ckpt_path)
                    finally:
                        # Restore original CUDA visibility
                        if old_cuda_visible is None:
                            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                        else:
                            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
                else:
                    raise e
            except NotImplementedError as e:
                # Handle meta tensor issue
                if "Cannot copy out of meta tensor" in str(e):
                    print(f"    🔧 Meta tensor issue detected for {self.model_id}, using to_empty()...")
                    # Patch torch.nn.Module.to to handle meta tensors globally
                    import torch.nn.modules.module
                    original_to = torch.nn.modules.module.Module.to
                    
                    def patched_to(self, *args, **kwargs):
                        """Patched to() method that uses to_empty() for meta tensors."""
                        try:
                            return original_to(self, *args, **kwargs)
                        except NotImplementedError as e:
                            if "Cannot copy out of meta tensor" in str(e):
                                # Use to_empty() instead
                                return self.to_empty(*args, **kwargs)
                            else:
                                raise e
                    
                    # Apply the patch
                    torch.nn.modules.module.Module.to = patched_to
                    
                    try:
                        # Try loading again with the patched method
                        self.model = _LENS_SALSA_Model(ckpt_path)
                    finally:
                        # Restore the original method
                        torch.nn.modules.module.Module.to = original_to
                else:
                    raise e
            
            # Update column names to match what the model expects
            self.input_column = self.model.source_column
            self.output_column = self.model.target_column

            # ------------------------------------------------------------------
            # Monkey-patch LENS encoder utilities to avoid rare assertion errors
            # caused by negative padding lengths when the span mask length is
            # larger than the token sequence length (this can happen if the
            # underlying model is forced to truncate the token sequence to the
            # 512-token limit but the span mask is not truncated accordingly).
            # Instead of raising an AssertionError, we now gracefully truncate
            # the mask so that it always fits the requested length.
            # ------------------------------------------------------------------

            def _safe_pad_tensor(this, tensor, length, padding_index):
                """Robust replacement for Encoder.pad_tensor.

                If the incoming *tensor* is already longer than *length* we
                simply truncate it; otherwise we fall back to the original
                right-padding behaviour.  This mirrors what happens to the
                token sequences themselves and prevents the negative-padding
                assertion that crashes evaluation jobs on long inputs.
                """
                import torch  # Import torch inside the function scope
                
                if tensor.shape[0] > length:
                    # Truncate rather than assert – this keeps span masks and
                    # token sequences aligned after the upstream truncation
                    # performed inside `Encoder.concat_sequences`.
                    return tensor[:length]

                n_padding = length - tensor.shape[0]
                if n_padding == 0:
                    return tensor

                padding = tensor.new_full((n_padding, *tensor.shape[1:]), padding_index)
                return torch.cat((tensor, padding), dim=0)

            # Replace the existing method if present (should be on every LENS
            # encoder instance) so that all subsequent calls use the safe
            # version defined above.  The encoder lives under
            # `self.model.model.encoder` in LENS checkpoints.

            enc = None
            if hasattr(self.model, "encoder"):
                enc = self.model.encoder  # some checkpoints expose it here
            elif hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
                enc = self.model.model.encoder

            if enc is not None and hasattr(enc, "pad_tensor"):
                enc.pad_tensor = MethodType(_safe_pad_tensor, enc)

            # ------------------------------------------------------------------
            # Monkey-patch UnifiedMetric.compute_loss to be device-agnostic.
            # Upstream calls .cuda() inside compute_loss, which crashes on CPU
            # or when Lightning has placed the module on a different device.
            # We override it to move tensors to the prediction/logit device
            # and align loss weights accordingly.
            # ------------------------------------------------------------------

            def _safe_compute_loss(this, prediction, target):
                import torch  # local import to keep module namespace clean

                sentence_loss = this.sentloss(prediction.scores, target.scores)
                if this.word_level:
                    sentence_loss = this.sentloss(prediction.scores, target.scores)
                    device = prediction.logits.device

                    if this.continuous_word_labels:
                        pred_vec = (
                            prediction.logits.reshape(-1, 1)
                            .view(-1)
                            .to(device=device, dtype=torch.float16)
                        )
                        tgt_vec = target.labels.view(-1).to(device=device, dtype=torch.float16)
                    else:
                        pred_vec = prediction.logits.reshape(-1, this.num_token_spans).to(device=device)
                        tgt_vec = target.labels.view(-1).to(device=device, dtype=torch.long)

                    # Ensure loss weights live on the same device
                    if hasattr(this, "wordloss") and hasattr(this.wordloss, "weight") and this.wordloss.weight is not None:
                        if this.wordloss.weight.device != device:
                            this.wordloss.weight = this.wordloss.weight.to(device)

                    word_loss = this.wordloss(pred_vec, tgt_vec) if hasattr(this, "wordloss") else 0.0
                    return sentence_loss * (1 - this.hparams.loss_lambda) + word_loss * (this.hparams.loss_lambda)

                return sentence_loss

            # Bind method if model matches expected structure
            inner_model = getattr(self.model, "model", None)
            if inner_model is not None and hasattr(inner_model, "compute_loss"):
                try:
                    inner_model.compute_loss = MethodType(_safe_compute_loss, inner_model)
                except Exception:
                    pass

    def _unload_model(self):
        """Unload SALSA model to free resources."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None
    
    def _get_tokenizer_length(self, text: str) -> int:
        """
        Get an approximate token count using the model's tokenizer.
        If the model isn't loaded, make a rough estimate based on whitespace.
        """
        if self.model is not None and hasattr(self.model.model, "encoder"):
            if hasattr(self.model.model.encoder, "tokenizer"):
                # If we have access to the tokenizer, use it for accurate counts
                return len(self.model.model.encoder.tokenizer.tokenize(text))
        
        # Fallback: rough estimate based on whitespace tokens
        # Rule of thumb: ~1.5 tokens per word for multilingual models
        return len(text.split()) * 2

    def _truncate_text(self, text: str, max_tokens: int = None) -> str:
        """
        Truncate *individual* text segments (source or target) so that the
        eventual concatenation performed inside the SALSA model will not exceed
        its maximum token length.

        We do this by allocating **half** of an *overall* budget that is
        slightly smaller (``truncation_fraction``) than the true
        ``model_token_limit``.  Concretely, if no ``max_tokens`` is supplied
        we use:

        ``int(model_token_limit * truncation_fraction / 2)``

        This keeps the *combined* source-plus-target length within
        ``model_token_limit * truncation_fraction`` while leaving additional
        room for any special tokens the model may insert.
        """
        if not text:
            return text
        
        if max_tokens is None:
            # Allocate half of the *reduced* budget to this single segment.
            max_tokens = int(self.model_token_limit * self.truncation_fraction / 2)

        # Start with a rough character-based truncation for efficiency
        # (most multilingual tokenizers average ~4 chars per token)
        if len(text) > max_tokens * 6:
            text = text[:max_tokens * 6]
            
        # Then do a more precise token-based truncation
        approx_tokens = self._get_tokenizer_length(text)
        
        if approx_tokens <= max_tokens:
            return text
        
        # Truncate by recursively removing words from end until under token limit
        words = text.split()
        while words and self._get_tokenizer_length(" ".join(words)) > max_tokens:
            words.pop()
            
        return " ".join(words)

    def _predict_score(self, input_text: str, output_text: str) -> float:
        """
        Compute the SALSA score for a single input/output pair **without** any
        multi-step fallback logic.  Any exception raised by the underlying model
        will be propagated to the caller so that failures are transparent.
        """

        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""

        # Treat empty or whitespace-only strings as a neutral failure score
        if not input_text or not input_text.strip() or not output_text or not output_text.strip():
            return 0.0

        truncated_input = self._truncate_text(input_text)
        truncated_output = self._truncate_text(output_text)

        # If truncation results in empty strings, also treat as neutral failure
        if not truncated_input.strip() or not truncated_output.strip():
            return 0.0

        all_data = [{
            self.input_column: truncated_input.lower(),
            self.output_column: truncated_output.lower(),
            "id": "0"
        }]

        prediction = self.model.model.predict(
            all_data,
            batch_size=self.batch_size,
            devices=self.devices,
        )

        return float(prediction.scores[0]) * 100

    def _calculate_impl(self,
              input: str,
              output: str,
              references: Union[List[str], None] = None,
              **kwargs) -> float:
        """
        Compute overall SALSA score for a single example.
        Returns a float score.
        """
        if self.model is None:
            self._load_model()

        input = str(input) if input is not None else ""
        output = str(output) if output is not None else ""

        result = self._predict_score(input, output)
        
        if not self.persistent:
            self._unload_model()

        return result

    def _calculate_batched_impl(self,
                          inputs: List[str],
                          outputs: List[str],
                          references: Union[List[List[str]], None] = None,
                          **kwargs) -> List[float]:
        """
        Compute overall SALSA scores for a batch of examples.

        This version processes each input individually **once** — it does
        not attempt any progressive truncation retries.  Any exception
        raised while scoring will propagate so that the caller has full
        visibility into the failure.
        """
        if self.model is None:
            self._load_model()

        inputs = [str(input) if input is not None else "" for input in inputs]
        outputs = [str(output) if output is not None else "" for output in outputs]

        results: List[float] = []
        try:
            for input_text, output_text in zip(inputs, outputs):
                # Handle empty input/output without hitting the model
                if not input_text or not input_text.strip() or not output_text or not output_text.strip():
                    results.append(0.0)
                    continue

                input_text = str(input_text) if input_text is not None else ""
                output_text = str(output_text) if output_text is not None else ""

                score = self._predict_score(input_text, output_text)
                results.append(score)
        finally:
            if not self.persistent:
                self._unload_model()

        return results 