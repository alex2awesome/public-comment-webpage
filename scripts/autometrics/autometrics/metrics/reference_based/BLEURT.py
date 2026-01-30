import torch
from typing import List, Union, ClassVar
import warnings
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class BLEURT(ReferenceBasedMetric):
    """---
# Metric Card for BLEURT

BLEURT (Bilingual Evaluation Understudy with Representations from Transformers) is a learned evaluation metric for natural language generation. It combines BERT-based contextual embeddings with a regression model fine-tuned on human-annotated data. BLEURT produces scalar scores that aim to reflect fluency, grammaticality, and semantic adequacy by measuring the similarity between generated and reference texts. It has demonstrated strong correlation with human judgment in machine translation and summarization tasks.

## Metric Details

### Metric Description

BLEURT is a regression-based evaluation metric trained to approximate human judgment of text quality. It leverages transfer learning in multiple stages: beginning with a pretrained BERT model, then performing further pretraining on synthetically noised data to improve robustness, and finally fine-tuning on human-labeled ratings from the WMT Metrics Shared Tasks.

The metric compares a generated sentence (candidate) to a human-written sentence (reference) and produces a scalar score indicating how well the candidate matches the reference in terms of fluency and adequacy. Different BLEURT checkpoints (e.g., BLEURT-20, BLEURT-20-D12) vary in size, accuracy, and multilingual support.

- **Metric Type:** Semantic Similarity  
- **Range:** Approximately 0 to 1 (but may exceed 1 or fall below 0 depending on checkpoint and inputs)  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** No  

### Formal Definition

Let $x$ be the reference sentence and $\hat{x}$ the candidate sentence. BLEURT computes a learned score:

$$
\text{BLEURT}(x, \hat{x}) = f_\theta(x, \hat{x})
$$

where $f_\theta$ is a regression model (typically based on BERT or RemBERT) fine-tuned to predict human ratings of text similarity. The model is trained to minimize mean squared error on labeled sentence pairs.

### Inputs and Outputs

- **Inputs:**  
  - Generated text (candidate sentence)  
  - Reference text (reference sentence)  
  - BLEURT checkpoint (e.g., `BLEURT-20`, `bleurt-base-128`, etc.)

- **Outputs:**  
  - A scalar score for each sentence pair, representing similarity and adequacy  

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Machine Translation, Summarization, Paraphrasing, Data-to-Text Generation  

### Applicability and Limitations

- **Best Suited For:**  
  Sentence-level evaluation where reference-based semantic adequacy is important. Particularly effective for machine translation, summarization, and tasks with high-quality references.

- **Not Recommended For:**  
  - Reference-free evaluation settings  
  - Tasks with highly diverse valid outputs (e.g., open-ended dialogue or storytelling) where many correct outputs may not resemble the reference lexically or structurally

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [google-research/bleurt](https://github.com/google-research/bleurt)  
  - [Hugging Face Evaluate: BLEURT](https://huggingface.co/spaces/evaluate-metric/bleurt)  
  - [lucadiliello/bleurt-pytorch](https://github.com/lucadiliello/bleurt-pytorch)

### Computational Complexity

- **Efficiency:**  
  BLEURT is significantly more computationally expensive than n-gram based metrics like BLEU. It requires loading a large transformer model and computing contextual embeddings and a regression head per sentence pair.

- **Scalability:**  
  BLEURT supports batched inference and length-based batching to speed up evaluation on large corpora. Distilled checkpoints (e.g., BLEURT-20-D6) provide faster and smaller alternatives.

## Known Limitations

- BLEURT scores may vary significantly across different checkpoints; scores are not directly comparable across models.
- Output scores may fall outside the nominal 0–1 range.
- BLEURT may reflect biases present in the pretrained models and training data.
- BLEURT primarily supports English; while BLEURT-20 supports several other languages (e.g., French, Chinese, German), its performance in low-resource or code-mixed settings is less validated.

- **Biases:**  
  BLEURT inherits biases from BERT/RemBERT and from the WMT training annotations. Its judgments may reflect linguistic and cultural norms present in the training data.

- **Task Misalignment Risks:**  
  BLEURT assumes a single reference and may unfairly penalize valid alternative phrasings or creative outputs not matching the reference closely.

- **Failure Cases:**  
  BLEURT may overvalue surface similarity or penalize outputs that are fluent but structurally divergent from the reference. It may also be unreliable on very short or very long sequences.

## Related Metrics

- **BERTScore:** Also uses contextual embeddings but focuses on token-level similarity using cosine similarity.
- **COMET:** Another learned metric based on multilingual encoder-decoder architecture trained on direct assessment data.
- **METEOR:** Incorporates synonym matching and paraphrase tables but is not learned.
- **BLEU/ROUGE:** Surface-level overlap metrics commonly used for baseline evaluation.

## Further Reading

- **Papers:**  
  - [BLEURT: Learning Robust Metrics for Text Generation (Sellam et al., 2020)](https://aclanthology.org/2020.acl-main.704/)  
  - [Learning Compact Metrics for MT (Pu et al., 2021)](https://arxiv.org/abs/2110.06341)

- **Blogs/Tutorials:**  
  - [Google AI Blog: Evaluating Natural Language Generation with BLEURT](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html)

## Citation

```
@inproceedings{sellam-etal-2020-bleurt,  
 title = "{BLEURT}: Learning Robust Metrics for Text Generation",  
 author = "Sellam, Thibault  and  
  Das, Dipanjan  and  
  Parikh, Ankur",  
 editor = "Jurafsky, Dan  and  
  Chai, Joyce  and  
  Schluter, Natalie  and  
  Tetreault, Joel",  
 booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",  
 month = jul,  
 year = "2020",  
 address = "Online",  
 publisher = "Association for Computational Linguistics",  
 url = "https://aclanthology.org/2020.acl-main.704/",  
 doi = "10.18653/v1/2020.acl-main.704",  
 pages = "7881--7892"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 2205.541015625  # in MB
    cpu_mem: ClassVar[float] = 2811.62890625  # in MB
    description: ClassVar[str] = "BLEURT is a learned evaluation metric for natural language generation. It combines BERT-based contextual embeddings with a regression model fine-tuned on human-annotated data. BLEURT produces scalar scores that aim to reflect fluency, grammaticality, and semantic adequacy by measuring the similarity between generated and reference texts."

    def __init__(
        self,
        name: str = "BLEURT",
        description: str = "BLEURT is a learned evaluation metric for natural language generation. It combines BERT-based contextual embeddings with a regression model fine-tuned on human-annotated data. BLEURT produces scalar scores that aim to reflect fluency, grammaticality, and semantic adequacy by measuring the similarity between generated and reference texts.",
        model_name: str = "lucadiliello/BLEURT-20",
        torch_dtype = "float32",
        batch_size: int = 2,
        persistent: bool = True,
        **kwargs
    ):
        # Pass ALL parameters to parent constructor
        super().__init__(
            name=name,
            description=description,
            model_name=model_name,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            persistent=persistent,
            **kwargs
        )
        
        # Store parameters as instance variables
        self.model_name = model_name
        self.torch_dtype = torch.float32 if torch_dtype == "float32" else torch.float16 if torch_dtype == "float16" else torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32
        self.batch_size = batch_size
        self.persistent = persistent
        # Prefer CPU-first for robustness under parallel loads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = None
        self.tokenizer = None
        self.model = None
        self._force_cpu = False
        
        # Exclude parameters that don't affect results from cache key
        self.exclude_from_cache_key('persistent', 'batch_size', 'device')

    def _forward_no_compile(self, inputs):
        """Run model forward while explicitly disabling torch._dynamo capture to avoid meta-device fake tensors."""
        try:
            # torch._dynamo is not guaranteed to exist on older PyTorch versions
            import torch._dynamo as dynamo  # type: ignore[attr-defined]

            @dynamo.disable  # type: ignore[misc]
            def _run(mod, kwargs):
                return mod(**kwargs)

            return _run(self.model, inputs)
        except Exception:
            # Fallback: just run normally
            return self.model(**inputs)

    def _load_model(self):
        """Load BLEURT tokenizer and model."""
        if self.model is None:
            self.config = BleurtConfig.from_pretrained(self.model_name)
            self.tokenizer = BleurtTokenizer.from_pretrained(self.model_name)
            # Load on CPU first, then move to device if CUDA is safe
            self.model = BleurtForSequenceClassification.from_pretrained(
                self.model_name,
                config=self.config
            )
            # Honor force-CPU if a previous fallback occurred
            if not self._force_cpu and str(self.device).startswith('cuda') and torch.cuda.is_available():
                try:
                    self.model = self.model.to(self.device)
                except Exception:
                    self.device = torch.device('cpu')
            self.model.eval()

    def _unload_model(self):
        """Unload the model and tokenizer to free resources."""
        if self.model is not None:
            del self.model, self.tokenizer, self.config
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.config = None

    def _calculate_impl(self,
                  input: str,
                  output: str,
                  references=None,
                  **kwargs) -> float:
        """
        Compute BLEURT score for one candidate-reference pair.
        """
        if references is None:
            references = []
        if self.model is None:
            self._load_model()
        # select first reference if list
        ref = references[0] if isinstance(references, (list, tuple)) and references else (
            references if isinstance(references, str) else ""
        )
        # Use tokenizer's max if available; cap to 512 for safety with BLEURT checkpoints
        tok_max_len = getattr(self.tokenizer, 'model_max_length', 512)
        if tok_max_len is None or tok_max_len > 512:
            tok_max_len = 512
        # tokenize single pair with explicit max_length to prevent position embedding issues
        inputs = self.tokenizer(
            [ref], 
            [output], 
            padding='longest', 
            truncation=True,
            max_length=tok_max_len,
            return_tensors='pt'
        )
        
        # Move tensors to the correct device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                try:
                    inputs[key] = inputs[key].to(self.device, non_blocking=True)
                except Exception:
                    inputs[key] = inputs[key].cpu()
        
        # Validate token ids are within embedding range to avoid CUDA indexing asserts
        try:
            num_embeddings = self.model.get_input_embeddings().num_embeddings  # type: ignore[attr-defined]
            max_id = int(inputs['input_ids'].max())
            min_id = int(inputs['input_ids'].min())
            if max_id >= num_embeddings or min_id < 0:
                warnings.warn(f"[BLEURT] Detected token id out of range (min={min_id}, max={max_id}, vocab={num_embeddings}). Clamping to valid range to avoid device-side asserts.")
                inputs['input_ids'] = inputs['input_ids'].clamp(min=0, max=num_embeddings - 1)
        except Exception:
            # If anything goes wrong with validation, proceed normally
            pass

        # Forward with robust CPU fallback that re-tokenizes to avoid sticky CUDA errors
        try:
            with torch.no_grad():
                outputs = self._forward_no_compile(inputs)
                logits = outputs.logits.flatten()
        except RuntimeError as e:
            msg = str(e).lower()
            if (
                'device-side assert' in msg
                or 'cublas' in msg
                or 'cuda' in msg
                or 'expected device meta' in msg
                or 'device meta' in msg
                or 'meta tensor' in msg
            ):
                warnings.warn("[BLEURT] CUDA/meta issue; forcing BLEURT to CPU and retrying.")
                # Force subsequent calls to remain on CPU
                self._force_cpu = True
                self.device = torch.device('cpu')
                # Re-tokenize cleanly on CPU (avoid moving existing tensors)
                inputs_cpu = self.tokenizer(
                    [ref], [output], padding='longest', truncation=True, max_length=tok_max_len, return_tensors='pt'
                )
                self.model = self.model.to('cpu')
                with torch.no_grad():
                    outputs = self._forward_no_compile(inputs_cpu)
                    logits = outputs.logits.flatten()
            else:
                raise
        score = logits[0].cpu().item()
        if not self.persistent:
            self._unload_model()
        return score

    def _calculate_batched_impl(self,
                          inputs_list: List[str],
                          outputs_list: List[str],
                          references=None,
                          **kwargs) -> List[float]:
        """
        Compute BLEURT scores for batches of candidate-reference pairs.
        """
        if references is None:
            references = [None] * len(outputs_list)
        if self.model is None:
            self._load_model()
        # prepare reference strings
        refs_flat: List[str] = []
        for ref in references:
            if isinstance(ref, (list, tuple)) and ref:
                refs_flat.append(ref[0])
            elif isinstance(ref, str):
                refs_flat.append(ref)
            else:
                refs_flat.append("")
        all_scores: List[float] = []
        for i in range(0, len(outputs_list), self.batch_size):
            chunk_refs = refs_flat[i:i+self.batch_size]
            chunk_outs = outputs_list[i:i+self.batch_size]
            # Use tokenizer's max if available; cap to 512 for safety
            tok_max_len = getattr(self.tokenizer, 'model_max_length', 512)
            if tok_max_len is None or tok_max_len > 512:
                tok_max_len = 512
            inputs = self.tokenizer(
                chunk_refs, 
                chunk_outs, 
                padding='longest', 
                truncation=True,
                max_length=tok_max_len,
                return_tensors='pt'
            )
            
            # Move tensors to the correct device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    try:
                        inputs[key] = inputs[key].to(self.device, non_blocking=True)
                    except Exception:
                        inputs[key] = inputs[key].cpu()
            
            # Validate token ids in batch
            try:
                num_embeddings = self.model.get_input_embeddings().num_embeddings  # type: ignore[attr-defined]
                max_id = int(inputs['input_ids'].max())
                min_id = int(inputs['input_ids'].min())
                if max_id >= num_embeddings or min_id < 0:
                    warnings.warn(f"[BLEURT] (batch) Token id out of range (min={min_id}, max={max_id}, vocab={num_embeddings}). Clamping.")
                    inputs['input_ids'] = inputs['input_ids'].clamp(min=0, max=num_embeddings - 1)
            except Exception:
                pass

            # Forward with fallback
            try:
                with torch.no_grad():
                    outputs = self._forward_no_compile(inputs)
                    logits_tensor = outputs.logits.flatten()
                    logits = logits_tensor.cpu().tolist()
            except RuntimeError as e:
                msg = str(e).lower()
                if (
                    'device-side assert' in msg
                    or 'cublas' in msg
                    or 'cuda' in msg
                    or 'expected device meta' in msg
                    or 'device meta' in msg
                    or 'meta tensor' in msg
                ):
                    warnings.warn("[BLEURT] CUDA/meta issue in batch; forcing CPU and retokenizing this chunk.")
                    self._force_cpu = True
                    self.device = torch.device('cpu')
                    # Re-tokenize on CPU to avoid copying from GPU tensors
                    inputs_cpu = self.tokenizer(
                        chunk_refs, chunk_outs, padding='longest', truncation=True, max_length=tok_max_len, return_tensors='pt'
                    )
                    self.model = self.model.to('cpu')
                    with torch.no_grad():
                        outputs = self._forward_no_compile(inputs_cpu)
                        logits_tensor = outputs.logits.flatten()
                        logits = logits_tensor.cpu().tolist()
                else:
                    raise
            # flatten to list
            if isinstance(logits, float):
                all_scores.append(logits)
            else:
                all_scores.extend(logits)
        if not self.persistent:
            self._unload_model()
        return all_scores 