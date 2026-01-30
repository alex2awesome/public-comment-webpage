import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
from typing import ClassVar

class FactCC(ReferenceFreeMetric):
    """---
# Metric Card for FactCC

FactCC is a reference-free metric designed to evaluate the factual consistency of summaries with respect to source documents. It uses a fine-tuned BERT-based sequence classification model trained on weakly supervised data to determine whether a summary sentence is factually supported by its source. The model was specifically developed to overcome the limitations of traditional summarization metrics that fail to capture factual inconsistencies.

## Metric Details

### Metric Description

FactCC formulates factual consistency as a binary classification task: given a source document and a candidate summary, the model predicts whether the summary is factually consistent (`CORRECT`) or inconsistent (`INCORRECT`) with the source. To train the model, synthetic inconsistent examples are generated using rule-based transformations such as entity swaps, negation, and date/number perturbations. The model is then fine-tuned using this weakly supervised dataset and evaluated against human-annotated summaries.

- **Metric Type:** Faithfulness  
- **Range:** [0, 1] (probability of label; final output is binary `CORRECT` or `INCORRECT`)  
- **Higher is Better?:** Yes (with respect to correctness probability or accuracy over a dataset)  
- **Reference-Based?:** No  
- **Input-Required?:** Yes  

### Formal Definition

Let $x$ denote the source document and $\hat{x}$ the generated summary. FactCC uses a BERT-based classifier to compute:

$$
\hat{y} = \arg\max _{y \in \{\text{CORRECT}, \text{INCORRECT}\}} f(x, \hat{x})
$$

where $f$ is the classification function learned by the model, trained on synthetic (x, summary, label) triples generated from CNN/DailyMail.

### Inputs and Outputs

- **Inputs:**  
  - Source document (string)  
  - Summary (string)  

- **Outputs:**  
  - Label: `CORRECT` or `INCORRECT`  
  - Confidence score (optional): probability assigned to the predicted label

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:** Summarization  

### Applicability and Limitations

- **Best Suited For:**  
  Factual consistency evaluation of abstractive summaries, particularly when reference summaries are unavailable or unreliable.

- **Not Recommended For:**  
  Tasks requiring nuanced semantic judgments beyond factual consistency, such as coherence, fluency, or stylistic alignment.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Hugging Face Transformers](https://huggingface.co/manueldeprada/FactCC)  
  - [Original GitHub Repository](https://github.com/salesforce/factCC) (archived)  

### Computational Complexity

- **Efficiency:**  
  Comparable to standard BERT-based classification; inference time is linear in input length and batch size.  

- **Scalability:**  
  Scales well with GPU acceleration and batched inference; model supports max input lengths up to 512 tokens and can be used with pipelines for large-scale evaluation.

## Known Limitations

- **Biases:**  
  Trained primarily on CNN/DailyMail, which may bias performance toward that domain. Synthetic negative examples may not reflect real-world inconsistencies in all domains.  

- **Task Misalignment Risks:**  
  May fail to detect hallucinations that are not captured by training transformations. Not suitable for evaluating semantic quality or fluency.  

- **Failure Cases:**  
  - May misclassify paraphrased or implicature-based summaries as incorrect  
  - Sensitive to token truncation due to 512-token max input length  

## Related Metrics

- **QAGS:** Uses QA-based probing to evaluate consistency  
- **SummaQA:** Another question-answering-based factuality metric  
- **BERTScore:** Captures semantic similarity but not factuality  
- **DAE (Dependency Arc Entailment):** Focuses on syntactic entailment between summary and source  

## Further Reading

- **Papers:**  
  - [Kryściński et al., 2020 (EMNLP)](https://aclanthology.org/2020.emnlp-main.750/)  
  - [Original ArXiv Preprint](https://arxiv.org/abs/1910.12840)  

- **Blogs/Tutorials:**  
  - [More Information Needed]

## Citation

```
@inproceedings{kryscinski-etal-2020-evaluating,  
  title = "Evaluating the Factual Consistency of Abstractive Text Summarization",  
  author = "Kryscinski, Wojciech  and  
    McCann, Bryan  and  
    Xiong, Caiming  and  
    Socher, Richard",  
  editor = "Webber, Bonnie  and  
    Cohn, Trevor  and  
    He, Yulan  and  
    Liu, Yang",  
  booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",  
  month = nov,  
  year = "2020",  
  address = "Online",  
  publisher = "Association for Computational Linguistics",  
  url = "https://aclanthology.org/2020.emnlp-main.750/",  
  doi = "10.18653/v1/2020.emnlp-main.750",  
  pages = "9332--9346"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 426.86083984375  # in MB
    cpu_mem: ClassVar[float] = 1317.58984375  # in MB
    description: ClassVar[str] = "FactCC is a reference-free metric designed to evaluate the factual consistency of summaries with respect to source documents. It uses a fine-tuned BERT-based sequence classification model trained on weakly supervised data to determine whether a summary sentence is factually supported by its source. The model was specifically developed to overcome the limitations of traditional summarization metrics that fail to capture factual inconsistencies."

    def __init__(
        self,
        name: str = "FactCC",
        description: str = "FactCC is a reference-free metric designed to evaluate the factual consistency of summaries with respect to source documents. It uses a fine-tuned BERT-based sequence classification model trained on weakly supervised data to determine whether a summary sentence is factually supported by its source. The model was specifically developed to overcome the limitations of traditional summarization metrics that fail to capture factual inconsistencies.",
        model_name: str = "manueldeprada/FactCC",
        device: str = None,
        batch_size: int = 8,
        persistent: bool = True,
        **kwargs
    ):
        super().__init__(name, description, model_name=model_name, device=device, batch_size=batch_size, persistent=persistent, **kwargs)
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.persistent = persistent
        self.tokenizer = None
        self.model = None

        self.exclude_from_cache_key('model_name', 'device', 'batch_size', 'persistent')

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load on CPU first to avoid meta/cuda placement issues, then move
            try:
                cpu_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ).eval()
                self.model = cpu_model.to(self.device)
            except NotImplementedError as e:
                # Handle meta tensor issue
                if "Cannot copy out of meta tensor" in str(e):
                    print(f"    🔧 Meta tensor issue detected for {self.model_name}, using to_empty()...")
                    base = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name
                    )
                    self.model = base.to_empty(device=self.device).eval()
                else:
                    raise e
            # Normalize model dtype to float32 to avoid BF16/FP32 matmul issues
            try:
                for p in self.model.parameters():
                    if p.dtype.is_floating_point:
                        p.data = p.data.float()
                for b_name, b in list(self.model.named_buffers(recurse=True)):
                    if hasattr(b, 'dtype') and b.dtype.is_floating_point:
                        self.model._buffers[b_name] = b.float()  # type: ignore[index]
            except Exception:
                pass

    def _unload_model(self):
        if getattr(self, 'model', None) is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.model = None
        if getattr(self, 'tokenizer', None) is not None:
            self.tokenizer = None

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs) -> float:
        # Lazy load
        if self.model is None:
            self._load_model()
        input_text = str(input_text) if input_text is not None else ""
        output = str(output) if output is not None else ""
        
        # Store original texts for potential CPU fallback
        original_input_text = input_text
        original_output = output
        
        # Encode text-summary pair with strict truncation/padding to model max
        encoded = self.tokenizer(
            input_text,
            output,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        # Sanitize token ids and masks, then move to device
        vocab_size = getattr(self.model.config, 'vocab_size', None)
        if vocab_size is not None and 'input_ids' in encoded:
            encoded['input_ids'] = torch.clamp(encoded['input_ids'], min=0, max=vocab_size - 1)
        # Ensure token_type_ids are within {0,1} if present (BERT-style)
        if 'token_type_ids' in encoded:
            tti = encoded['token_type_ids']
            if isinstance(tti, torch.Tensor):
                encoded['token_type_ids'] = torch.clamp(tti, min=0, max=1)
        # Force attention_mask to be boolean 0/1 on same device
        if 'attention_mask' in encoded and isinstance(encoded['attention_mask'], torch.Tensor):
            am = encoded['attention_mask']
            encoded['attention_mask'] = (am != 0).to(dtype=torch.bool)
        encoded = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in encoded.items()}
        with torch.no_grad():
            try:
                logits = self.model(**encoded).logits.float()
                probs = F.softmax(logits, dim=-1)
            except RuntimeError as e:
                # Handle rare GPU matmul/cublas failures by retrying on CPU
                if ('CUBLAS_STATUS_EXECUTION_FAILED' in str(e) or 'cublas' in str(e).lower() or
                    'device-side assert' in str(e).lower()):
                    # Clear CUDA cache to free corrupted tensors
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    
                    # Recreate inputs on CPU from original text (don't try to move corrupted tensors)
                    cpu_encoded = self.tokenizer(
                        original_input_text,
                        original_output,
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        return_attention_mask=True
                    )
                    # Sanitize token ids and masks for CPU
                    if vocab_size is not None and 'input_ids' in cpu_encoded:
                        cpu_encoded['input_ids'] = torch.clamp(cpu_encoded['input_ids'], min=0, max=vocab_size - 1)
                    if 'token_type_ids' in cpu_encoded:
                        tti = cpu_encoded['token_type_ids']
                        if isinstance(tti, torch.Tensor):
                            cpu_encoded['token_type_ids'] = torch.clamp(tti, min=0, max=1)
                    if 'attention_mask' in cpu_encoded and isinstance(cpu_encoded['attention_mask'], torch.Tensor):
                        am = cpu_encoded['attention_mask']
                        cpu_encoded['attention_mask'] = (am != 0).to(dtype=torch.bool)
                    
                    # Move model to CPU and run inference
                    try:
                        cpu_model = self.model.to('cpu')
                        logits = cpu_model(**cpu_encoded).logits.float()
                        probs = F.softmax(logits, dim=-1)
                        
                        # Move model back if persistent and GPU available
                        if self.persistent and torch.cuda.is_available():
                            self.model = self.model.to(self.device)
                    except Exception as cpu_e:
                        # If CPU fallback also fails, return a default score
                        print(f"Warning: Both GPU and CPU inference failed for FactCC. GPU error: {e}, CPU error: {cpu_e}")
                        return 0.5  # Return neutral score
                else:
                    raise
        # Probability of CORRECT label
        label2id = self.model.config.label2id  # e.g. {'INCORRECT':0, 'CORRECT':1}
        correct_id = label2id.get("CORRECT", 1)
        score = probs[0, correct_id].item()
        if not self.persistent:
            self._unload_model()
        return score

    def _calculate_batched_impl(self, inputs: list, outputs: list, references=None, **kwargs) -> list:
        # Lazy load
        if self.model is None:
            self._load_model()
        inputs = [str(inp) if inp is not None else "" for inp in inputs]
        outputs = [str(out) if out is not None else "" for out in outputs]
        all_scores = []
        # Process in batches
        for i in range(0, len(inputs), self.batch_size):
            batch_src = inputs[i:i+self.batch_size]
            batch_out = outputs[i:i+self.batch_size]
            
            # Store original batch texts for potential CPU fallback
            original_batch_src = batch_src.copy()
            original_batch_out = batch_out.copy()
            
            encoded = self.tokenizer(
                batch_src,
                batch_out,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            # Sanitize token ids and masks, then move to device
            vocab_size = getattr(self.model.config, 'vocab_size', None)
            if vocab_size is not None and 'input_ids' in encoded:
                encoded['input_ids'] = torch.clamp(encoded['input_ids'], min=0, max=vocab_size - 1)
            if 'token_type_ids' in encoded:
                tti = encoded['token_type_ids']
                if isinstance(tti, torch.Tensor):
                    encoded['token_type_ids'] = torch.clamp(tti, min=0, max=1)
            if 'attention_mask' in encoded and isinstance(encoded['attention_mask'], torch.Tensor):
                am = encoded['attention_mask']
                encoded['attention_mask'] = (am != 0).to(dtype=torch.bool)
            encoded = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in encoded.items()}
            with torch.no_grad():
                try:
                    logits = self.model(**encoded).logits.float()
                    probs = F.softmax(logits, dim=-1).cpu()
                except RuntimeError as e:
                    if ('CUBLAS_STATUS_EXECUTION_FAILED' in str(e) or 'cublas' in str(e).lower() or
                        'device-side assert' in str(e).lower()):
                        # Clear CUDA cache to free corrupted tensors
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        
                        # Recreate inputs on CPU from original text (don't try to move corrupted tensors)
                        cpu_encoded = self.tokenizer(
                            original_batch_src,
                            original_batch_out,
                            max_length=512,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                            return_attention_mask=True
                        )
                        # Sanitize token ids and masks for CPU
                        if vocab_size is not None and 'input_ids' in cpu_encoded:
                            cpu_encoded['input_ids'] = torch.clamp(cpu_encoded['input_ids'], min=0, max=vocab_size - 1)
                        if 'token_type_ids' in cpu_encoded:
                            tti = cpu_encoded['token_type_ids']
                            if isinstance(tti, torch.Tensor):
                                cpu_encoded['token_type_ids'] = torch.clamp(tti, min=0, max=1)
                        if 'attention_mask' in cpu_encoded and isinstance(cpu_encoded['attention_mask'], torch.Tensor):
                            am = cpu_encoded['attention_mask']
                            cpu_encoded['attention_mask'] = (am != 0).to(dtype=torch.bool)
                        
                        # Move model to CPU and run inference
                        try:
                            cpu_model = self.model.to('cpu')
                            logits = cpu_model(**cpu_encoded).logits.float()
                            probs = F.softmax(logits, dim=-1).cpu()
                            
                            # Move model back if persistent and GPU available
                            if self.persistent and torch.cuda.is_available():
                                self.model = self.model.to(self.device)
                        except Exception as cpu_e:
                            # If CPU fallback also fails, return default scores for this batch
                            print(f"Warning: Both GPU and CPU inference failed for FactCC batch. GPU error: {e}, CPU error: {cpu_e}")
                            batch_size = len(original_batch_src)
                            probs = torch.full((batch_size, 2), 0.5)  # Neutral scores for all items in batch
                    else:
                        raise
            label2id = self.model.config.label2id
            correct_id = label2id.get("CORRECT", 1)
            batch_scores = probs[:, correct_id].tolist()
            all_scores.extend(batch_scores)
        if not self.persistent:
            self._unload_model()
        return all_scores 