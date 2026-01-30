import torch
import torch.nn.functional as F
from typing import List, Union, ClassVar
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric

class Sentiment(ReferenceFreeMetric):
    """---
# Metric Card for Sentiment

The Sentiment metric quantifies the sentiment polarity of generated text using a pretrained model fine-tuned on Twitter data: `cardiffnlp/twitter-roberta-base-sentiment-latest`. It converts the model's categorical output (negative, neutral, positive) into a continuous regression score by computing the normalized difference between positive and negative class probabilities. This allows sentiment to be used as a reference-free, scalar evaluation metric for generation tasks.

## Metric Details

### Metric Description

This metric uses the Cardiff NLP Twitter RoBERTa-based model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to classify input text into three categories: negative, neutral, and positive. Instead of returning a discrete label, it converts the model's output probabilities into a scalar sentiment score by computing:

$$
\text{Score} = \frac{p_{\text{positive}} - p_{\text{negative}}}{2}
$$

This results in a range of $[-0.5, 0.5]$, where higher values indicate more positive sentiment and lower values indicate more negative sentiment.

- **Metric Type:** Reference-Free  
- **Range:** $[-0.5, 0.5]$  
- **Higher is Better?:** Yes (when positive sentiment is desirable)  
- **Reference-Based?:** No  
- **Input-Required?:** No

### Formal Definition

Given a generated output string $\hat{y}$, the model predicts class probabilities:
- $p_{\text{neg}}$: probability of negative sentiment
- $p_{\text{neu}}$: probability of neutral sentiment
- $p_{\text{pos}}$: probability of positive sentiment

The sentiment score is computed as:

$$
\text{Sentiment}(\hat{y}) = \frac{p_{\text{pos}} - p_{\text{neg}}}{2}
$$

### Inputs and Outputs

- **Inputs:**  
  - Generated text (output string)

- **Outputs:**  
  - Scalar sentiment score in $[-0.5, 0.5]$

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems  
- **Tasks:** Dialogue Generation, Response Generation, Creative Writing

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating sentiment polarity of model-generated outputs, particularly in tasks where emotional tone or user alignment matters (e.g., empathetic chatbots, personalized generation).

- **Not Recommended For:**  
  Tasks where neutrality or factuality is the goal (e.g., summarization, translation). The metric does not account for semantic correctness or fluency.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`)

### Computational Complexity

- **Efficiency:**  
  The model performs a forward pass through a transformer for each input, with complexity roughly $O(n)$ in sequence length. Inference is efficient but not suitable for extremely high-throughput settings without batching.

- **Scalability:**  
  Supports batched inference for moderate-scale evaluation (e.g., thousands of generations).

## Known Limitations

- **Biases:**  
  The model is trained on Twitter data and may reflect social media-specific biases, such as overemphasis on sarcasm, slang, or cultural expressions common to Twitter users.

- **Task Misalignment Risks:**  
  Applying this metric to domains outside Twitter or to tasks where sentiment is irrelevant can result in misleading evaluations.

- **Failure Cases:**  
  - Neutral content may still yield non-zero scores.  
  - The model can misinterpret ambiguous or sarcastic statements.  
  - Sentiment extremes may be underrepresented or flattened due to the regression mapping.

## Related Metrics

- **BERTScore:** Measures semantic similarity, not sentiment.  
- **Toxicity (e.g., FastTextToxicity):** Related in spirit but designed to identify harmful rather than emotionally valenced content.  
- **PPLM Sentiment Control Metrics:** Used in controlled generation evaluation but not standardized.

## Further Reading

- **Papers:**  
  - [TweetNLP: Cutting-Edge Natural Language Processing for Social Media](https://aclanthology.org/2022.emnlp-demos.5)  
  - [TimeLMs: Diachronic Language Models from Twitter](https://aclanthology.org/2022.acl-demo.25)

- **Blogs/Tutorials:**  
  - [CardiffNLP Model Hub (Hugging Face)](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

## Citation

```
@inproceedings{camacho-collados-etal-2022-tweetnlp,  
    title = "{T}weet{NLP}: Cutting-Edge Natural Language Processing for Social Media",  
    author = "Camacho-collados, Jose  and  
      Rezaee, Kiamehr  and  
      Riahi, Talayeh  and  
      Ushio, Asahi  and  
      Loureiro, Daniel  and  
      Antypas, Dimosthenis  and  
      Boisson, Joanne  and  
      Espinosa Anke, Luis  and  
      Liu, Fangyu  and  
      Mart{\'\i}nez C{\'a}mara, Eugenio",  
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",  
    month = dec,  
    year = "2022",  
    address = "Abu Dhabi, UAE",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2022.emnlp-demos.5",  
    pages = "38--49"  
}
```

```
@inproceedings{loureiro-etal-2022-timelms,  
    title = "{T}ime{LM}s: Diachronic Language Models from {T}witter",  
    author = "Loureiro, Daniel  and  
      Barbieri, Francesco  and  
      Neves, Leonardo  and  
      Espinosa Anke, Luis  and  
      Camacho-collados, Jose",  
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",  
    month = may,  
    year = "2022",  
    address = "Dublin, Ireland",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2022.acl-demo.25",  
    doi = "10.18653/v1/2022.acl-demo.25",  
    pages = "251--260"  
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 484.86181640625  # in MB
    cpu_mem: ClassVar[float] = 1393.8125  # in MB
    description: ClassVar[str] = "The Sentiment metric quantifies the sentiment polarity of generated text using a pretrained model fine-tuned on Twitter data: `cardiffnlp/twitter-roberta-base-sentiment-latest`. It converts the model's categorical output (negative, neutral, positive) into a continuous regression score by computing the normalized difference between positive and negative class probabilities. This allows sentiment to be used as a reference-free, scalar evaluation metric for generation tasks."

    def __init__(
        self,
        name: str = "Sentiment",
        description: str = "The Sentiment metric quantifies the sentiment polarity of generated text using a pretrained model fine-tuned on Twitter data: `cardiffnlp/twitter-roberta-base-sentiment-latest`. It converts the model's categorical output (negative, neutral, positive) into a continuous regression score by computing the normalized difference between positive and negative class probabilities. This allows sentiment to be used as a reference-free, scalar evaluation metric for generation tasks.",
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        torch_dtype = "float32",
        device_map: Union[str, dict] = None,
        batch_size: int = 8,
        persistent: bool = True,
        **kwargs
    ):
        super().__init__(name, description, model_name=model_name, torch_dtype=torch_dtype, device_map=device_map, batch_size=batch_size, persistent=persistent, **kwargs)
        self.model_name = model_name
        self.torch_dtype = torch.float32 if torch_dtype == "float32" else torch.float16 if torch_dtype == "float16" else torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32
        self.device_map = device_map
        self.batch_size = batch_size
        self.persistent = persistent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.config = None

        self.exclude_from_cache_key('batch_size', 'device_map', 'persistent')

    def _load_model(self):
        """Load tokenizer, model, and config."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Some HF models define model_max_length = int(1e30). We cap it at 512 so that
            # truncation=True works as expected instead of silently skipping.
            self.tokenizer.model_max_length = 512
            self.config = AutoConfig.from_pretrained(self.model_name)
            
            try:
                # Try loading with device_map first
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    trust_remote_code=True
                )
            except NotImplementedError as e:
                # Handle meta tensor issue
                if "Cannot copy out of meta tensor" in str(e):
                    print(f"    🔧 Meta tensor issue detected for {self.model_name}, using to_empty()...")
                    # Load model without device_map first, then use to_empty()
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        device_map=None,  # Don't use device_map initially
                        trust_remote_code=True
                    )
                    # Use to_empty() to properly move from meta to device
                    if self.device_map is not None:
                        # If device_map was specified, use to_empty() to move to the target device
                        if isinstance(self.device_map, dict) and "" in self.device_map:
                            target_device = self.device_map[""]
                            self.model = self.model.to_empty(device=target_device)
                        elif isinstance(self.device_map, str) and self.device_map == "auto":
                            # For auto device mapping, use to_empty() with the default device
                            self.model = self.model.to_empty(device=self.device)
                    else:
                        # No device_map specified, use to_empty() to move to default device
                        self.model = self.model.to_empty(device=self.device)
                else:
                    raise e
            
            # Ensure model is on the correct device
            if not hasattr(self.model, 'hf_device_map') or self.model.hf_device_map is None:
                # Model is not device-mapped, ensure it's on the correct device
                if hasattr(self.model, 'device'):
                    current_device = str(self.model.device)
                    if current_device == 'cpu' and torch.cuda.is_available():
                        self.model = self.model.to(self.device)
                else:
                    # Model doesn't have device attribute, move it
                    self.model = self.model.to(self.device)
            
            self.model.eval()

    def _unload_model(self):
        """Unload model and tokenizer to free memory."""
        if self.model is not None:
            del self.model, self.tokenizer, self.config
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.config = None

    @staticmethod
    def _preprocess(text: str) -> str:
        """Normalize mentions and URLs for Twitter text."""
        tokens = []
        for t in text.split():
            if t.startswith('@') and len(t) > 1:
                tokens.append('@user')
            elif t.startswith('http'):
                tokens.append('http')
            else:
                tokens.append(t)
        return " ".join(tokens)

    def _calculate_impl(self, input: str, output: str, references=None, **kwargs) -> float:
        """Compute sentiment regression for a single text."""
        if self.model is None:
            self._load_model()
        
        # Ensure model is on the correct device
        if hasattr(self.model, 'device'):
            model_device = self.model.device
        else:
            # Fallback: assume model is on the same device as the first parameter
            model_device = next(self.model.parameters()).device
        
        text = self._preprocess(output)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512
        ).to(model_device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)
            probs = F.softmax(logits, dim=-1)
        neg = probs[0].item()
        pos = probs[2].item()
        score = (pos - neg) / 2.0
        
        if not self.persistent:
            self._unload_model()
        return score

    def _calculate_batched_impl(self, inputs: List[str], outputs: List[str], references=None, **kwargs) -> List[float]:
        """Compute sentiment regression for batches of texts."""
        if self.model is None:
            self._load_model()
        
        # Ensure model is on the correct device
        if hasattr(self.model, 'device'):
            model_device = self.model.device
        else:
            # Fallback: assume model is on the same device as the first parameter
            model_device = next(self.model.parameters()).device
        
        all_scores: List[float] = []
        for i in range(0, len(outputs), self.batch_size):
            batch_texts = outputs[i:i+self.batch_size]
            batch_proc = [self._preprocess(text) for text in batch_texts]
            inputs_tok = self.tokenizer(
                batch_proc,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            ).to(model_device)
            
            with torch.no_grad():
                logits = self.model(**inputs_tok).logits
                probs = F.softmax(logits, dim=-1)
            batch_scores = ((probs[:, 2] - probs[:, 0]) / 2.0).cpu().tolist()
            all_scores.extend(batch_scores)
        
        if not self.persistent:
            self._unload_model()
        return all_scores 