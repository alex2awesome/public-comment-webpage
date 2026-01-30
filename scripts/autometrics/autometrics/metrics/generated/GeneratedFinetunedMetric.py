import json
import os
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd

import dspy

from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code
from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric

__all__ = ["GeneratedRefFreeFinetunedMetric", "GeneratedRefBasedFinetunedMetric"]


# Base mixin for shared finetuned functionality
class _FinetunedMetricMixin:
    """Shared functionality for both reference-free and reference-based fine-tuned metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        model_path: str,
        task_description: Optional[str] = None,
        target_measure: Optional[str] = None,
        dataset_name: Optional[str] = None,
        training_stats: Optional[Dict[str, Any]] = None,
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_seq_length: int = 2048,
        batch_size: int = 8,  # Batch size for efficient inference
        is_reference_based: bool = False,
        **kwargs,
    ):
        self.model_path = model_path
        self.task_description = task_description or "No task description provided"
        self.target_measure = target_measure or "quality"
        self.dataset_name = dataset_name or "unknown"
        self.training_stats = training_stats or {}
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.is_reference_based = is_reference_based

        # Model and tokenizer (loaded lazily)
        self._model = None
        self._tokenizer = None

        if metric_card_author_model is None:
            # For fine-tuned metrics, we don't have an LLM model to use as author
            # We'll generate the card programmatically or use a provided one
            metric_card_author_model = None

        if metric_card == "provided":
            self.metric_card = self.__doc__
            metric_card = self.metric_card

        # Initialize parent with shared parameters
        super().__init__(
            name,
            description,
            metric_card=metric_card,
            metric_card_author_model=metric_card_author_model,
            model_path=model_path,
            task_description=self.task_description,
            target_measure=self.target_measure,
            dataset_name=self.dataset_name,
            training_stats=self.training_stats,
            max_seq_length=self.max_seq_length,
            **kwargs,
        )

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("_model", "_tokenizer", "batch_size")

    def _load_model_and_tokenizer(self):
        """Lazily load the fine-tuned model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                from peft import AutoPeftModelForSequenceClassification
                import json
                import os
                import torch
            except ImportError as e:
                raise ImportError(f"Required libraries not installed: {e}. Please install transformers and peft.")

            print(f"🤖 Loading fine-tuned model from: {self.model_path}")
            print(f"🤖 Model path exists: {os.path.exists(self.model_path)}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
            
            try:
                # Load tokenizer
                print(f"🤖 Loading tokenizer...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                print(f"✅ Tokenizer loaded successfully")
                
                # Check if this is a PEFT model by looking for adapter_config.json
                adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
                
                if os.path.exists(adapter_config_path):
                    print(f"🤖 Found adapter config, loading PEFT model...")
                    # Load PEFT model with device_map="auto" to handle meta tensors properly
                    self._model = AutoPeftModelForSequenceClassification.from_pretrained(
                        self.model_path,
                        num_labels=1,  # Regression
                        torch_dtype=torch.float32,  # Use float32 for stability
                        device_map="auto",  # Let transformers handle device placement
                        low_cpu_mem_usage=True,  # More efficient loading
                    )
                    print("✅ PEFT adapter model loaded successfully")
                else:
                    print(f"🤖 No adapter config found, loading standard model...")
                    # Fallback to standard model loading
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_path,
                        num_labels=1,  # Regression
                        torch_dtype=torch.float32,  # Use float32 for stability
                        device_map="auto",  # Let transformers handle device placement
                        low_cpu_mem_usage=True,  # More efficient loading
                    )
                    print("✅ Standard fine-tuned model loaded successfully")
                
                # Set to evaluation mode
                self._model.eval()
                
                # Print model info
                device = next(self._model.parameters()).device
                print(f"✅ Model loaded on device: {device}")
                print(f"✅ Model dtype: {next(self._model.parameters()).dtype}")
                
                # Don't manually move to CUDA since device_map="auto" handles placement
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                raise e

        return self._model, self._tokenizer

    def _predict_batch(self, texts: List[str]) -> List[float]:
        """Make predictions on a batch of texts using the fine-tuned model."""
        print("=" * 80)
        print(f"🚨 FINETUNED DEBUG: _predict_batch called with {len(texts)} texts")
        print("=" * 80)
        
        print(f"🔍 Processing batch of {len(texts)} texts")
        if texts:
            print(f"🔍 Sample text: {texts[0][:150]}...")
        
        try:
            print(f"🚨 FINETUNED DEBUG: About to call _load_model_and_tokenizer()")
            model, tokenizer = self._load_model_and_tokenizer()
            print(f"🚨 FINETUNED DEBUG: Model and tokenizer loaded successfully")
            print(f"🔍 Model and tokenizer loaded successfully")
            
            # Tokenize the entire batch
            print(f"🚨 FINETUNED DEBUG: Tokenizing {len(texts)} texts...")
            print(f"🔍 Tokenizing {len(texts)} texts...")
            
            import torch
            with torch.no_grad():
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length
                )
                print(f"🚨 FINETUNED DEBUG: Tokenization complete. Input shape: {inputs['input_ids'].shape}")
                print(f"🔍 Tokenization complete. Input shape: {inputs['input_ids'].shape}")
                
                # Move inputs to the same device as the model
                device = next(model.parameters()).device
                print(f"🚨 FINETUNED DEBUG: Moving inputs to device: {device}")
                print(f"🔍 Moving inputs to device: {device}")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get predictions for the entire batch
                print(f"🚨 FINETUNED DEBUG: Running model inference...")
                print(f"🔍 Running model inference...")
                outputs = model(**inputs)
                logits = outputs.logits.detach().cpu().numpy().flatten()
                print(f"🚨 FINETUNED DEBUG: Raw logits: {logits}")
                print(f"🔍 Raw logits: {logits}")
                print(f"🔍 Logits shape: {logits.shape}")
                print(f"🔍 Logits dtype: {logits.dtype}")
                
                # For regression, directly use the logit outputs
                predictions = [float(logit) for logit in logits]
                print(f"🚨 FINETUNED DEBUG: Final predictions: {predictions}")
                print(f"🎯 Final predictions: {predictions}")
                
        except Exception as e:
            print("=" * 80)
            print(f"🚨 CRITICAL ERROR in Fine-tuned Model:")
            print(f"   Error: {e}")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Model Path: {getattr(self, 'model_path', 'UNKNOWN')}")
            print("=" * 80)
            import traceback
            traceback.print_exc()
            # Return zeros if prediction fails
            predictions = [0.0] * len(texts)
            print(f"❌ Returning zeros due to error: {predictions}")
            
        return predictions

    def _format_text(self, input_text: str, output_text: str, references: Optional[str] = None) -> str:
        """Format text for prediction (consistent with training format)."""
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""
        
        if self.is_reference_based and references is not None:
            if isinstance(references, list):
                refs = " ".join([str(ref) for ref in references if ref is not None])
            else:
                refs = str(references)
            return f"Input: {input_text} Output: {output_text} Reference: {refs}"
        else:
            return f"Input: {input_text} Output: {output_text}"

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        
        print(f"📊 Fine-tuned inference on {len(inputs)} examples with batch_size={self.batch_size}")
        
        # Efficient batch processing for transformers model
        all_results = []
        references = references or [None] * len(outputs)
        
        # Format all texts first
        formatted_texts = [
            self._format_text(i, o, r) 
            for i, o, r in zip(inputs, outputs, references)
        ]
        
        # Process in batches
        num_batches = (len(formatted_texts) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=num_batches, desc=f"Fine-tuned Predictions (batch_size={self.batch_size})") as pbar:
            for batch_idx, batch_start in enumerate(range(0, len(formatted_texts), self.batch_size)):
                batch_end = min(batch_start + self.batch_size, len(formatted_texts))
                batch_texts = formatted_texts[batch_start:batch_end]
                
                # Get predictions for this batch
                batch_results = self._predict_batch(batch_texts)
                all_results.extend(batch_results)
                
                pbar.update(1)
        
        print(f"🎯 Final predictions: {all_results}")
        return all_results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        class_name = "GeneratedRefBasedFinetunedMetric" if self.is_reference_based else "GeneratedRefFreeFinetunedMetric"
        code = f"""# Auto-generated fine-tuned metric file for {self.name}
import os
from pathlib import Path
from autometrics.metrics.generated.GeneratedFinetunedMetric import {class_name}
from typing import ClassVar

# Note: This metric requires the fine-tuned model to be available at the specified path
MODEL_PATH = r"{self.model_path}"

class {self.name.replace(" ", "_").replace("-", "_")}_Finetuned({class_name}):
    \"\"\"{self.metric_card if include_metric_card else ""}\"\"\"

    description: ClassVar[str] = {json.dumps(self.description)}

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned model not found at {{model_path}}. Please ensure the model is available.")
        
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            model_path=model_path,
            task_description={json.dumps(self.task_description)},
            target_measure={json.dumps(self.target_measure)},
            dataset_name={json.dumps(self.dataset_name)},
            training_stats={json.dumps(self.training_stats)},
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_seq_length={self.max_seq_length},
            batch_size={self.batch_size},
        )

    def __repr__(self):
        return f"{self.name.replace(' ', '_').replace('-', '_')}_Finetuned(model_path='{self.model_path}')"

"""
        return code

    def _serialize(self) -> dict:
        """Serialize the metric to a dictionary for in-memory operations."""
        return {
            "name": self.name,
            "description": self.description,
            "model_path": self.model_path,
            "task_description": self.task_description,
            "target_measure": self.target_measure,
            "dataset_name": self.dataset_name,
            "training_stats": self.training_stats,
            "metric_card": self.metric_card,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "is_reference_based": self.is_reference_based,
        }

    @classmethod
    def _deserialize(cls, data: dict):
        """Deserialize a dictionary to create a metric instance."""
        return cls(**data)
    
    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based fine-tuned metrics."""
        kind = "reference-based" if reference_based else "reference-free"
        ref_flag = "Yes" if reference_based else "No"
        input_req = "Yes (plus reference)" if reference_based else "Yes"

        # Get training statistics
        train_size = self.training_stats.get("train_size", "Unknown")
        val_size = self.training_stats.get("val_size", "Unknown")
        target_mean = self.training_stats.get("target_mean", "Unknown")
        target_std = self.training_stats.get("target_std", "Unknown")
        epochs = self.training_stats.get("epochs", "Unknown")
        learning_rate = self.training_stats.get("learning_rate", "Unknown")

        lines = [
            f"**{self.name}** is a **{kind}** fine-tuned metric that uses a regression-trained ModernBERT model to predict quality scores.",
            f"The model was fine-tuned on the `{self.dataset_name}` dataset to predict `{self.target_measure}` values.",
            "",
            "The model was trained using supervised learning on input-output pairs with quality scores,",
            "learning to predict numerical quality ratings directly from text patterns.",
            "",
        ]

        # Add training data size warnings
        if isinstance(train_size, int):
            if train_size < 100:
                lines.extend([
                    "⚠️ **WARNING: Limited Training Data** ⚠️",
                    "",
                    f"This model was trained on only **{train_size} examples**, which is quite small for fine-tuning.",
                    "Performance may be limited due to insufficient training data. Consider:",
                    "- Gathering more training examples if possible",
                    "- Using this metric cautiously and validating against human judgments",
                    "- Combining with other metrics for more robust evaluation",
                    "",
                ])
            elif train_size < 250:
                lines.extend([
                    "⚠️ **Note: Moderate Training Data Size** ⚠️",
                    "",
                    f"This model was trained on **{train_size} examples**. While this can work,",
                    "more training data typically leads to better performance. Consider validating",
                    "against human judgments and using additional metrics when possible.",
                    "",
                ])

        lines.extend([
            "### Training Details",
            "",
            f"- **Base Model:** ModernBERT-Large (answerdotai/ModernBERT-large) using PEFT/LoRA",
            f"- **Training Dataset:** {self.dataset_name}",
            f"- **Target Measure:** {self.target_measure}",
            f"- **Training Examples:** {train_size}",
            f"- **Validation Examples:** {val_size}",
            f"- **Training Epochs:** {epochs}",
            f"- **Learning Rate:** {learning_rate}",
            f"- **Target Statistics:** Mean={target_mean:.3f}, Std={target_std:.3f}" if isinstance(target_mean, (int, float)) else f"- **Target Statistics:** {target_mean}",
            "",
            "### Input Format",
            "",
            "The model expects input in the format used during training:",
        ])
        
        if reference_based:
            lines.append("- `Input: [input_text] Output: [output_text] Reference: [reference_text]`")
        else:
            lines.append("- `Input: [input_text] Output: [output_text]`")

        lines.extend([
            "",
            "### Formal Definition",
            "",
            r"Let $f_{\theta}$ be the fine-tuned ModernBERT model with parameters $\theta$",
            r"learned through supervised regression training.",
        ])

        if reference_based:
            lines.append(
                r"The metric computes $\hat{s} = f_{\theta}(\text{Input: } x \text{ Output: } y \text{ Reference: } r)$"
            )
        else:
            lines.append(
                r"The metric computes $\hat{s} = f_{\theta}(\text{Input: } x \text{ Output: } y)$"
            )

        lines.extend([
            "",
            r"where $\hat{s}$ is the predicted quality score in the same range as the training targets.",
            "",
            "- **Metric Type:** Fine-tuned Neural Regression",
            f"- **Range:** Continuous (similar to training target range)",
            "- **Higher is Better?:** Depends on training target",
            f"- **Reference-Based?:** {ref_flag}",
            f"- **Input-Required?:** {input_req}",
            "",
            "### Inputs and Outputs",
            "- **Inputs:**",
            "  - **Input text** *x*",
            "  - **Output text** *y*",
        ])
        
        if reference_based:
            lines.append("  - **Reference text** *r*")
        
        lines.extend([
            "- **Outputs:**",
            "  - Predicted quality score "
            r"$\hat{s} \in \mathbb{R}$ (continuous)",
        ])

        return "\n".join(lines)
    
    def generate_metric_details_ref_free(self) -> str:
        """Metric-details section for the **reference-free** variant."""
        return self._metric_details_template(reference_based=False)

    def generate_metric_details_ref_based(self) -> str:
        """Metric-details section for the **reference-based** variant."""
        return self._metric_details_template(reference_based=True)

    def generate_intended_use(self):
        """Generate intended use section for fine-tuned metrics."""
        return f"""- **Domain:** {self.dataset_name} Domain
- **Tasks:** Quality prediction tasks similar to the training domain
- **Best Suited For:** 
  - Datasets with similar characteristics to {self.dataset_name}
  - Tasks requiring {self.target_measure} assessment
  - Scenarios where training data is available for the specific domain
  - High-throughput evaluation scenarios
- **Not Recommended For:**
  - Datasets from significantly different domains
  - Tasks requiring different quality aspects than {self.target_measure}
  - Scenarios where model interpretability is critical
  - Very small datasets (model may overfit)"""

    def generate_metric_implementation(self):
        """Generate implementation details section."""
        ref_type = "reference-based" if self.is_reference_based else "reference-free"
        return f"""### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics Fine-tuned Metric ({ref_type})](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedFinetunedMetric.py)
  - [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large)

### Computational Complexity

- **Model Loading:**
  - One-time model loading overhead (a few seconds depending on hardware)
  - Model cached in memory after first load

- **Inference Efficiency:**
  - Single forward pass per input-output pair
  - Batch processing supported for efficiency
  - GPU acceleration available

- **Scalability:**
  - Linear scaling with number of examples
  - Efficient batching reduces per-example overhead
  - Memory requirements scale with batch size and sequence length

### Model Requirements

- **Model Path:** `{self.model_path}`
- **Dependencies:** transformers, torch
- **Hardware:** GPU recommended for faster inference
- **Memory:** ~16-24GB GPU memory for model loading"""

    def generate_known_limitations(self):
        """Generate known limitations section."""
        return f"""- **Domain Specificity:**
  - Model is specifically trained on {self.dataset_name} and may not generalize to other domains
  - Performance may degrade on inputs significantly different from training data
  
- **Target Measure Alignment:**
  - Model is optimized for {self.target_measure} and may not capture other quality aspects
  - Predictions may be biased toward patterns seen in the training data
  
- **Training Data Dependencies:**
  - Model quality depends heavily on the quality and representativeness of training data
  - May perpetuate biases present in the original dataset
  
- **Interpretability:**
  - Neural model predictions are not easily interpretable
  - Difficult to understand why specific scores were assigned
  
- **Computational Requirements:**
  - Requires model loading and GPU resources for optimal performance
  - Larger memory footprint compared to simpler metrics"""

    def generate_further_reading(self):
        """Generate further reading section."""
        return """- [ModernBERT: Modernizing BERT with Better Pre-training](https://arxiv.org/abs/2412.13663)
- [Fine-tuning Language Models for Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [BERT for Regression Tasks](https://arxiv.org/abs/1810.04805)"""

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class FinetunedMetricCardBuilder(MetricCardBuilder):
            def metric_details(self) -> str:
                if self.metric.is_reference_based:
                    return self.metric.generate_metric_details_ref_based()
                else:
                    return self.metric.generate_metric_details_ref_free()
            
            def intended_use(self) -> str:
                return self.metric.generate_intended_use()
            
            def metric_implementation(self) -> str:
                return self.metric.generate_metric_implementation()
            
            def known_limitations(self) -> str:
                return self.metric.generate_known_limitations()
            
            def further_reading(self) -> str:
                return self.metric.generate_further_reading()

        # For fine-tuned metrics, we build the card programmatically
        # since we don't have an LLM to generate it
        builder = FinetunedMetricCardBuilder(self)
        return builder.build()


class GeneratedRefFreeFinetunedMetric(_FinetunedMetricMixin, GeneratedRefFreeMetric):
    """Reference-free metric that uses a fine-tuned ModernBERT model for quality prediction.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    model_path      Path to the fine-tuned model directory
    task_description Optional task context
    target_measure  The target measure this model was trained to predict
    dataset_name    Name of the dataset used for training
    training_stats  Dictionary of training statistics
    metric_card_author_model  LLM used to generate the metric-card (optional)
    max_workers     Number of worker threads for batch processing
    max_seq_length  Maximum sequence length for model input
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = False
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._predict_single(input, output)


class GeneratedRefBasedFinetunedMetric(_FinetunedMetricMixin, GeneratedRefBasedMetric):
    """Reference-based metric that uses a fine-tuned ModernBERT model for quality prediction using reference text.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    model_path      Path to the fine-tuned model directory
    task_description Optional task context
    target_measure  The target measure this model was trained to predict
    dataset_name    Name of the dataset used for training
    training_stats  Dictionary of training statistics
    metric_card_author_model  LLM used to generate the metric-card (optional)
    max_workers     Number of worker threads for batch processing
    max_seq_length  Maximum sequence length for model input
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = True
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._predict_single(input, output, references) 