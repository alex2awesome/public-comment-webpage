# autometrics/metrics/reference_free/LDLReward27B.py

import os
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Dict, Any, ClassVar
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Gemma2PreTrainedModel, Gemma2Model, Gemma2Config,
    AutoTokenizer
)
from transformers.utils import ModelOutput
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from huggingface_hub import snapshot_download
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
from autometrics.metrics.utils.device_utils import get_model_device, ensure_tensor_on_device


class MultiOutputNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [4096, 4096]):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.LeakyReLU()]
        for h0, h1 in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.extend([nn.Linear(h0, h1), nn.LeakyReLU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        # reshape to (batch, num_distributions, dist_size)
        return self.softmax(out.view(x.size(0), -1, 10))


class GatingNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 4096,
        num_layers: int = 2,
        temperature: float = 1.0,
        dropout_prob: float = 0.0,
        softmax: bool = False
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = softmax
        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.softmax:
            out = F.softmax(out / self.temperature, dim=1)
        return out


@dataclass
class CustomOutput(ModelOutput):
    rewards: torch.FloatTensor
    hidden_state: Optional[Tuple[torch.FloatTensor, ...]] = None
    score: Optional[torch.FloatTensor] = None
    total_reward_distribution: Optional[torch.FloatTensor] = None
    weights: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class LDLRewardModel27B(Gemma2PreTrainedModel):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        cfg = config.to_dict()
        self.num_objectives = cfg.get("num_objectives", 220)
        self.regression_layer = MultiOutputNN(config.hidden_size, self.num_objectives)
        self.gating_layer = GatingNN(
            config.hidden_size,
            self.num_objectives // 10,
            temperature=cfg.get("temperature", 1.0),
            softmax=cfg.get("softmax", False)
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> CustomOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        token_states = transformer_outputs.last_hidden_state
        # find last non-pad token index
        if input_ids is not None:
            batch_size = input_ids.size(0)
            if self.config.pad_token_id is not None:
                seq_lens = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                seq_lens = seq_lens % input_ids.size(-1)
                seq_lens = seq_lens.to(token_states.device)
            else:
                if batch_size != 1:
                    raise ValueError("Batch>1 requires pad_token_id")
                seq_lens = -1
        else:
            batch_size = inputs_embeds.size(0)
            seq_lens = -1
        idx = torch.arange(batch_size, device=token_states.device)
        hidden = token_states[idx, seq_lens]
        # compute distributions and score
        # Ensure aux layers live on same device as hidden
        try:
            if next(self.regression_layer.parameters()).device != hidden.device:
                self.regression_layer = self.regression_layer.to(hidden.device)
            if next(self.gating_layer.parameters()).device != hidden.device:
                self.gating_layer = self.gating_layer.to(hidden.device)
        except StopIteration:
            pass
        with torch.autocast(device_type=hidden.device.type, dtype=torch.float32):
            rewards = self.regression_layer(hidden)
            weights = self.gating_layer(hidden).unsqueeze(1)
            total_dist = torch.bmm(weights, rewards).squeeze(1)
            score = (total_dist * torch.linspace(0, 1, total_dist.size(-1), device=hidden.device)).sum(-1)
        return CustomOutput(
            rewards=rewards,
            weights=weights,
            hidden_state=hidden,
            total_reward_distribution=total_dist,
            score=score,
            logits=score
        )

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory, dtype=torch.bfloat16)
        torch.save(self.regression_layer.state_dict(), os.path.join(save_directory, "regression_layer.pt"))
        torch.save(self.gating_layer.state_dict(), os.path.join(save_directory, "gating_layer.pt"))
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        device_map: Union[str, dict] = None,
        *model_args,
        **kwargs
    ):
        # Ensure a consistent dtype is used across the model to avoid matmul dtype mismatches
        if 'torch_dtype' not in kwargs or kwargs['torch_dtype'] is None:
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    kwargs['torch_dtype'] = torch.bfloat16
                else:
                    kwargs['torch_dtype'] = torch.float16
            else:
                kwargs['torch_dtype'] = torch.float32
        if not os.path.exists(load_directory):
            cached_dir = snapshot_download(repo_id=load_directory)
        else:
            cached_dir = load_directory

        # 1) Load the model on CPU first. Do NOT pass device_map here to avoid meta tensors.
        base_kwargs = dict(kwargs)
        if 'device_map' in base_kwargs:
            base_kwargs.pop('device_map')
        model = super().from_pretrained(cached_dir, *model_args, **base_kwargs)

        # 2) Load additional layers with real weights on CPU first
        reg_path = os.path.join(cached_dir, "regression_layer.pt")
        gate_path = os.path.join(cached_dir, "gating_layer.pt")
        # Materialize custom heads if they were created on meta device
        try:
            if any(p.is_meta for p in model.regression_layer.parameters()):
                model.regression_layer = model.regression_layer.to_empty(device='cpu')
            if any(p.is_meta for p in model.gating_layer.parameters()):
                model.gating_layer = model.gating_layer.to_empty(device='cpu')
        except Exception:
            # Fallback: move modules to CPU; if meta, this will be a no-op but load_state_dict with assign may still work on newer torch
            try:
                model.regression_layer = model.regression_layer.to('cpu')
                model.gating_layer = model.gating_layer.to('cpu')
            except Exception:
                pass
        # Now load actual weights for custom heads
        sd_reg = torch.load(reg_path, map_location='cpu')
        sd_gate = torch.load(gate_path, map_location='cpu')
        try:
            model.regression_layer.load_state_dict(sd_reg)
            model.gating_layer.load_state_dict(sd_gate)
        except Exception:
            # Try non-strict as a last resort
            model.regression_layer.load_state_dict(sd_reg, strict=False)
            model.gating_layer.load_state_dict(sd_gate, strict=False)

        # 3) If a device_map is requested, dispatch after all weights are materialized
        if device_map is not None:
            if not hasattr(model, 'hf_device_map') or model.hf_device_map is None:
                if isinstance(device_map, (str,)) and device_map in ("auto", "balanced"):
                    max_mem = get_balanced_memory(model, no_split_module_classes=["Gemma2DecoderLayer", "Gemma2RMSNorm"])
                    dm = infer_auto_device_map(
                        model,
                        no_split_module_classes=["Gemma2DecoderLayer", "Gemma2RMSNorm"],
                        max_memory=max_mem
                    )
                    model = dispatch_model(model, device_map=dm)
                elif isinstance(device_map, dict):
                    model = dispatch_model(model, device_map=device_map)
                else:
                    # Handle explicit single-device strings like "cuda:0"
                    try:
                        model = model.to(device_map)
                    except Exception:
                        pass
            else:
                print(f"[LDLRewardModel] Model already device-mapped: {model.hf_device_map}")

        return model


class LDLRewardModel(ReferenceFreeMetric):
    """---
# Metric Card for LDL Reward Model 27B

LDL Reward Model 27B is a large-scale, reference-free reward model designed for evaluating generative model outputs across multiple sub-dimensions such as helpfulness, correctness, coherence, and others. It is built on top of the **Gemma 2-27B** transformer model, with additional modules for reward prediction using **Label Distribution Learning (LDL)**. LDL models human rating uncertainty as probability distributions rather than single point estimates, improving robustness to subjective human annotations. The model outputs a final scalar reward score that can be used for ranking or fine-tuning generative models.

## Metric Details

### Metric Description

LDL Reward Model 27B predicts sub-score distributions for various dimensions of quality (e.g., helpfulness, correctness, complexity) by applying Label Distribution Learning (LDL). In the first stage, the model learns to predict label distributions using a regression layer based on the last hidden token representation from Gemma 2. In the second stage, a gating mechanism aggregates these sub-score distributions into a final scalar reward, trained via Bradley-Terry modeling on pairwise preference data. This approach allows the model to capture uncertainty in human feedback while providing a single, interpretable reward score for evaluation or training.

- **Metric Type:** Reference-Free
- **Range:** Needs more information (example outputs observed between approximately -3.5 and 7.0)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

First-stage sub-score prediction (for each sub-score $s$):

Given a hidden state vector $h$, the regression layer predicts a label distribution $D_s$ modeled as a discrete Gaussian over possible scores:

$$
D_s(h) = \text{Regression}(h)
$$

Second-stage score aggregation:

A gating layer computes weights $w_s$ over the sub-scores:

$$
w = \text{Gating}(h)
$$

The final reward distribution $D_{\text{final}}$ is computed as a weighted combination:

$$
D_{\text{final}} = \sum _{s} w_s \cdot D_s
$$

The final scalar reward $r$ is then obtained by taking the expectation over $D_{\text{final}}$:

$$
r = \sum _{i} p_i \cdot v_i
$$

where $p_i$ are probabilities from $D_{\text{final}}$ and $v_i$ are the associated score values (typically discretized between 0 and 1).

Pairwise comparison is trained with a Bradley-Terry model:

$$
\mathbb{P}_{\theta}(a_1 > a_2 | x, a_1, a_2) = \sigma(r_1 - r_2)
$$

where $\sigma$ is the sigmoid function, and $r_1$, $r_2$ are the final reward scores of answers $a_1$, $a_2$ respectively.

### Inputs and Outputs

- **Inputs:**  
  - Prompt and generated response (formatted as conversation history).
  
- **Outputs:**  
  - Scalar reward score (real-valued).

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Code Generation, Dialogue Systems
- **Tasks:** Response Generation, Dialogue Generation, Code Completion

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating and fine-tuning models for response quality in dialogue, general language generation, or code generation settings where human preference or judgment is a relevant evaluation signal.

- **Not Recommended For:**  
  Tasks requiring strict factual verification or tasks outside general conversational and code-related domains. Also not recommended when highly domain-specific expertise (e.g., medical, legal) is required without further adaptation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [Hugging Face Model Page](https://huggingface.co/ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1)

### Computational Complexity

- **Efficiency:**  
  Inference involves a forward pass through a large model (27B parameters) and multiple small neural network layers. Efficiency is similar to other large reward models; inference is relatively fast for single comparisons but costly for large-scale evaluations.

- **Scalability:**  
  Scalability is dependent on hardware; intended for GPU acceleration, with support for device mapping and memory optimization (e.g., using FlashAttention-2).

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  Potential risk if deployed directly to domains very different from the training data distribution (e.g., highly technical, legal, or medical content).

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- Skywork Reward Models (Skywork-Reward-Gemma-2-27B)  
- INF-ORM Llama 3.1-70B (similar architecture using label distribution learning)

## Further Reading

- **Papers:**  
  - Tech Report (Coming Soon)

- **Blogs/Tutorials:**  
  - [Hugging Face Model Page](https://huggingface.co/ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1)

## Citation

```
@misc{chen2025ldl,
  author       = {Shikai Chen and Jin Yuan and Yang Zhang and Zhongchao Shi and Jianping Fan and Xin Geng and Yong Rui},
  title        = {LDL-Reward-Gemma-2-27B-v0.1},
  year         = {2025},
  howpublished = {https://huggingface.co/ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1},
  note         = {Hugging Face Model Repository},
}
```
  
## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and relevant documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 104171.59765625 # in MB
    cpu_mem: ClassVar[float] = 2057.19140625 # in MB
    description: ClassVar[str] = "LDL Reward Model 27B is a large-scale, reference-free reward model designed for evaluating generative model outputs across multiple sub-dimensions such as helpfulness, correctness, coherence, and others. It is built on top of the **Gemma 2-27B** transformer model, with additional modules for reward prediction using **Label Distribution Learning (LDL)**. LDL models human rating uncertainty as probability distributions rather than single point estimates, improving robustness to subjective human annotations. The model outputs a final scalar reward score that can be used for ranking or fine-tuning generative models."

    def __init__(
        self,
        name: str = "LDLReward27B",
        description: str = "LDL Reward Model 27B is a large-scale, reference-free reward model designed for evaluating generative model outputs across multiple sub-dimensions such as helpfulness, correctness, coherence, and others. It is built on top of the **Gemma 2-27B** transformer model, with additional modules for reward prediction using **Label Distribution Learning (LDL)**. LDL models human rating uncertainty as probability distributions rather than single point estimates, improving robustness to subjective human annotations. The model outputs a final scalar reward score that can be used for ranking or fine-tuning generative models.",
        model_name: str = "ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1",
        device_map: Union[str, dict] = "auto",
        batch_size: int = 2,
        persistent: bool = True,
        **kwargs
    ):
        super().__init__(name, description, model_name=model_name, device_map=device_map, batch_size=batch_size, persistent=persistent, **kwargs)
        self.model_name = model_name
        self.device_map = device_map
        self.batch_size = batch_size
        self.persistent = persistent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LDLRewardModel27B] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.exclude_from_cache_key('model_name', 'device_map', 'batch_size', 'persistent')

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Ensure a pad token is set for consistent batching
            if self.tokenizer.pad_token is None:
                # Fall back to EOS token as pad if the tokenizer has no explicit pad token
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = LDLRewardModel27B.from_pretrained(
                self.model_name,
                device_map=self.device_map
            )
            self.model.eval()
            # Ensure the loaded model shares the same pad token ID as the tokenizer
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None

    # Helper to handle both tensor and dict inputs
    def _call_model(self, tok: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        """Forward `tok` through the model regardless of its exact structure."""
        with torch.no_grad():
            if isinstance(tok, torch.Tensor):
                outputs = self.model(input_ids=tok)
            else:
                outputs = self.model(**tok)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            return logits

    def _calculate_impl(self, input: str, output: str, references=None, **kwargs) -> float:
        """Score a single input–output pair."""
        if self.model is None:
            self._load_model()

        # Prepare conversation
        conv = [{"role": "user", "content": input}, {"role": "assistant", "content": output}]

        # Ensure tensors are placed on the same device as the model
        model_device = get_model_device(self.model, fallback_device=self.device)
        # Tokenize with automatic padding/truncation to avoid tensor shape mismatches
        tok = self.tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        tok = ensure_tensor_on_device(tok, model_device)

        logits = self._call_model(tok)
        score = logits.squeeze().cpu().item()

        if not self.persistent:
            self._unload_model()
        return score

    def _calculate_batched_impl(self, inputs: List[str], outputs: List[str], references=None, **kwargs) -> List[float]:
        """Score batches of input–output pairs."""
        if self.model is None:
            self._load_model()

        model_device = get_model_device(self.model, fallback_device=self.device)
        all_scores: List[float] = []

        for i in range(0, len(inputs), self.batch_size):
            chunk_in = inputs[i : i + self.batch_size]
            chunk_out = outputs[i : i + self.batch_size]

            # Prepare batch conversations
            convs = [[{"role": "user", "content": inp}, {"role": "assistant", "content": out}] for inp, out in zip(chunk_in, chunk_out)]

            # Tokenize with automatic padding/truncation to ensure consistent tensor shapes across the batch
            tok = self.tokenizer.apply_chat_template(
                convs,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
            tok = ensure_tensor_on_device(tok, model_device)

            logits = self._call_model(tok)
            vals = logits.squeeze(-1).cpu().tolist()
            if isinstance(vals, float):
                all_scores.append(vals)
            else:
                all_scores.extend(vals)

        if not self.persistent:
            self._unload_model()
        return all_scores
