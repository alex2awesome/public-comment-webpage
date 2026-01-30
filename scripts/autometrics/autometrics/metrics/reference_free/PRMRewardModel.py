import nltk
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from autometrics.metrics.reference_free.ReferenceFreeMultiMetric import ReferenceFreeMultiMetric
from autometrics.metrics.utils.device_utils import get_model_device, ensure_tensor_on_device
from typing import Tuple, List, ClassVar

class MathProcessRewardModel(ReferenceFreeMultiMetric):
    """---
# Metric Card for MathProcessRewardModel (Qwen2.5-Math-PRM-7B)

MathProcessRewardModel is a process-level reward model that evaluates each intermediate step in a multi-step mathematical reasoning problem. Rather than scoring the final answer alone, it provides token-level feedback across a reasoning chain, identifying helpful versus unhelpful steps using a learned binary classifier. This allows for granular supervision of multi-hop reasoning in LLMs and is particularly effective in domains where correctness must be verified incrementally.

## Metric Details

### Metric Description

MathProcessRewardModel evaluates step-by-step mathematical reasoning by assigning a reward score to each reasoning step in a sequence. The model inserts a special token (`<extra_0>`) after each reasoning step and computes the probability that the token is classified as "positive" using a softmax over logits. This yields a scalar between 0 and 1 indicating how helpful or correct the step is deemed to be. The model is trained on labels derived from whether a step leads to a correct solution trajectory, allowing it to generalize to unseen reasoning processes.

- **Metric Type:** Semantic Similarity, Reference-Free, Faithfulness
- **Range:** [0, 1]
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Let $x$ be a problem prompt, and $z_1, z_2, \dots, z_T$ be a sequence of reasoning steps. Let $<\!extra_0\!>$ be a separator token inserted after each step. Let $s_i$ denote the model's score for step $z_i$.

For each step:
$$
s_i = P(\text{label} = \text{positive} \mid z_i)
$$

This is computed via softmax over the model's logits:
$$
s_i = \text{softmax}(l_i)[\text{positive\_class}]
$$

where $l_i$ are the logits at the token position corresponding to $<\!extra_0\!>$ following step $z_i$.

### Inputs and Outputs

- **Inputs:**  
  - Problem prompt (e.g., math word problem)  
  - Step-by-step reasoning sequence (as multiple text spans, each ending with `<extra_0>`)  

- **Outputs:**  
  - List of step-level scores (floats between 0 and 1), one for each reasoning step

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Mathematical Reasoning, Step-by-Step Problem Solving, Chain-of-Thought Evaluation

### Applicability and Limitations

- **Best Suited For:**  
  Evaluation of mathematical reasoning chains, especially in contexts where intermediate steps need supervision (e.g., tutoring systems, math QA).

- **Not Recommended For:**  
  Open-ended creative generation tasks, final-answer-only assessments, or settings where reasoning steps are implicit or uninterpretable.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - Hugging Face Transformers ([Model Page](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B))
  - Source example and logic: see [Qwen2.5 PRM README](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)

### Computational Complexity

- **Efficiency:**  
  Inference is linear in the number of steps and token length; a forward pass over the model is required per input with all reasoning steps and inserted tokens.

- **Scalability:**  
  Scales well for moderate-length reasoning chains (~4–20 steps), but cost grows with step count due to softmax computation per `<extra_0>` token.

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  May be less applicable to domains where step-wise correctness is hard to define or subjective.

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- **Outcome Reward Models (ORM):** Only evaluate final answer correctness.
- **BERTScore:** Evaluates semantic similarity but not process reasoning.
- **Verifier Models:** Sometimes used to verify entire chains rather than local steps.

## Further Reading

- **Papers:**  
  - [Zhang et al. (2025) - The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)

- **Blogs/Tutorials:**  
  - [Stephen Diehl – Process Reward Models](https://www.stephendiehl.com/posts/process-reward-models.html)

## Citation

```
@misc{zhang2025lessonsdevelopingprocessreward,
      title={The Lessons of Developing Process Reward Models in Mathematical Reasoning}, 
      author={Zhenru Zhang and Chujie Zheng and Yangzhen Wu and Beichen Zhang and Runji Lin and Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin},
      year={2025},
      eprint={2501.07301},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.07301}, 
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    
    # Resource usage statistics (in megabytes)
    # TODO: Check this, because gpu memory being zero is suspicious
    gpu_mem: ClassVar[float] = 130000 # 13970.72265625 * 3  # in MB # THIS NUMBER IS AN ESTIMATE BASED ON HOW MUCH MEMORY THIS MODEL IS TAKING UP AS I RUN.  I SHOULD STILL RERUN THE BENCHMARKING SCRIPT
    cpu_mem: ClassVar[float] = 13970.72265625  # in MB 
    description: ClassVar[str] = "MathProcessRewardModel is a process-level reward model that evaluates each intermediate step in a multi-step mathematical reasoning problem. Rather than scoring the final answer alone, it provides token-level feedback across a reasoning chain, identifying helpful versus unhelpful steps using a learned binary classifier. This allows for granular supervision of multi-hop reasoning in LLMs and is particularly effective in domains where correctness must be verified incrementally."

    def __init__(
        self,
        name: str = "PRMRewardModel",
        description: str = "MathProcessRewardModel is a process-level reward model that evaluates each intermediate step in a multi-step mathematical reasoning problem. Rather than scoring the final answer alone, it provides token-level feedback across a reasoning chain, identifying helpful versus unhelpful steps using a learned binary classifier. This allows for granular supervision of multi-hop reasoning in LLMs and is particularly effective in domains where correctness must be verified incrementally.",
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device_map=None,
        persistent: bool = True,
        **kwargs
    ):
        super().__init__(name, description, submetric_names=["PRM_min", "PRM_max", "PRM_mean"], model_name=model_name, device_map=device_map, persistent=persistent, **kwargs)
        self.model_name = model_name
        self.device_map = device_map
        self.persistent = persistent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        self.exclude_from_cache_key('device_map', 'persistent')
        # Classes that must not be split across devices when sharding
        self.no_split_module_classes = [
            "Qwen2DecoderLayer", "Qwen2RMSNorm", "Qwen2SdpaAttention", "Qwen2MLP"
        ]

    def _load_model(self):
        if self.model is None:
            # Download sentence tokenizer if needed
            nltk.download('punkt', quiet=True)
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            # CPU-first model load to avoid meta tensors
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            # Apply device mapping after weights are materialized
            if self.device_map is not None:
                try:
                    if isinstance(self.device_map, (str,)) and self.device_map in ("auto", "balanced"):
                        try:
                            max_mem = get_balanced_memory(self.model, no_split_module_classes=self.no_split_module_classes)
                            dm = infer_auto_device_map(
                                self.model,
                                max_memory=max_mem,
                                no_split_module_classes=self.no_split_module_classes,
                            )
                        except Exception:
                            dm = infer_auto_device_map(
                                self.model,
                                no_split_module_classes=self.no_split_module_classes,
                            )
                        self.model = dispatch_model(self.model, device_map=dm)
                    elif isinstance(self.device_map, dict):
                        # Ensure caller-provided map respects no-split boundaries
                        self.model = dispatch_model(self.model, device_map=self.device_map)
                    else:
                        # explicit device string like 'cuda:0'
                        self.model = self.model.to(self.device_map)
                except Exception:
                    # Fallback: move to default device
                    self.model = self.model.to(self.device)
            else:
                # No mapping requested: move entire model to preferred device
                self.model = self.model.to(self.device)
            # Disable cache to avoid cross-device PKV issues in sharded forward
            try:
                if hasattr(self.model, "config") and getattr(self.model.config, "use_cache", None) is not None:
                    self.model.config.use_cache = False
            except Exception:
                pass
            self.model = self.model.eval()

            # Store the model's dtype for input tensor compatibility
            if hasattr(self.model, 'dtype'):
                self.model_dtype = self.model.dtype
            else:
                try:
                    first_param = next(self.model.parameters())
                    self.model_dtype = first_param.dtype
                except Exception:
                    self.model_dtype = torch.bfloat16

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None

    def _redispatch_sharded_model(self):
        """Recompute and apply a safe device map with no-split classes.
        Assumes the model is already loaded on CPU or a single device.
        """
        if self.model is None:
            self._load_model()
            return
        if self.device_map is None:
            return
        try:
            max_mem = None
            try:
                max_mem = get_balanced_memory(self.model, no_split_module_classes=self.no_split_module_classes)
            except Exception:
                pass
            if isinstance(self.device_map, dict):
                dm = self.device_map
            else:
                dm = infer_auto_device_map(
                    self.model,
                    max_memory=max_mem,
                    no_split_module_classes=self.no_split_module_classes,
                )
            self.model = dispatch_model(self.model, device_map=dm)
            try:
                if hasattr(self.model, "config") and getattr(self.model.config, "use_cache", None) is not None:
                    self.model.config.use_cache = False
            except Exception:
                pass
            self.model = self.model.eval()
        except Exception:
            # Best-effort redispatch; leave model as-is if this fails
            pass

    def _make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """Compute positive-class probability for each <extra_0> step in a single forward pass."""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
        all_scores_res: List[List[float]] = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores_res.append(positive_probs.cpu().tolist())
        return all_scores_res

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs) -> Tuple[float, float, float]:
        # Safeguard: ensure both input_text and output are strings to avoid type errors (e.g., float inputs)
        input_text = "" if input_text is None else str(input_text)
        output = "" if output is None else str(output)

        # Lazy load model if needed
        if self.model is None:
            self._load_model()
        # Ensure sentence tokenizer is available
        nltk.download('punkt', quiet=True)
        # Sentence-tokenize and append <extra_0> separators
        sentences = nltk.sent_tokenize(output)
        assistant_content = "<extra_0>".join(sentences) + "<extra_0>"
        # Build full conversation
        system_prompt = "Please reason step by step."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": assistant_content}
        ]
        conv_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Decide where to place inputs. If the model is sharded across multiple devices
        # or includes CPU offload, keep inputs on CPU and let accelerate handle routing.
        # If the model is fully on a single CUDA device, move inputs to that device to
        # avoid index-select device mismatches inside custom modules.
        model_device = get_model_device(self.model, fallback_device=self.device)
        hf_device_map = getattr(self.model, 'hf_device_map', None)
        keep_on_cpu = False
        exec_device = model_device
        if isinstance(hf_device_map, dict):
            unique_devices = set(hf_device_map.values())
            if len(unique_devices) == 1:
                only_device = next(iter(unique_devices))
                if isinstance(only_device, str) and only_device.startswith("cuda"):
                    exec_device = torch.device(only_device)
                    keep_on_cpu = False
                elif only_device == "cpu":
                    keep_on_cpu = True
                else:
                    keep_on_cpu = True
            else:
                # Mixed devices → keep inputs on CPU
                keep_on_cpu = True
        elif hf_device_map is not None:
            # Unknown map type → conservative default: keep on CPU
            keep_on_cpu = True

        # Tokenize
        input_ids = self.tokenizer.encode(conv_str, return_tensors="pt")
        if not keep_on_cpu:
            input_ids = ensure_tensor_on_device(input_ids, exec_device)
        # Attention mask on same device as input_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # Single forward pass (do NOT pass position_ids; let the model create them on the correct device)
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except RuntimeError as e:
            msg = str(e).lower()
            if ("same device" in msg or "different devices" in msg or "cuda:" in msg) and hasattr(self.model, 'hf_device_map') and getattr(self.model, 'hf_device_map') is not None:
                # Attempt to redispatch with stricter no-split and retry once
                self._redispatch_sharded_model()
                # Re-evaluate placement policy after redispatch
                model_device = get_model_device(self.model, fallback_device=self.device)
                hf_device_map = getattr(self.model, 'hf_device_map', None)
                keep_on_cpu = False
                exec_device = model_device
                if isinstance(hf_device_map, dict):
                    unique_devices = set(hf_device_map.values())
                    if len(unique_devices) == 1:
                        only_device = next(iter(unique_devices))
                        if isinstance(only_device, str) and only_device.startswith("cuda"):
                            exec_device = torch.device(only_device)
                            keep_on_cpu = False
                        elif only_device == "cpu":
                            keep_on_cpu = True
                        else:
                            keep_on_cpu = True
                    else:
                        keep_on_cpu = True
                elif hf_device_map is not None:
                    keep_on_cpu = True

                if not keep_on_cpu:
                    input_ids = ensure_tensor_on_device(input_ids, exec_device)
                # Rebuild attention_mask to match input device
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                raise
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        # Identify step separators and compute masks
        sep_id = self.tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == sep_id)
        # Compute per-step rewards
        step_rewards_list = self._make_step_rewards(logits, token_masks)[0]
        if not step_rewards_list:
            step_rewards_list = [0.0]
        # Aggregate rewards
        min_score = min(step_rewards_list)
        max_score = max(step_rewards_list)
        mean_score = sum(step_rewards_list) / len(step_rewards_list)
        # Optionally unload model
        if not self.persistent:
            self._unload_model()
        return (min_score, max_score, mean_score) 