import torch
from typing import Union, Optional, Any, Dict

_MODULE_TO_PATCHED: bool = False

def apply_meta_tensor_safe_module_to_patch() -> None:
    """
    Globally patch torch.nn.Module.to to gracefully handle meta tensor errors and
    avoid disrupting models that are already device-mapped via Accelerate
    (hf_device_map).

    Behavior:
    - If a Module has attribute `hf_device_map` (non-None), we no-op on `.to()`
      to prevent copying out of meta tensors. These models are already
      sharded/placed by Accelerate and should not be moved wholesale.
    - Otherwise, we call the original `.to()` and, upon encountering the common
      NotImplementedError "Cannot copy out of meta tensor", we fall back to
      `.to_empty()` targeting the requested device (if provided) or leave as-is
      if no device was specified.

    The patch is idempotent and safe to call multiple times.
    """
    global _MODULE_TO_PATCHED
    if _MODULE_TO_PATCHED:
        return

    import inspect
    import torch.nn.modules.module

    original_to = torch.nn.modules.module.Module.to

    def _extract_target_device(args: tuple, kwargs: Dict[str, Any]) -> Optional[torch.device]:
        # First positional arg may be a device or dtype; we only accept device-like
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                try:
                    return torch.device(first)
                except Exception:
                    pass
        # Keyword 'device' if present
        if 'device' in kwargs:
            try:
                return torch.device(kwargs['device'])
            except Exception:
                return None
        return None

    def patched_to(self, *args, **kwargs):  # type: ignore[override]
        # Respect Accelerate's device mapping – do not move such models
        try:
            if hasattr(self, 'hf_device_map') and getattr(self, 'hf_device_map') is not None:
                return self
        except Exception:
            # If attribute access fails, fall through to original behavior
            pass

        try:
            return original_to(self, *args, **kwargs)
        except NotImplementedError as e:
            if 'meta tensor' in str(e).lower() or 'Cannot copy out of meta tensor' in str(e):
                target = _extract_target_device(args, kwargs)
                print("    🔧 Meta tensor issue detected for Module.to(); using to_empty() fallback…")
                try:
                    if target is not None and hasattr(self, 'to_empty'):
                        return self.to_empty(device=target)
                    elif hasattr(self, 'to_empty'):
                        return self.to_empty()
                except Exception:
                    # If to_empty also fails, re-raise the original for transparency
                    pass
            raise e

    # Double-check we only patch Module.to (avoid patching tensors, etc.)
    if inspect.isfunction(original_to) or inspect.ismethod(original_to):
        torch.nn.modules.module.Module.to = patched_to  # type: ignore[assignment]
        _MODULE_TO_PATCHED = True

def apply_roberta_token_type_guard() -> None:
    """
    Patch RobertaEmbeddings.forward globally to guard against invalid
    token_type_ids when the embedding table has size 1 (RoBERTa default).

    Many third-party pipelines (e.g., LENS) may pass segment ids > 0, which
    triggers index errors. This guard clamps token_type_ids to the valid range
    and zeros them when type_vocab_size == 1.
    """
    try:
        from transformers.models.roberta.modeling_roberta import RobertaEmbeddings  # type: ignore
    except Exception:
        return  # transformers not available; nothing to patch

    import inspect
    if getattr(RobertaEmbeddings.forward, "__patched_token_type_guard__", False):
        return

    original_forward = RobertaEmbeddings.forward

    def guarded_forward(self, *args, **kwargs):  # type: ignore[override]
        # Match HF signature: (input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0)
        bound = None
        try:
            sig = inspect.signature(original_forward)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
        except Exception:
            # If signature binding fails, just call through
            return original_forward(self, *args, **kwargs)

        token_type_ids = bound.arguments.get("token_type_ids", None)
        input_ids = bound.arguments.get("input_ids", None)
        if token_type_ids is not None:
            try:
                vocab_sz = int(self.token_type_embeddings.num_embeddings)  # type: ignore[attr-defined]
                if vocab_sz <= 1:
                    # RoBERTa default: force zeros to avoid index errors
                    if input_ids is not None and isinstance(input_ids, torch.Tensor):
                        token_type_ids = torch.zeros_like(input_ids)
                    elif isinstance(token_type_ids, torch.Tensor):
                        token_type_ids = torch.zeros_like(token_type_ids)
                else:
                    if isinstance(token_type_ids, torch.Tensor):
                        token_type_ids = token_type_ids.clamp_(0, vocab_sz - 1)
                bound.arguments["token_type_ids"] = token_type_ids
            except Exception:
                pass

        return original_forward(*bound.args, **bound.kwargs)

    guarded_forward.__patched_token_type_guard__ = True  # type: ignore[attr-defined]
    RobertaEmbeddings.forward = guarded_forward  # type: ignore[assignment]

def get_model_device(model: Any, fallback_device: Optional[torch.device] = None) -> torch.device:
    """
    Determine the device a model is on by checking various attributes.
    
    Args:
        model: The model to check
        fallback_device: Fallback device if no device can be determined from the model
        
    Returns:
        torch.device: The device the model is on
    """
    # Check direct device property
    if hasattr(model, "device") and model.device is not None:
        return model.device
    
    # Check for get_device method and make sure it's callable
    if hasattr(model, "get_device") and callable(model.get_device):
        try:
            device = model.get_device()
            if device is not None:
                return device
        except Exception:
            pass  # Fall through to next method
    
    # Try to find a parameter's device
    try:
        for param in model.parameters():
            if hasattr(param, "device"):
                return param.device
    except (StopIteration, AttributeError, TypeError):
        pass  # Fall through to fallback
    
    # Fall back to provided device or CPU
    if fallback_device is not None:
        return fallback_device
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_tensor_on_device(tensor: Union[torch.Tensor, Dict, Any], 
                            device: torch.device) -> Union[torch.Tensor, Dict, Any]:
    """
    Ensure that a tensor or dictionary of tensors is on the specified device.
    If the input is not a tensor or dictionary, it is returned unchanged.
    
    Args:
        tensor: The tensor, dictionary of tensors, or other object
        device: The target device
        
    Returns:
        The tensor or dict of tensors on the specified device, or the unchanged input
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        # Handle dictionaries of tensors (common in HuggingFace models)
        return {k: ensure_tensor_on_device(v, device) for k, v in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        # Handle lists or tuples of tensors
        tensor_type = type(tensor)
        return tensor_type([ensure_tensor_on_device(t, device) for t in tensor])
    return tensor

def safe_model_to_device(model: torch.nn.Module, device: Union[str, torch.device], 
                        model_name: str = "model") -> torch.nn.Module:
    """
    Safely move a model to a device, handling meta tensor issues.
    
    This function attempts to move a model to the specified device using the standard
    .to() method, but falls back to .to_empty() if a meta tensor error occurs.
    
    Args:
        model: The PyTorch model to move
        device: The target device (string or torch.device)
        model_name: Name of the model for error messages
        
    Returns:
        The model moved to the target device
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    try:
        # Try standard .to() method first
        return model.to(device)
    except NotImplementedError as e:
        # Handle meta tensor issue
        if "Cannot copy out of meta tensor" in str(e):
            print(f"    🔧 Meta tensor issue detected for {model_name}, using to_empty()...")
            # Use to_empty() to properly move from meta to device
            return model.to_empty(device=device)
        else:
            # Re-raise if it's a different NotImplementedError
            raise e
    except Exception as e:
        # Handle any other device-related errors
        print(f"    ⚠ Device mapping issue for {model_name}: {e}")
        print(f"    🔧 This may be due to device mapping conflicts in parallel execution")
        raise e

def safe_model_loading(model_class, model_name_or_path: str, device: Union[str, torch.device] = None,
                      **kwargs) -> torch.nn.Module:
    """
    Safely load a model with proper device handling.
    
    This function loads a model and safely moves it to the specified device,
    handling meta tensor issues that commonly occur with device_map parameters.
    
    Args:
        model_class: The model class to instantiate (e.g., AutoModelForSequenceClassification)
        model_name_or_path: The model name or path to load
        device: The target device (if None, uses CUDA if available, otherwise CPU)
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        The loaded model on the target device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    try:
        # First attempt: try loading with provided kwargs
        model = model_class.from_pretrained(model_name_or_path, **kwargs)
        model.eval()
        
        # Check if model needs to be moved to device
        if not hasattr(model, 'hf_device_map') or model.hf_device_map is None:
            # Model is not device-mapped, move it to the target device
            model = safe_model_to_device(model, device, model_name_or_path)
        
        return model
        
    except NotImplementedError as e:
        # Handle meta tensor issue
        if "Cannot copy out of meta tensor" in str(e):
            print(f"    🔧 Meta tensor issue detected for {model_name_or_path}, using to_empty()...")
            
            # Remove device_map from kwargs if present to avoid conflicts
            load_kwargs = kwargs.copy()
            if 'device_map' in load_kwargs:
                del load_kwargs['device_map']
            
            # Load model without device_map first
            model = model_class.from_pretrained(model_name_or_path, **load_kwargs)
            model.eval()
            
            # Use to_empty() to properly move from meta to device
            model = model.to_empty(device=device)
            
            return model
        else:
            raise e
    except Exception as e:
        print(f"    ❌ Failed to load {model_name_or_path}: {e}")
        raise e 