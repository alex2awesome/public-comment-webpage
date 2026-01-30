from datetime import datetime
from typing import Optional
import dspy
import re
import os

__all__ = ["build_llm_judge_metric_card"]

def generate_llm_constructor_code(model: dspy.LM) -> str:
    model_name = str(getattr(model, "model", model))
    kwargs = {}
    
    # Extract all kwargs if present
    if hasattr(model, "kwargs"):
        kwargs = {k: v for k, v in model.kwargs.items() if v is not None}
        # Minimal, surgical fallback: if model_name is missing/None, prefer kwargs['model'] when available
        try:
            if (not model_name or model_name.lower() == 'none') and isinstance(model.kwargs, dict):
                candidate = model.kwargs.get('model')
                if isinstance(candidate, str) and candidate:
                    model_name = candidate
        except Exception:
            pass
    
    """Generate constructor code for DSPy LLMs"""
    if "openai" in model_name.lower():
        kwargs["api_key"] = "os.getenv(\"OPENAI_API_KEY\")"
    elif "anthropic" in model_name.lower():
        kwargs["api_key"] = "os.getenv(\"ANTHROPIC_API_KEY\")"
    elif "gemini" in model_name.lower():
        kwargs["api_key"] = "os.getenv(\"GEMINI_API_KEY\")"
    else:
        kwargs["api_key"] = "None"
        
    kwargs_str = ", ".join(f"{k}={v}" if type(v) != str or k == "api_key" else f"{k}='{v}'" for k, v in kwargs.items())
    return f"dspy.LM(model=\'{model_name}\', {kwargs_str})"

def smart_truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    Intelligently truncate text while avoiding breaking markdown links.
    
    If the truncation point would fall in the middle of a markdown link,
    extend the truncation to complete the link or truncate before it starts.
    
    Args:
        text: The text to truncate
        max_length: Maximum allowed length
        ellipsis: String to append when truncating
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    
    # Basic truncation point (accounting for ellipsis)
    truncate_at = max_length - len(ellipsis)
    
    # Define markdown link patterns
    # Pattern 1: [text](url)
    # Pattern 2: [text][ref]
    # Pattern 3: <url>
    # Pattern 4: ![alt](url) for images
    link_patterns = [
        r'\[([^\]]*)\]\(([^)]*)\)',  # [text](url)
        r'\[([^\]]*)\]\[([^\]]*)\]',  # [text][ref]
        r'<([^>]+)>',                # <url>
        r'!\[([^\]]*)\]\(([^)]*)\)'  # ![alt](url)
    ]
    
    # Find all markdown links in the text
    links = []
    for pattern in link_patterns:
        for match in re.finditer(pattern, text):
            links.append((match.start(), match.end()))
    
    # Check if truncation point falls within any link
    for link_start, link_end in links:
        if link_start <= truncate_at < link_end:
            # We're in the middle of a link
            # Be more generous with completing links - allow extending significantly for small links
            # Allow at least 100% more characters (double the limit) or 100 extra chars, whichever is larger
            min_extension = max(max_length, 100)  # At least double the limit or 100 chars
            extended_limit = max_length + min_extension
            
            if link_end <= extended_limit:
                # Complete the link and then truncate
                truncate_at = link_end
                break
            else:
                # Option 2: Truncate before the link starts
                truncate_at = max(0, link_start)
                break
    
    # Perform the truncation
    if truncate_at < len(text):
        return text[:truncate_at] + ellipsis
    else:
        return text