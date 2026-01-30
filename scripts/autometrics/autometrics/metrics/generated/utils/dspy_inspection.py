"""
Utilities for inspecting DSPy optimized programs to extract prompts and examples.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import dspy
from .utils import smart_truncate_text


def load_dspy_program_from_path(prompt_path: str) -> Optional[Dict[str, Any]]:
    """Load DSPy program data from a JSON file path."""
    if not os.path.exists(prompt_path):
        return None
    
    try:
        with open(prompt_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not load DSPy program from {prompt_path}: {e}")
        return None


def extract_examples_from_dspy_data(dspy_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract examples/demos from DSPy program data."""
    examples = []
    
    try:
        # Look for examples in the predict.demos section
        if 'predict' in dspy_data and 'demos' in dspy_data['predict']:
            demos = dspy_data['predict']['demos']
            if isinstance(demos, list):
                examples.extend(demos)
    except Exception as e:
        print(f"Warning: Could not extract examples from DSPy data: {e}")
    
    return examples


def extract_signature_from_dspy_data(dspy_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract signature information from DSPy program data."""
    try:
        if 'predict' in dspy_data and 'signature' in dspy_data['predict']:
            return dspy_data['predict']['signature']
    except Exception as e:
        print(f"Warning: Could not extract signature from DSPy data: {e}")
    
    return None


def format_examples_as_markdown_table(examples: List[Dict[str, Any]], max_examples: int = 3) -> List[str]:
    """Format examples as a markdown table for display in metric cards."""
    lines = []
    
    if not examples:
        return ["*No examples available.*"]
    
    # Take the first few examples
    sample_examples = examples[:max_examples]
    
    # Determine if we have reasoning field (augmented examples)
    has_reasoning = any(example.get('reasoning') for example in sample_examples)
    
    if has_reasoning:
        # Create markdown table header with reasoning
        lines.extend([
            "| Input Text | Output Text | Score | Reasoning |",
            "|------------|-------------|-------|-----------|"
        ])
    else:
        # Create markdown table header without reasoning
        lines.extend([
            "| Input Text | Output Text | Score |",
            "|------------|-------------|-------|"
        ])
    
    # Add each example as a table row
    for i, example in enumerate(sample_examples):
        # Extract fields from the example - handle both augmented and non-augmented formats
        input_text = example.get('input_text', 'N/A')
        output_text = example.get('output_text', 'N/A')
        score = example.get('score', 'N/A')
        reasoning = example.get('reasoning', 'N/A') if has_reasoning else None
        
        # Handle different field names that might exist
        if input_text == 'N/A':
            input_text = example.get('text', 'N/A')  # Fallback for some formats
        
        # Increased text limits for better readability
        input_limit = 400  # Increased from 80
        output_limit = 400  # Increased from 80
        reasoning_limit = 300  # Increased from 100
        
        # Truncate long text for table readability, intelligently avoiding breaking markdown links
        input_text = smart_truncate_text(str(input_text), input_limit)
        output_text = smart_truncate_text(str(output_text), output_limit)
        if reasoning:
            reasoning = smart_truncate_text(str(reasoning), reasoning_limit)
        
        # Escape pipe characters and newlines for markdown table
        input_text = str(input_text).replace("|", "\\|").replace("\n", " ")
        output_text = str(output_text).replace("|", "\\|").replace("\n", " ")
        
        if has_reasoning and reasoning:
            reasoning = str(reasoning).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {input_text} | {output_text} | {score} | {reasoning} |")
        else:
            lines.append(f"| {input_text} | {output_text} | {score} |")
    
    # Add note if there are more examples
    if len(examples) > max_examples:
        lines.append("")
        lines.append(f"*Showing {max_examples} of {len(examples)} total examples.*")
    
    return lines


def extract_prompt_instructions_from_signature(signature: Dict[str, Any]) -> str:
    """Extract the main instructions from a DSPy signature."""
    try:
        # Check for direct instructions field
        if 'instructions' in signature:
            instructions = signature['instructions']
            if instructions and instructions.strip():
                return instructions.strip()
        
        # Check for instructions in fields
        if 'fields' in signature:
            for field in signature['fields']:
                if isinstance(field, dict) and 'instructions' in field:
                    instructions = field['instructions']
                    if instructions and instructions.strip():
                        return instructions.strip()
        
        # For some formats, instructions might be in the signature description
        if 'description' in signature:
            description = signature['description']
            if description and description.strip():
                return description.strip()
        
        # Check if the signature itself has a docstring-like instruction
        if isinstance(signature, dict):
            for key, value in signature.items():
                if key.lower() in ['instruction', 'prompt', 'description', 'task'] and isinstance(value, str):
                    if value and value.strip():
                        return value.strip()
                        
    except Exception as e:
        print(f"Warning: Could not extract instructions from signature: {e}")
    
    return "No instructions available."


def load_optimized_program_from_embedded_data(embedded_data_str: str) -> Tuple[Optional[Dict], List[Dict], Optional[str]]:
    """
    Load DSPy program data, examples, and instructions from embedded JSON string.
    
    Returns:
        Tuple of (full_data, examples, instructions)
    """
    try:
        data = json.loads(embedded_data_str)
        examples = extract_examples_from_dspy_data(data)
        signature = extract_signature_from_dspy_data(data)
        instructions = extract_prompt_instructions_from_signature(signature) if signature else None
        
        return data, examples, instructions
    except Exception as e:
        print(f"Warning: Could not parse embedded DSPy data: {e}")
        return None, [], None


def inspect_dspy_program_from_path(prompt_path: str) -> Tuple[Optional[Dict], List[Dict], Optional[str]]:
    """
    Load and inspect a DSPy program from a file path.
    
    Returns:
        Tuple of (full_data, examples, instructions)
    """
    data = load_dspy_program_from_path(prompt_path)
    if not data:
        return None, [], None
    
    examples = extract_examples_from_dspy_data(data)
    signature = extract_signature_from_dspy_data(data)
    instructions = extract_prompt_instructions_from_signature(signature) if signature else None
    
    return data, examples, instructions 