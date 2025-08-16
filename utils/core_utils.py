"""
core_utils.py

General utility functions for the LLM application.
Contains memory management and text processing utilities.
"""

import re
import gc
import torch


def clean_model_name(name: str) -> str:
    """
    Remove parenthetical details from model name for display.
    Args:
        name (str): Model name string.
    Returns:
        str: Cleaned model name.
    """
    return re.sub(r"\s*\([^)]*\)", "", name).strip()


def free_memory():
    """
    Run garbage collection and clear CUDA memory if available.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
