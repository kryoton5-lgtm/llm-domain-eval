"""Shared utilities for the LLM evaluation pipeline."""

import json
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict | list, path: str, indent: int = 2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
