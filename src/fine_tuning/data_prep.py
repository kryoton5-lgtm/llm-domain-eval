"""
Data Preparation — Load, filter, and format datasets for fine-tuning and evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_oasst1(split: str = "train", lang: str = "en",
                max_rank: int = 2, max_samples: Optional[int] = None):
    """Load OpenAssistant OASST1 with quality filtering."""
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    ds = ds.filter(lambda x: x["lang"] == lang and x.get("rank") is not None and x["rank"] <= max_rank)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    logger.info(f"OASST1: {len(ds)} samples (lang={lang}, max_rank={max_rank})")
    return ds


def load_dolly(max_samples: Optional[int] = None):
    """Load Databricks Dolly 15K."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    logger.info(f"Dolly: {len(ds)} samples")
    return ds


def load_truthfulqa():
    """Load TruthfulQA for evaluation."""
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    logger.info(f"TruthfulQA: {len(ds)} samples")
    return ds


def load_mmlu(subjects: list = None, max_per_subject: int = 100):
    """Load MMLU with optional subject filtering."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    if subjects:
        ds = ds.filter(lambda x: x.get("subject") in subjects)
    logger.info(f"MMLU: {len(ds)} samples")
    return ds


def export_to_jsonl(dataset, output_path: str, format_fn=None):
    """Export a HuggingFace dataset to JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for i, example in enumerate(dataset):
            entry = format_fn(example) if format_fn else dict(example)
            entry["index"] = i
            f.write(json.dumps(entry, default=str) + "\n")
            count += 1
    logger.info(f"Exported {count} entries to {output_path}")
