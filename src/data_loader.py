"""
Data Loading & Preprocessing Module
====================================
Handles dataset loading, domain filtering, and format conversion
for evaluation and fine-tuning pipelines.
"""

import logging
from typing import Optional

from datasets import load_dataset, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_DATASETS = {
    "mmlu": {
        "hf_path": "cais/mmlu",
        "splits": ["test", "validation"],
        "type": "multiple_choice",
    },
    "truthfulqa": {
        "hf_path": "truthfulqa/truthful_qa",
        "config": "multiple_choice",
        "splits": ["validation"],
        "type": "multiple_choice",
    },
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "config": "main",
        "splits": ["test", "train"],
        "type": "open_ended",
    },
    "arc": {
        "hf_path": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "splits": ["test", "train"],
        "type": "multiple_choice",
    },
    "hellaswag": {
        "hf_path": "Rowan/hellaswag",
        "splits": ["validation"],
        "type": "multiple_choice",
    },
    "oasst1": {
        "hf_path": "OpenAssistant/oasst1",
        "splits": ["train", "validation"],
        "type": "conversation",
    },
    "dolly": {
        "hf_path": "databricks/databricks-dolly-15k",
        "splits": ["train"],
        "type": "instruction",
    },
}

# MMLU subject-to-domain mapping
MMLU_DOMAIN_MAP = {
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "pharmacy": [
        "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "medical_genetics", "nutrition", "professional_medicine", "virology",
    ],
    "finance": [
        "business_ethics", "econometrics", "high_school_macroeconomics",
        "high_school_microeconomics", "management", "marketing",
        "professional_accounting",
    ],
}


def load_evaluation_dataset(
    name: str,
    domain: Optional[str] = None,
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load and format a dataset for evaluation.

    Args:
        name: Dataset identifier (see SUPPORTED_DATASETS).
        domain: Optional domain filter ("stem", "pharmacy", "finance").
        split: Dataset split to load.
        n_samples: Max number of samples (None = all).
        seed: Random seed for sampling.

    Returns:
        List of formatted evaluation items.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. Supported: {list(SUPPORTED_DATASETS.keys())}"
        )

    meta = SUPPORTED_DATASETS[name]
    config = meta.get("config")

    logger.info(f"Loading {name} (split={split}, domain={domain})")

    if name == "mmlu" and domain:
        subjects = MMLU_DOMAIN_MAP.get(domain, [])
        if not subjects:
            logger.warning(f"No MMLU subjects for domain '{domain}', loading all")
            ds = load_dataset(meta["hf_path"], "all", split=split)
        else:
            # Load each subject and concatenate
            from datasets import concatenate_datasets
            all_ds = []
            for subj in subjects:
                try:
                    subj_ds = load_dataset(meta["hf_path"], subj, split=split)
                    all_ds.append(subj_ds)
                except Exception as e:
                    logger.warning(f"Failed to load MMLU/{subj}: {e}")
            ds = concatenate_datasets(all_ds) if all_ds else Dataset.from_list([])
    elif config:
        ds = load_dataset(meta["hf_path"], config, split=split)
    else:
        ds = load_dataset(meta["hf_path"], split=split)

    # Sample if needed
    if n_samples and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))

    # Format to common schema
    items = _format_dataset(ds, name)
    logger.info(f"Loaded {len(items)} items from {name}")
    return items


def _format_dataset(ds: Dataset, name: str) -> list[dict]:
    """Convert dataset-specific schema to common evaluation format."""
    items = []

    for row in ds:
        if name == "mmlu":
            items.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": chr(65 + row["answer"]),
                "type": "multiple_choice",
                "source": "mmlu",
                "subject": row.get("subject", "unknown"),
            })

        elif name == "truthfulqa":
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            correct_idx = labels.index(1) if 1 in labels else 0
            items.append({
                "question": row["question"],
                "choices": choices[:4],
                "answer": chr(65 + correct_idx),
                "type": "multiple_choice",
                "source": "truthfulqa",
            })

        elif name == "gsm8k":
            answer = row["answer"].split("####")[-1].strip()
            items.append({
                "question": row["question"],
                "choices": None,
                "answer": answer,
                "type": "open_ended",
                "source": "gsm8k",
            })

        elif name == "arc":
            choices = row["choices"]["text"]
            labels = row["choices"]["label"]
            answer = row["answerKey"]
            items.append({
                "question": row["question"],
                "choices": choices,
                "answer": answer,
                "type": "multiple_choice",
                "source": "arc",
            })

        elif name == "dolly":
            instruction = row["instruction"]
            if row.get("context"):
                instruction = f"{instruction}\n\nContext: {row['context']}"
            items.append({
                "instruction": instruction,
                "response": row["response"],
                "category": row.get("category", "unknown"),
                "type": "instruction",
                "source": "dolly",
            })

    return items


def get_dataset_info(name: str) -> dict:
    """Return metadata about a supported dataset."""
    if name not in SUPPORTED_DATASETS:
        return {"error": f"Unknown dataset: {name}"}
    return SUPPORTED_DATASETS[name]


def list_datasets() -> list[str]:
    """List all supported dataset names."""
    return list(SUPPORTED_DATASETS.keys())
