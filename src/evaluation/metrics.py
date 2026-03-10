"""
Custom Metrics — Robustness scoring and perturbation gap analysis.

Implements the "reasoning robustness score" that measures the delta
between model performance on original vs. perturbed prompts.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RobustnessReport:
    """Robustness analysis of a model across perturbation types."""
    model: str
    domain: str
    original_accuracy: float
    perturbed_accuracy: float
    robustness_score: float  # 1 - (original - perturbed) / original
    absolute_drop: float
    relative_drop: float
    per_perturbation: dict  # type -> accuracy
    n_original: int
    n_perturbed: int


def compute_robustness_score(original_results_path: str,
                              perturbed_results_path: str,
                              model_name: str = "",
                              domain: str = "") -> RobustnessReport:
    """
    Compute robustness score by comparing original vs perturbed results.

    Robustness Score = 1 - (accuracy_drop / original_accuracy)
    Score of 1.0 = perfectly robust (no drop)
    Score of 0.0 = complete collapse under perturbation
    """
    def load_results(path):
        results = []
        with open(path) as f:
            for line in f:
                results.append(json.loads(line.strip()))
        return results

    original = load_results(original_results_path)
    perturbed = load_results(perturbed_results_path)

    orig_correct = sum(1 for r in original if r.get("correct", False))
    orig_acc = orig_correct / len(original) if original else 0.0

    # Group perturbed results by perturbation type
    per_type = {}
    total_perturbed_correct = 0
    total_perturbed = 0

    for r in perturbed:
        p_type = r.get("perturbation_type", "unknown")
        is_correct = r.get("correct", False)

        if p_type not in per_type:
            per_type[p_type] = {"correct": 0, "total": 0}
        per_type[p_type]["total"] += 1
        per_type[p_type]["correct"] += int(is_correct)
        total_perturbed_correct += int(is_correct)
        total_perturbed += 1

    perturbed_acc = total_perturbed_correct / total_perturbed if total_perturbed else 0.0

    absolute_drop = orig_acc - perturbed_acc
    relative_drop = absolute_drop / orig_acc if orig_acc > 0 else 0.0
    robustness_score = 1.0 - relative_drop

    per_perturbation_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in per_type.items()
    }

    report = RobustnessReport(
        model=model_name,
        domain=domain,
        original_accuracy=round(orig_acc, 4),
        perturbed_accuracy=round(perturbed_acc, 4),
        robustness_score=round(robustness_score, 4),
        absolute_drop=round(absolute_drop, 4),
        relative_drop=round(relative_drop, 4),
        per_perturbation={k: round(v, 4) for k, v in per_perturbation_acc.items()},
        n_original=len(original),
        n_perturbed=total_perturbed,
    )

    logger.info(f"Robustness Report for {model_name}:")
    logger.info(f"  Original Accuracy:  {report.original_accuracy:.2%}")
    logger.info(f"  Perturbed Accuracy: {report.perturbed_accuracy:.2%}")
    logger.info(f"  Robustness Score:   {report.robustness_score:.4f}")
    logger.info(f"  Absolute Drop:      {report.absolute_drop:.2%}")
    for p_type, acc in per_perturbation_acc.items():
        logger.info(f"    {p_type}: {acc:.2%}")

    return report


def compute_dbas_robustness(original_scores_path: str,
                             perturbed_scores_path: str) -> dict:
    """
    Compare D-B-A-S rubric scores between original and perturbed outputs.
    Returns per-dimension robustness analysis.
    """
    with open(original_scores_path) as f:
        original = json.load(f)
    with open(perturbed_scores_path) as f:
        perturbed = json.load(f)

    dimensions = ["depth", "breadth", "accuracy", "safety", "composite"]
    report = {}

    for dim in dimensions:
        orig_avg = original["averages"].get(dim, 0.0)
        pert_avg = perturbed["averages"].get(dim, 0.0)
        drop = orig_avg - pert_avg
        relative = drop / orig_avg if orig_avg > 0 else 0.0

        report[dim] = {
            "original": round(orig_avg, 3),
            "perturbed": round(pert_avg, 3),
            "absolute_drop": round(drop, 3),
            "relative_drop": round(relative, 4),
            "robustness": round(1 - relative, 4),
        }

    return report


def statistical_significance(scores_a: list, scores_b: list,
                              alpha: float = 0.05) -> dict:
    """
    Paired bootstrap test for statistical significance between two score sets.
    """
    n = min(len(scores_a), len(scores_b))
    a = np.array(scores_a[:n])
    b = np.array(scores_b[:n])
    observed_diff = np.mean(a) - np.mean(b)

    n_bootstrap = 10000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_diff = np.mean(a[indices]) - np.mean(b[indices])
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)
    p_value = np.mean(bootstrap_diffs <= 0) if observed_diff > 0 else np.mean(bootstrap_diffs >= 0)

    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return {
        "observed_difference": round(float(observed_diff), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < alpha),
        "confidence_interval": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }
