"""
Adversarial Evaluation Module
=============================
Generates controlled perturbations of benchmark questions and measures
model robustness by comparing original vs. perturbed accuracy.

Perturbation types:
- numerical_swap: Change numerical values while preserving problem logic
- irrelevant_context: Inject distracting but irrelevant sentences
- phrasing_variation: Rephrase questions without changing meaning
- domain_transfer: Same reasoning structure, different domain surface
- negation_flip: Introduce or remove negation in questions
"""

import json
import random
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PERTURBATION_TYPES = [
    "numerical_swap",
    "irrelevant_context",
    "phrasing_variation",
    "domain_transfer",
    "negation_flip",
]

IRRELEVANT_CONTEXTS = [
    "Note that the experiment was conducted on a Tuesday.",
    "The researcher's office had a window facing north.",
    "This question was originally written in French and translated.",
    "The textbook containing this problem was published in 2019.",
    "A colleague mentioned this topic at a recent conference.",
    "The laboratory where this was tested also studies marine biology.",
    "The dataset was stored on a server located in Oregon.",
    "An unrelated study on sleep patterns was published the same week.",
]

PHRASING_TEMPLATES = {
    "calculate": ["compute", "determine", "find the value of", "work out"],
    "explain": ["describe", "elaborate on", "provide an account of", "clarify"],
    "which of the following": ["among the options below", "from the choices listed", "select the option that"],
    "what is": ["identify", "state", "name"],
    "how many": ["what is the count of", "determine the number of"],
    "why does": ["what causes", "what is the reason that", "explain why"],
}


@dataclass
class PerturbationResult:
    """Stores results for a single question's perturbation analysis."""
    original_question: str
    perturbed_questions: list[str] = field(default_factory=list)
    perturbation_types: list[str] = field(default_factory=list)
    original_correct: bool = False
    perturbed_correct: list[bool] = field(default_factory=list)
    original_answer: str = ""
    perturbed_answers: list[str] = field(default_factory=list)
    ground_truth: str = ""


@dataclass
class RobustnessReport:
    """Aggregate robustness metrics across a dataset."""
    original_acc: float = 0.0
    perturbed_acc: float = 0.0
    robustness_score: float = 0.0
    per_type_scores: dict = field(default_factory=dict)
    n_questions: int = 0
    n_perturbations: int = 0
    detailed_results: list[PerturbationResult] = field(default_factory=list)


class AdversarialEvaluator:
    """
    Generates adversarial perturbations of benchmark questions and
    evaluates model robustness.

    Args:
        perturbation_types: List of perturbation strategies to apply.
        model_name: HuggingFace model identifier.
        device: Torch device ("cuda", "cpu", or "auto").
        max_new_tokens: Max tokens for model generation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        perturbation_types: list[str] | None = None,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 256,
        seed: int = 42,
    ):
        self.perturbation_types = perturbation_types or PERTURBATION_TYPES
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        random.seed(seed)

        self._validate_perturbation_types()

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _validate_perturbation_types(self):
        for pt in self.perturbation_types:
            if pt not in PERTURBATION_TYPES:
                raise ValueError(
                    f"Unknown perturbation type: {pt}. "
                    f"Valid types: {PERTURBATION_TYPES}"
                )

    # ------------------------------------------------------------------
    # Perturbation generators
    # ------------------------------------------------------------------

    def _perturb_numerical(self, question: str) -> str:
        """Swap numerical values while preserving structure."""
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', question)
        if not numbers:
            return question

        result = question
        for num_str in numbers:
            num = float(num_str)
            if num == 0:
                new_num = random.choice([1, 2, 5])
            elif num.is_integer():
                delta = max(1, int(num * 0.3))
                new_num = int(num) + random.choice([-delta, delta])
                new_num = max(1, new_num)  # keep positive
            else:
                delta = num * 0.3
                new_num = round(num + random.uniform(-delta, delta), 2)
                new_num = max(0.01, new_num)
            result = result.replace(num_str, str(new_num), 1)

        return result

    def _perturb_irrelevant_context(self, question: str) -> str:
        """Insert an irrelevant sentence into the question."""
        distractor = random.choice(IRRELEVANT_CONTEXTS)
        sentences = question.split('. ')
        if len(sentences) > 1:
            insert_pos = random.randint(1, len(sentences) - 1)
            sentences.insert(insert_pos, distractor.rstrip('.'))
            return '. '.join(sentences)
        return f"{distractor} {question}"

    def _perturb_phrasing(self, question: str) -> str:
        """Rephrase using synonym templates."""
        result = question.lower()
        for original, replacements in PHRASING_TEMPLATES.items():
            if original in result:
                replacement = random.choice(replacements)
                result = result.replace(original, replacement, 1)
                # Restore original capitalization for first char
                if question[0].isupper():
                    result = result[0].upper() + result[1:]
                return result
        return question  # no matching template found

    def _perturb_negation(self, question: str) -> str:
        """Flip negation in the question."""
        negation_patterns = [
            (r'\bis NOT\b', 'is'),
            (r'\bis not\b', 'is'),
            (r'\bNOT\b', ''),
            (r'\bnot\b', ''),
            (r'\bcannot\b', 'can'),
            (r"\bcan't\b", 'can'),
            (r"\bdon't\b", 'do'),
            (r"\bdoesn't\b", 'does'),
        ]
        for pattern, replacement in negation_patterns:
            if re.search(pattern, question):
                return re.sub(pattern, replacement, question, count=1).strip()

        # If no negation found, add one
        question = question.replace(' is ', ' is NOT ', 1)
        return question

    def _perturb_domain_transfer(self, question: str) -> str:
        """Apply surface-level domain swaps."""
        domain_swaps = {
            "molecule": "recipe ingredient",
            "atom": "building block",
            "electron": "particle",
            "cell": "compartment",
            "patient": "client",
            "chemical": "substance",
            "compound": "mixture",
            "reaction": "process",
            "enzyme": "catalyst",
            "gene": "instruction set",
        }
        result = question
        for original, replacement in domain_swaps.items():
            if original.lower() in result.lower():
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                result = pattern.sub(replacement, result, count=1)
                break
        return result

    def generate_perturbation(
        self, question: str, perturbation_type: str
    ) -> str:
        """Generate a single perturbation of a question."""
        generators = {
            "numerical_swap": self._perturb_numerical,
            "irrelevant_context": self._perturb_irrelevant_context,
            "phrasing_variation": self._perturb_phrasing,
            "negation_flip": self._perturb_negation,
            "domain_transfer": self._perturb_domain_transfer,
        }
        generator = generators.get(perturbation_type)
        if generator is None:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        return generator(question)

    def generate_all_perturbations(
        self, question: str, n_per_type: int = 1
    ) -> list[tuple[str, str]]:
        """Generate perturbations across all configured types."""
        perturbations = []
        for ptype in self.perturbation_types:
            for _ in range(n_per_type):
                perturbed = self.generate_perturbation(question, ptype)
                if perturbed != question:  # only keep actual perturbations
                    perturbations.append((ptype, perturbed))
        return perturbations

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def _get_model_answer(self, question: str, choices: list[str] | None = None) -> str:
        """Get model's answer for a question."""
        if choices:
            options_str = "\n".join(
                f"{chr(65+i)}. {c}" for i, c in enumerate(choices)
            )
            prompt = (
                f"Answer the following multiple-choice question. "
                f"Reply with ONLY the letter (A, B, C, or D).\n\n"
                f"Question: {question}\n{options_str}\n\nAnswer:"
            )
        else:
            prompt = f"Answer the following question concisely.\n\nQuestion: {question}\n\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    def _check_correct(
        self, model_answer: str, ground_truth: str, choices: list[str] | None = None
    ) -> bool:
        """Check if model answer matches ground truth."""
        model_clean = model_answer.strip().upper()
        truth_clean = ground_truth.strip().upper()

        # For multiple choice, check letter match
        if choices and len(truth_clean) == 1 and truth_clean.isalpha():
            return model_clean.startswith(truth_clean)

        return truth_clean in model_clean or model_clean in truth_clean

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(
        self, dataset: str, subset: str | None = None, n_samples: int = 100
    ) -> list[dict]:
        """Load and format evaluation dataset."""
        logger.info(f"Loading dataset: {dataset} (subset={subset}, n={n_samples})")

        if dataset == "mmlu":
            config = subset or "all"
            ds = load_dataset("cais/mmlu", config, split="test")
            items = []
            for row in ds.select(range(min(n_samples, len(ds)))):
                items.append({
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": chr(65 + row["answer"]),  # 0->A, 1->B, etc.
                })
            return items

        elif dataset == "truthfulqa":
            ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
            items = []
            for row in ds.select(range(min(n_samples, len(ds)))):
                choices = row["mc1_targets"]["choices"]
                labels = row["mc1_targets"]["labels"]
                correct_idx = labels.index(1)
                items.append({
                    "question": row["question"],
                    "choices": choices[:4],
                    "answer": chr(65 + correct_idx),
                })
            return items

        elif dataset == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="test")
            items = []
            for row in ds.select(range(min(n_samples, len(ds)))):
                answer = row["answer"].split("####")[-1].strip()
                items.append({
                    "question": row["question"],
                    "choices": None,
                    "answer": answer,
                })
            return items

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate_robustness(
        self,
        dataset: str = "mmlu",
        subset: str | None = "stem",
        n_samples: int = 100,
        n_perturbations: int = 5,
        output_path: str | None = None,
    ) -> RobustnessReport:
        """
        Run full robustness evaluation.

        Args:
            dataset: Dataset name ("mmlu", "truthfulqa", "gsm8k").
            subset: Dataset subset/config (e.g., "stem" for MMLU).
            n_samples: Number of questions to evaluate.
            n_perturbations: Number of perturbation variants per question.
            output_path: Optional path to save detailed results JSON.

        Returns:
            RobustnessReport with aggregate and per-question metrics.
        """
        data = self._load_dataset(dataset, subset, n_samples)
        report = RobustnessReport(n_questions=len(data))

        original_correct_count = 0
        perturbed_correct_count = 0
        perturbed_total = 0
        per_type_correct = {pt: 0 for pt in self.perturbation_types}
        per_type_total = {pt: 0 for pt in self.perturbation_types}

        for item in tqdm(data, desc="Evaluating robustness"):
            question = item["question"]
            choices = item.get("choices")
            ground_truth = item["answer"]

            # Evaluate original
            orig_answer = self._get_model_answer(question, choices)
            orig_correct = self._check_correct(orig_answer, ground_truth, choices)
            original_correct_count += int(orig_correct)

            # Generate and evaluate perturbations
            result = PerturbationResult(
                original_question=question,
                original_correct=orig_correct,
                original_answer=orig_answer,
                ground_truth=ground_truth,
            )

            perturbations = self.generate_all_perturbations(question, n_per_type=1)
            for ptype, perturbed_q in perturbations[:n_perturbations]:
                pert_answer = self._get_model_answer(perturbed_q, choices)
                pert_correct = self._check_correct(pert_answer, ground_truth, choices)

                result.perturbed_questions.append(perturbed_q)
                result.perturbation_types.append(ptype)
                result.perturbed_answers.append(pert_answer)
                result.perturbed_correct.append(pert_correct)

                perturbed_correct_count += int(pert_correct)
                perturbed_total += 1
                per_type_correct[ptype] += int(pert_correct)
                per_type_total[ptype] += 1

            report.detailed_results.append(result)

        # Compute aggregate metrics
        report.original_acc = original_correct_count / max(len(data), 1)
        report.perturbed_acc = perturbed_correct_count / max(perturbed_total, 1)
        report.n_perturbations = perturbed_total
        report.robustness_score = (
            report.perturbed_acc / report.original_acc
            if report.original_acc > 0 else 0.0
        )

        for pt in self.perturbation_types:
            if per_type_total[pt] > 0:
                report.per_type_scores[pt] = {
                    "accuracy": per_type_correct[pt] / per_type_total[pt],
                    "n_samples": per_type_total[pt],
                }

        logger.info(
            f"Results — Original: {report.original_acc:.2%}, "
            f"Perturbed: {report.perturbed_acc:.2%}, "
            f"Robustness: {report.robustness_score:.3f}"
        )

        if output_path:
            self._save_results(report, output_path)

        return report

    def _save_results(self, report: RobustnessReport, path: str):
        """Save report to JSON."""
        output = {
            "model": self.model_name,
            "original_accuracy": report.original_acc,
            "perturbed_accuracy": report.perturbed_acc,
            "robustness_score": report.robustness_score,
            "n_questions": report.n_questions,
            "n_perturbations": report.n_perturbations,
            "per_type_scores": report.per_type_scores,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial LLM Evaluation")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", default="mmlu", choices=["mmlu", "truthfulqa", "gsm8k"])
    parser.add_argument("--subset", default="stem")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_perturbations", type=int, default=5)
    parser.add_argument("--output", default="results/robustness_analysis.json")
    parser.add_argument("--perturbation_types", nargs="+", default=None)
    args = parser.parse_args()

    evaluator = AdversarialEvaluator(
        model_name=args.model,
        perturbation_types=args.perturbation_types,
    )
    report = evaluator.evaluate_robustness(
        dataset=args.dataset,
        subset=args.subset,
        n_samples=args.n_samples,
        n_perturbations=args.n_perturbations,
        output_path=args.output,
    )
