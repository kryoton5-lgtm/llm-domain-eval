"""
Perturbation Engine — Generate controlled adversarial variants of evaluation prompts.

Applies systematic perturbations (numerical swap, context injection, paraphrase,
negation flip, domain transfer) while preserving core logical structure.
Inspired by GSM-Symbolic (Mirzadeh et al., 2024) and IFEval++ methodologies.
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from copy import deepcopy

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)


# ── Perturbation Types ──────────────────────────────────────────────

@dataclass
class Perturbation:
    """A single perturbation applied to a prompt."""
    original: str
    perturbed: str
    perturbation_type: str
    changes: list  # list of (old, new) substitutions
    preserves_answer: bool  # whether the correct answer changes


class NumericalSwap:
    """Replace numerical values with different but plausible numbers."""

    def __init__(self, scale_range=(0.5, 3.0), decimal_places=1):
        self.scale_range = scale_range
        self.decimal_places = decimal_places

    def apply(self, text: str) -> Perturbation:
        pattern = r'(\d+\.?\d*)\s*(kg|m|cm|km|g|mg|L|mL|s|hr|min|mph|km/h|N|J|W|Hz|mol|°C|°F|K|%|\$|€|£)?'
        matches = list(re.finditer(pattern, text))

        if not matches:
            return None

        changes = []
        result = text
        offset = 0

        for match in matches:
            original_num = float(match.group(1))
            unit = match.group(2) or ""

            # Generate a different but plausible number
            scale = random.uniform(*self.scale_range)
            while abs(scale - 1.0) < 0.15:  # Ensure meaningful change
                scale = random.uniform(*self.scale_range)

            new_num = round(original_num * scale, self.decimal_places)
            if new_num == original_num:
                new_num = original_num + random.choice([1, 2, 3, 5, 10])

            # Format to match original style
            if '.' not in match.group(1):
                new_str = str(int(new_num))
            else:
                new_str = f"{new_num:.{self.decimal_places}f}"

            old_str = match.group(1)
            start = match.start(1) + offset
            end = match.end(1) + offset
            result = result[:start] + new_str + result[end:]
            offset += len(new_str) - len(old_str)
            changes.append((f"{old_str} {unit}".strip(), f"{new_str} {unit}".strip()))

        return Perturbation(
            original=text,
            perturbed=result,
            perturbation_type="numerical_swap",
            changes=changes,
            preserves_answer=False,  # Answer changes with numbers
        )


class ContextInjection:
    """Inject irrelevant but plausible context into the prompt."""

    DISTRACTORS = [
        "Note that the experiment was conducted on a Tuesday afternoon in the university lab.",
        "The researcher had previously worked on unrelated projects in marine biology.",
        "This problem was originally formulated by a professor during a conference in Vienna.",
        "Interestingly, the same equipment is also used in food processing applications.",
        "The laboratory where this was measured recently underwent renovation.",
        "A similar but unrelated study was published last year in a different journal.",
        "The student who collected this data was in their third year of graduate school.",
        "This calculation is commonly assigned as homework in introductory courses.",
        "The instrument used for measurement has a serial number of X-4872.",
        "Weather conditions on the day of the experiment were partly cloudy with mild wind.",
        "The funding for this research came from a government science grant.",
        "The textbook containing this problem is now in its 12th edition.",
    ]

    def apply(self, text: str) -> Perturbation:
        distractor = random.choice(self.DISTRACTORS)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) >= 2:
            insert_pos = random.randint(1, len(sentences) - 1)
            sentences.insert(insert_pos, distractor)
        else:
            sentences.append(distractor)

        perturbed = " ".join(sentences)
        return Perturbation(
            original=text,
            perturbed=perturbed,
            perturbation_type="context_injection",
            changes=[("", distractor)],
            preserves_answer=True,  # Correct answer unchanged
        )


class ParaphraseTransform:
    """Apply rule-based paraphrasing transformations."""

    TRANSFORMS = [
        (r'\bWhat is\b', 'Determine'),
        (r'\bCalculate\b', 'Find the value of'),
        (r'\bFind\b', 'Compute'),
        (r'\bHow many\b', 'What is the total number of'),
        (r'\bHow much\b', 'What quantity of'),
        (r'\bif\b', 'assuming that'),
        (r'\bgiven that\b', 'provided that'),
        (r'\bwhich\b', 'that'),
        (r'\bwill be\b', 'is going to be'),
        (r'\bcan\b', 'is able to'),
        (r'\bmust\b', 'is required to'),
        (r'\bshould\b', 'is recommended to'),
        (r'\bincreases\b', 'grows'),
        (r'\bdecreases\b', 'diminishes'),
        (r'\bapproximately\b', 'roughly'),
        (r'\bimportant\b', 'significant'),
    ]

    def apply(self, text: str) -> Perturbation:
        result = text
        changes = []
        applied = 0

        # Apply 2-4 random transforms
        transforms = random.sample(self.TRANSFORMS, min(4, len(self.TRANSFORMS)))

        for pattern, replacement in transforms:
            new_result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
            if new_result != result:
                changes.append((pattern, replacement))
                result = new_result
                applied += 1
            if applied >= 3:
                break

        if not changes:
            return None

        return Perturbation(
            original=text,
            perturbed=result,
            perturbation_type="paraphrase",
            changes=changes,
            preserves_answer=True,
        )


class NegationFlip:
    """Invert boolean conditions or add/remove negations."""

    NEGATION_PAIRS = [
        (r'\bis\b', 'is not'),
        (r'\bcan\b', 'cannot'),
        (r'\bwill\b', 'will not'),
        (r'\bdoes\b', 'does not'),
        (r'\bhas\b', 'does not have'),
        (r'\btrue\b', 'false'),
        (r'\bpossible\b', 'impossible'),
        (r'\bvalid\b', 'invalid'),
        (r'\bcorrect\b', 'incorrect'),
        (r'\bgreater than\b', 'less than'),
        (r'\babove\b', 'below'),
        (r'\bmore than\b', 'fewer than'),
    ]

    def apply(self, text: str) -> Perturbation:
        pair = random.choice(self.NEGATION_PAIRS)
        pattern, replacement = pair

        new_text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)

        if new_text == text:
            return None

        return Perturbation(
            original=text,
            perturbed=new_text,
            perturbation_type="negation_flip",
            changes=[(pattern, replacement)],
            preserves_answer=False,  # Answer changes with negation
        )


class DomainTransfer:
    """Swap domain-specific vocabulary while preserving structure."""

    DOMAIN_SWAPS = {
        "physics_to_chemistry": [
            ("velocity", "reaction rate"),
            ("mass", "molar mass"),
            ("force", "concentration"),
            ("acceleration", "rate constant"),
            ("energy", "enthalpy"),
            ("particle", "molecule"),
            ("field", "solution"),
            ("wave", "reaction"),
        ],
        "physics_to_finance": [
            ("velocity", "growth rate"),
            ("mass", "market cap"),
            ("force", "leverage"),
            ("acceleration", "compound rate"),
            ("energy", "capital"),
            ("particle", "asset"),
            ("field", "market"),
            ("equilibrium", "equilibrium price"),
        ],
    }

    def __init__(self, swap_type="physics_to_chemistry"):
        self.swap_type = swap_type
        self.swaps = self.DOMAIN_SWAPS.get(swap_type, [])

    def apply(self, text: str) -> Perturbation:
        result = text
        changes = []

        for old_term, new_term in self.swaps:
            if old_term.lower() in result.lower():
                result = re.sub(
                    rf'\b{re.escape(old_term)}\b',
                    new_term, result, count=1, flags=re.IGNORECASE
                )
                changes.append((old_term, new_term))

            if len(changes) >= 3:
                break

        if not changes:
            return None

        return Perturbation(
            original=text,
            perturbed=result,
            perturbation_type=f"domain_transfer_{self.swap_type}",
            changes=changes,
            preserves_answer=False,
        )


# ── Engine ──────────────────────────────────────────────────────────

class PerturbationEngine:
    """Orchestrate multiple perturbation types on a set of prompts."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.perturbation_types = {
            "numerical_swap": NumericalSwap(
                scale_range=config.get("numerical_scale_range", (0.5, 3.0)),
            ),
            "context_injection": ContextInjection(),
            "paraphrase": ParaphraseTransform(),
            "negation_flip": NegationFlip(),
            "domain_transfer": DomainTransfer(
                swap_type=config.get("domain_swap_type", "physics_to_chemistry"),
            ),
        }
        self.enabled_types = config.get("enabled_types", list(self.perturbation_types.keys()))
        self.perturbations_per_question = config.get("perturbations_per_question", 3)

    def perturb(self, text: str, question_id: str = "") -> list:
        """Generate perturbations for a single question."""
        results = []
        types_to_try = [t for t in self.enabled_types if t in self.perturbation_types]
        random.shuffle(types_to_try)

        for p_type in types_to_try:
            if len(results) >= self.perturbations_per_question:
                break

            perturbator = self.perturbation_types[p_type]
            try:
                result = perturbator.apply(text)
                if result and result.perturbed != result.original:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Perturbation {p_type} failed for '{question_id}': {e}")

        return results

    def perturb_dataset(self, input_path: str, output_path: str) -> dict:
        """Process a JSONL file and generate perturbed variants."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {"total_questions": 0, "total_perturbations": 0, "by_type": {}}

        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                entry = json.loads(line.strip())
                question = entry.get("question", entry.get("text", ""))
                q_id = str(entry.get("index", entry.get("id", stats["total_questions"])))

                perturbations = self.perturb(question, q_id)
                stats["total_questions"] += 1

                for p in perturbations:
                    output_entry = deepcopy(entry)
                    output_entry["question"] = p.perturbed
                    output_entry["original_question"] = p.original
                    output_entry["perturbation_type"] = p.perturbation_type
                    output_entry["changes"] = p.changes
                    output_entry["preserves_answer"] = p.preserves_answer
                    output_entry["source_id"] = q_id

                    fout.write(json.dumps(output_entry) + "\n")
                    stats["total_perturbations"] += 1
                    stats["by_type"][p.perturbation_type] = \
                        stats["by_type"].get(p.perturbation_type, 0) + 1

        logger.info(f"Generated {stats['total_perturbations']} perturbations "
                     f"from {stats['total_questions']} questions")
        for p_type, count in stats["by_type"].items():
            logger.info(f"  {p_type}: {count}")

        return stats


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate adversarial perturbations")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--perturbations-per-question", type=int, default=3)
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    config["perturbations_per_question"] = args.perturbations_per_question

    engine = PerturbationEngine(config)
    engine.perturb_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
