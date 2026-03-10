"""
D-B-A-S Rubric Scoring Module
==============================
Structured evaluation of LLM outputs using the Depth-Breadth-Accuracy-Style
framework. Supports both automated (LLM-as-judge) and manual scoring modes.

Dimensions:
- Depth: Quality and sophistication of reasoning
- Breadth: Coverage of relevant aspects and edge cases
- Accuracy: Factual correctness and absence of hallucinations
- Style: Clarity, structure, and domain-appropriate register
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score for a single rubric dimension."""
    dimension: str
    score: int  # 1-4
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class DBASScore:
    """Complete D-B-A-S evaluation result."""
    depth: int = 0
    breadth: int = 0
    accuracy: int = 0
    style: int = 0
    overall: float = 0.0
    dimension_details: list[DimensionScore] = field(default_factory=list)
    question: str = ""
    model_response: str = ""
    reference_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "breadth": self.breadth,
            "accuracy": self.accuracy,
            "style": self.style,
            "overall": self.overall,
            "dimension_details": [
                {
                    "dimension": d.dimension,
                    "score": d.score,
                    "rationale": d.rationale,
                    "evidence": d.evidence,
                }
                for d in self.dimension_details
            ],
        }


# Default rubric criteria
DEFAULT_RUBRIC = {
    "depth": {
        "description": "Quality and sophistication of reasoning",
        "levels": {
            4: "Expert-level reasoning with multi-step logic chains, considers edge cases and nuances, demonstrates deep domain understanding",
            3: "Solid reasoning that addresses the core problem correctly, minor gaps in explanation depth",
            2: "Surface-level explanation that identifies the right concept but lacks detailed reasoning",
            1: "Incorrect reasoning, missing key logical steps, or no reasoning provided",
        },
        "weight": 0.30,
    },
    "breadth": {
        "description": "Coverage of relevant aspects and edge cases",
        "levels": {
            4: "Covers all relevant aspects including edge cases, alternative approaches, and practical implications",
            3: "Covers main aspects of the topic with minor omissions",
            2: "Partial coverage, missing important aspects or perspectives",
            1: "Misses key aspects, addresses only a narrow slice of the question",
        },
        "weight": 0.25,
    },
    "accuracy": {
        "description": "Factual correctness and absence of hallucinations",
        "levels": {
            4: "Entirely factually correct, no hallucinations, all claims verifiable",
            3: "Minor inaccuracies that don't affect overall correctness, no hallucinations",
            2: "Some factual errors present, possible hallucinated details",
            1: "Major factual errors, clear hallucinations, or fundamentally wrong information",
        },
        "weight": 0.30,
    },
    "style": {
        "description": "Clarity, structure, and domain-appropriate register",
        "levels": {
            4: "Clear, well-structured, uses domain-appropriate terminology, good flow between ideas",
            3: "Clear with minor style issues, generally well-organized",
            2: "Understandable but poorly organized, inconsistent terminology or register",
            1: "Unclear, poorly structured, inappropriate register for the domain",
        },
        "weight": 0.15,
    },
}


class DBASScorer:
    """
    Evaluates LLM outputs using the D-B-A-S rubric framework.

    Supports two modes:
    1. Automated (LLM-as-judge): Uses a judge model to score responses
    2. Manual: Returns rubric criteria for human evaluators

    Args:
        rubric_path: Path to custom rubric JSON (uses default if None).
        judge_model: Optional LLM for automated scoring.
        domain: Domain context for evaluation ("stem", "pharmacy", "finance").
    """

    def __init__(
        self,
        rubric_path: str | None = None,
        judge_model=None,
        judge_tokenizer=None,
        domain: str = "general",
    ):
        if rubric_path and Path(rubric_path).exists():
            with open(rubric_path) as f:
                self.rubric = json.load(f)
            logger.info(f"Loaded custom rubric from {rubric_path}")
        else:
            self.rubric = DEFAULT_RUBRIC
            logger.info("Using default D-B-A-S rubric")

        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.domain = domain

    def _build_judge_prompt(
        self, question: str, response: str, reference: str, dimension: str
    ) -> str:
        """Build evaluation prompt for the judge model."""
        criteria = self.rubric[dimension]
        levels_text = "\n".join(
            f"  Score {k}: {v}" for k, v in criteria["levels"].items()
        )

        prompt = f"""You are an expert evaluator assessing the quality of an AI response.

DIMENSION: {dimension.upper()} — {criteria['description']}

SCORING CRITERIA:
{levels_text}

QUESTION: {question}

REFERENCE ANSWER: {reference}

MODEL RESPONSE TO EVALUATE: {response}

Evaluate the model response on the {dimension.upper()} dimension.
Respond in this exact JSON format:
{{"score": <1-4>, "rationale": "<brief explanation>", "evidence": ["<specific quote or observation>"]}}

Your evaluation:"""
        return prompt

    def _score_with_judge(
        self, question: str, response: str, reference: str, dimension: str
    ) -> DimensionScore:
        """Score a single dimension using the judge model."""
        import torch

        prompt = self._build_judge_prompt(question, response, reference, dimension)
        inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(
            self.judge_model.device
        )

        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
            )
        raw = self.judge_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse JSON response
        try:
            # Find JSON in response
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            return DimensionScore(
                dimension=dimension,
                score=max(1, min(4, int(parsed["score"]))),
                rationale=parsed.get("rationale", ""),
                evidence=parsed.get("evidence", []),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse judge output for {dimension}: {e}")
            return DimensionScore(
                dimension=dimension,
                score=2,
                rationale=f"Parse error, defaulted to 2. Raw: {raw[:100]}",
            )

    def _score_heuristic(
        self, question: str, response: str, reference: str, dimension: str
    ) -> DimensionScore:
        """Heuristic scoring when no judge model is available."""
        response_lower = response.lower()
        reference_lower = reference.lower()
        ref_words = set(reference_lower.split())
        resp_words = set(response_lower.split())

        if dimension == "accuracy":
            # Rough overlap-based accuracy heuristic
            if not reference:
                return DimensionScore(dimension=dimension, score=2, rationale="No reference provided")
            overlap = len(ref_words & resp_words) / max(len(ref_words), 1)
            if overlap > 0.6:
                score = 4
            elif overlap > 0.4:
                score = 3
            elif overlap > 0.2:
                score = 2
            else:
                score = 1
            return DimensionScore(
                dimension=dimension,
                score=score,
                rationale=f"Word overlap with reference: {overlap:.2%}",
            )

        elif dimension == "depth":
            # Length and reasoning indicators
            reasoning_markers = [
                "because", "therefore", "since", "due to", "as a result",
                "this means", "consequently", "implies", "follows that",
                "first", "second", "third", "step",
            ]
            marker_count = sum(1 for m in reasoning_markers if m in response_lower)
            word_count = len(response.split())

            if marker_count >= 4 and word_count > 150:
                score = 4
            elif marker_count >= 2 and word_count > 80:
                score = 3
            elif word_count > 30:
                score = 2
            else:
                score = 1
            return DimensionScore(
                dimension=dimension,
                score=score,
                rationale=f"Reasoning markers: {marker_count}, word count: {word_count}",
            )

        elif dimension == "breadth":
            # Paragraph/section count as proxy for coverage
            paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
            unique_topics = len(set(
                word for p in paragraphs for word in p.lower().split()[:5]
            ))

            if len(paragraphs) >= 4 and unique_topics > 15:
                score = 4
            elif len(paragraphs) >= 2:
                score = 3
            elif len(paragraphs) >= 1 and len(response) > 50:
                score = 2
            else:
                score = 1
            return DimensionScore(
                dimension=dimension,
                score=score,
                rationale=f"Paragraphs: {len(paragraphs)}, topic diversity: {unique_topics}",
            )

        elif dimension == "style":
            # Basic style checks
            has_structure = any(
                marker in response for marker in ["1.", "2.", "- ", "* ", "##", "**"]
            )
            avg_sentence_len = len(response.split()) / max(
                response.count(".") + response.count("!") + response.count("?"), 1
            )
            is_well_paced = 8 < avg_sentence_len < 30

            if has_structure and is_well_paced and len(response) > 100:
                score = 4
            elif is_well_paced:
                score = 3
            elif len(response) > 20:
                score = 2
            else:
                score = 1
            return DimensionScore(
                dimension=dimension,
                score=score,
                rationale=f"Structured: {has_structure}, avg sentence len: {avg_sentence_len:.1f}",
            )

        return DimensionScore(dimension=dimension, score=2, rationale="Unknown dimension")

    def evaluate(
        self,
        question: str,
        model_response: str,
        reference_answer: str = "",
    ) -> DBASScore:
        """
        Evaluate a model response using the full D-B-A-S rubric.

        Args:
            question: The input question/prompt.
            model_response: The model's generated response.
            reference_answer: Expert reference answer for comparison.

        Returns:
            DBASScore with per-dimension and overall scores.
        """
        result = DBASScore(
            question=question,
            model_response=model_response,
            reference_answer=reference_answer,
        )

        scoring_fn = (
            self._score_with_judge if self.judge_model else self._score_heuristic
        )

        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, criteria in self.rubric.items():
            dim_score = scoring_fn(question, model_response, reference_answer, dimension)
            result.dimension_details.append(dim_score)

            setattr(result, dimension, dim_score.score)
            weight = criteria.get("weight", 0.25)
            weighted_sum += dim_score.score * weight
            total_weight += weight

        result.overall = round(weighted_sum / max(total_weight, 0.01), 2)

        return result

    def evaluate_batch(
        self,
        items: list[dict],
        output_path: str | None = None,
    ) -> list[DBASScore]:
        """
        Evaluate a batch of question-response pairs.

        Args:
            items: List of dicts with keys: question, model_response, reference_answer
            output_path: Optional path to save results JSON.

        Returns:
            List of DBASScore objects.
        """
        from tqdm import tqdm

        results = []
        for item in tqdm(items, desc="D-B-A-S Evaluation"):
            score = self.evaluate(
                question=item["question"],
                model_response=item["model_response"],
                reference_answer=item.get("reference_answer", ""),
            )
            results.append(score)

        if output_path:
            output = [r.to_dict() for r in results]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Batch results saved to {output_path}")

            # Print summary
            avg_scores = {
                "depth": sum(r.depth for r in results) / len(results),
                "breadth": sum(r.breadth for r in results) / len(results),
                "accuracy": sum(r.accuracy for r in results) / len(results),
                "style": sum(r.style for r in results) / len(results),
                "overall": sum(r.overall for r in results) / len(results),
            }
            logger.info(f"Average scores: {avg_scores}")

        return results

    def get_rubric_display(self) -> str:
        """Return a formatted string of the rubric for human evaluators."""
        lines = ["=" * 60, "D-B-A-S EVALUATION RUBRIC", "=" * 60, ""]
        for dim, criteria in self.rubric.items():
            lines.append(f"{'─' * 40}")
            lines.append(f"  {dim.upper()} (weight: {criteria['weight']:.0%})")
            lines.append(f"  {criteria['description']}")
            lines.append(f"{'─' * 40}")
            for level, desc in criteria["levels"].items():
                lines.append(f"    [{level}] {desc}")
            lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo with heuristic scoring
    scorer = DBASScorer(domain="pharmacy")
    print(scorer.get_rubric_display())

    score = scorer.evaluate(
        question="Explain the mechanism of action of metformin in Type 2 diabetes.",
        model_response=(
            "Metformin works primarily by reducing hepatic glucose production through "
            "activation of AMP-activated protein kinase (AMPK). This leads to decreased "
            "gluconeogenesis in the liver. Additionally, metformin improves insulin "
            "sensitivity in peripheral tissues, particularly skeletal muscle, enhancing "
            "glucose uptake. It also reduces intestinal absorption of glucose. Unlike "
            "sulfonylureas, metformin does not stimulate insulin secretion, which means "
            "it carries a lower risk of hypoglycemia. Common side effects include "
            "gastrointestinal issues such as nausea and diarrhea, particularly at "
            "treatment initiation."
        ),
        reference_answer=(
            "Metformin's primary mechanism involves AMPK activation leading to reduced "
            "hepatic gluconeogenesis. Secondary mechanisms include improved peripheral "
            "insulin sensitivity and reduced intestinal glucose absorption."
        ),
    )

    print(f"\nResults:")
    print(f"  Depth:    {score.depth}/4")
    print(f"  Breadth:  {score.breadth}/4")
    print(f"  Accuracy: {score.accuracy}/4")
    print(f"  Style:    {score.style}/4")
    print(f"  Overall:  {score.overall}/4")
    for detail in score.dimension_details:
        print(f"\n  [{detail.dimension}] {detail.rationale}")
