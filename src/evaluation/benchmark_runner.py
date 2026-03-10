"""
Benchmark Runner — Evaluate LLMs against standard and custom benchmarks.

Supports HuggingFace datasets (MMLU, TruthfulQA, GSM8K, ARC, HellaSwag)
and custom domain-specific question sets in JSONL format.
"""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name: str = "truthfulqa"
    split: str = "validation"
    max_samples: Optional[int] = None
    batch_size: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.0
    use_adapter: Optional[str] = None  # Path to LoRA adapter
    output_dir: str = "results"
    device: str = "auto"
    quantize_4bit: bool = True
    few_shot: int = 0
    domains: list = field(default_factory=lambda: ["all"])

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Dataset Registry ────────────────────────────────────────────────

DATASET_REGISTRY = {
    "truthfulqa": {
        "hf_path": "truthfulqa/truthful_qa",
        "hf_name": "generation",
        "split": "validation",
        "question_key": "question",
        "answer_key": "best_answer",
        "task_type": "generation",
    },
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "split": "test",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answer",
        "task_type": "multiple_choice",
    },
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
        "task_type": "generation",
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split": "test",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
        "task_type": "multiple_choice",
    },
    "hellaswag": {
        "hf_path": "Rowan/hellaswag",
        "hf_name": None,
        "split": "validation",
        "question_key": "ctx",
        "choices_key": "endings",
        "answer_key": "label",
        "task_type": "multiple_choice",
    },
}


# ── Model Loading ───────────────────────────────────────────────────

def load_model_and_tokenizer(config: EvalConfig):
    """Load model with optional LoRA adapter and quantization."""
    logger.info(f"Loading model: {config.model_name}")

    model_kwargs = {"device_map": config.device, "torch_dtype": torch.bfloat16}

    if config.quantize_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    if config.use_adapter:
        logger.info(f"Loading LoRA adapter from: {config.use_adapter}")
        model = PeftModel.from_pretrained(model, config.use_adapter)

    model.eval()
    return model, tokenizer


# ── Prompt Formatting ───────────────────────────────────────────────

def format_prompt(question: str, task_type: str, choices: list = None,
                  few_shot_examples: list = None) -> str:
    """Format a question into a model-ready prompt."""
    prompt_parts = []

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt_parts.append(f"Q: {ex['question']}\nA: {ex['answer']}\n")

    if task_type == "multiple_choice" and choices:
        labels = "ABCDE"
        choice_str = "\n".join(f"  {labels[i]}. {c}" for i, c in enumerate(choices))
        prompt_parts.append(
            f"Q: {question}\n{choice_str}\n\n"
            f"Answer with just the letter (A, B, C, D, or E):\nA:"
        )
    else:
        prompt_parts.append(f"Q: {question}\nA:")

    return "\n".join(prompt_parts)


# ── Generation ──────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, config: EvalConfig) -> str:
    """Generate a single response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature if config.temperature > 0 else None,
        do_sample=config.temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Evaluation Logic ────────────────────────────────────────────────

def evaluate_multiple_choice(prediction: str, answer, choices: list = None) -> bool:
    """Check if a multiple-choice prediction matches the answer."""
    pred_clean = prediction.strip().upper()
    if pred_clean and pred_clean[0] in "ABCDE":
        pred_letter = pred_clean[0]
    else:
        pred_letter = None

    if isinstance(answer, int):
        correct_letter = "ABCDE"[answer]
    elif isinstance(answer, str) and answer.strip().upper() in "ABCDE":
        correct_letter = answer.strip().upper()
    else:
        correct_letter = str(answer).strip().upper()

    return pred_letter == correct_letter


def evaluate_generation(prediction: str, reference: str) -> dict:
    """Evaluate a generated response against a reference answer."""
    pred_lower = prediction.lower().strip()
    ref_lower = reference.lower().strip()

    exact_match = pred_lower == ref_lower
    contains_answer = ref_lower in pred_lower

    # Extract numerical answer for math problems (GSM8K style)
    import re
    pred_numbers = re.findall(r"[-+]?\d*\.?\d+", prediction.split("####")[-1] if "####" in prediction else prediction)
    ref_numbers = re.findall(r"[-+]?\d*\.?\d+", reference.split("####")[-1] if "####" in reference else reference)

    numerical_match = False
    if pred_numbers and ref_numbers:
        try:
            numerical_match = abs(float(pred_numbers[-1]) - float(ref_numbers[-1])) < 1e-6
        except ValueError:
            pass

    return {
        "exact_match": exact_match,
        "contains_answer": contains_answer,
        "numerical_match": numerical_match,
    }


# ── Main Runner ─────────────────────────────────────────────────────

def run_benchmark(config: EvalConfig) -> dict:
    """Run a complete benchmark evaluation."""
    ds_info = DATASET_REGISTRY.get(config.dataset_name)
    if ds_info is None:
        raise ValueError(f"Unknown dataset: {config.dataset_name}. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")

    logger.info(f"Loading dataset: {ds_info['hf_path']}")
    dataset = load_dataset(ds_info["hf_path"], ds_info.get("hf_name"),
                           split=ds_info["split"])

    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    logger.info(f"Evaluating {len(dataset)} samples")

    model, tokenizer = load_model_and_tokenizer(config)

    results = []
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        question = example[ds_info["question_key"]]
        answer = example[ds_info["answer_key"]]

        choices = None
        if "choices_key" in ds_info and ds_info["choices_key"] in example:
            raw_choices = example[ds_info["choices_key"]]
            if isinstance(raw_choices, dict) and "text" in raw_choices:
                choices = raw_choices["text"]
            elif isinstance(raw_choices, list):
                choices = raw_choices

        prompt = format_prompt(question, ds_info["task_type"], choices)
        prediction = generate_response(model, tokenizer, prompt, config)

        if ds_info["task_type"] == "multiple_choice":
            is_correct = evaluate_multiple_choice(prediction, answer, choices)
            correct += int(is_correct)
            result_entry = {"correct": is_correct}
        else:
            gen_metrics = evaluate_generation(prediction, answer)
            is_correct = gen_metrics.get("numerical_match") or gen_metrics.get("contains_answer")
            correct += int(is_correct)
            result_entry = gen_metrics

        total += 1
        results.append({
            "index": i,
            "question": question,
            "prediction": prediction,
            "reference": answer,
            **result_entry,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(dataset)} | Accuracy: {correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "model": config.model_name,
        "adapter": config.use_adapter,
        "dataset": config.dataset_name,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "config": asdict(config),
    }

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{config.dataset_name}_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary_path = output_dir / f"{config.dataset_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Final Accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"Results saved to: {results_path}")

    return summary


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--config", type=str, default=None, help="Path to eval_config.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = EvalConfig.from_yaml(args.config)
    else:
        config = EvalConfig()

    config.model_name = args.model
    config.dataset_name = args.dataset
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.adapter:
        config.use_adapter = args.adapter
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_quantize:
        config.quantize_4bit = False

    run_benchmark(config)


if __name__ == "__main__":
    main()
