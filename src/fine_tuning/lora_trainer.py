"""
LoRA Trainer — Parameter-efficient fine-tuning pipeline for domain-specific LLM adaptation.

Supports QLoRA (4-bit quantization) with configurable LoRA parameters,
domain-specific data filtering, and evaluation during training.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────

DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "task_type": "CAUSAL_LM",
    "bias": "none",
}

DEFAULT_TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "bf16": True,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
}

# Domain-specific keywords for filtering instruction data
DOMAIN_KEYWORDS = {
    "stem": [
        "physics", "chemistry", "biology", "mathematics", "engineering",
        "equation", "theorem", "formula", "calculate", "derive",
        "experiment", "hypothesis", "scientific", "quantum", "molecular",
        "thermodynamic", "electromagnetic", "kinematic", "integral", "derivative",
    ],
    "pharmacy": [
        "drug", "medication", "dosage", "pharmacology", "prescription",
        "adverse", "interaction", "contraindication", "therapeutic",
        "pharmaceutical", "clinical", "patient", "diagnosis", "treatment",
        "enzyme", "receptor", "bioavailability", "half-life", "toxicity",
    ],
    "finance": [
        "financial", "investment", "stock", "bond", "portfolio",
        "revenue", "profit", "valuation", "accounting", "audit",
        "EBITDA", "cash flow", "balance sheet", "P/E ratio", "market cap",
        "interest rate", "inflation", "fiscal", "monetary", "derivative",
    ],
}


# ── Dataset Preparation ─────────────────────────────────────────────

def load_and_filter_dataset(dataset_name: str, domain: str,
                             max_samples: Optional[int] = None):
    """Load a dataset and optionally filter for domain relevance."""
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "oasst1":
        dataset = load_dataset("OpenAssistant/oasst1")
        # Filter for English, top-level messages with high quality scores
        train_data = dataset["train"].filter(
            lambda x: x["lang"] == "en" and x["rank"] is not None and x["rank"] <= 2
        )
    elif dataset_name == "dolly":
        dataset = load_dataset("databricks/databricks-dolly-15k")
        train_data = dataset["train"]
    elif dataset_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca")
        train_data = dataset["train"]
    else:
        dataset = load_dataset(dataset_name)
        train_data = dataset["train"]

    # Domain filtering
    if domain != "all" and domain in DOMAIN_KEYWORDS:
        keywords = DOMAIN_KEYWORDS[domain]
        logger.info(f"Filtering for domain: {domain} ({len(keywords)} keywords)")

        def is_domain_relevant(example):
            text = json.dumps(example).lower()
            return any(kw in text for kw in keywords)

        train_data = train_data.filter(is_domain_relevant)
        logger.info(f"Domain-filtered samples: {len(train_data)}")

    if max_samples and len(train_data) > max_samples:
        train_data = train_data.shuffle(seed=42).select(range(max_samples))

    return train_data


def format_instruction(example: dict, dataset_name: str) -> str:
    """Format a dataset example into an instruction-response string."""
    if dataset_name == "oasst1":
        return f"### Human:\n{example.get('text', '')}\n\n### Assistant:\n"
    elif dataset_name == "dolly":
        instruction = example.get("instruction", "")
        context = example.get("context", "")
        response = example.get("response", "")
        if context:
            return (f"### Instruction:\n{instruction}\n\n"
                    f"### Context:\n{context}\n\n"
                    f"### Response:\n{response}")
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    elif dataset_name == "alpaca":
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        if inp:
            return (f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{inp}\n\n"
                    f"### Response:\n{output}")
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    else:
        return str(example)


# ── Model Setup ─────────────────────────────────────────────────────

def setup_model_for_training(model_name: str, lora_config: dict):
    """Load base model with QLoRA and attach LoRA adapters."""
    logger.info(f"Loading base model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        task_type=TaskType.CAUSAL_LM,
        bias=lora_config.get("bias", "none"),
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config


# ── Training ────────────────────────────────────────────────────────

def train(base_model: str, dataset_name: str, domain: str,
          lora_config: dict, training_config: dict, output_dir: str):
    """Run the full LoRA fine-tuning pipeline."""
    model, tokenizer, peft_config = setup_model_for_training(base_model, lora_config)
    train_data = load_and_filter_dataset(dataset_name, domain)

    logger.info(f"Training samples: {len(train_data)}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2e-4),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_seq_length=training_config.get("max_seq_length", 2048),
        logging_steps=training_config.get("logging_steps", 10),
        save_strategy=training_config.get("save_strategy", "epoch"),
        bf16=training_config.get("bf16", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optim", "paged_adamw_8bit"),
        report_to="none",
    )

    def formatting_func(examples):
        return [format_instruction(ex, dataset_name) for ex in
                [dict(zip(examples.keys(), vals)) for vals in zip(*examples.values())]]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save adapter
    adapter_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"Adapter saved to: {adapter_path}")

    return str(adapter_path)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for domain-specific LLMs")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="oasst1",
                        choices=["oasst1", "dolly", "alpaca"])
    parser.add_argument("--domain", type=str, default="stem",
                        choices=["stem", "pharmacy", "finance", "all"])
    parser.add_argument("--config", type=str, default=None,
                        help="Path to lora_config.yaml")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    lora_config = DEFAULT_LORA_CONFIG.copy()
    training_config = DEFAULT_TRAINING_CONFIG.copy()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        lora_config.update(cfg.get("lora", {}))
        training_config.update(cfg.get("training", {}))

    train(
        base_model=args.base_model,
        dataset_name=args.dataset,
        domain=args.domain,
        lora_config=lora_config,
        training_config=training_config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
