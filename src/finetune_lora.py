"""
PEFT/LoRA Fine-Tuning Module
=============================
Parameter-efficient fine-tuning of open-source LLMs on curated
domain-specific instruction data using LoRA/QLoRA.

Supports:
- LoRA and QLoRA (4-bit NF4 quantization)
- Multiple base models (Llama-3, Mistral, Phi, etc.)
- Domain-filtered training from OASST1, Dolly-15K, and custom data
- Wandb logging and checkpoint management
"""

import os
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Domain keyword filters for extracting relevant subsets
DOMAIN_KEYWORDS = {
    "stem": [
        "physics", "chemistry", "biology", "mathematics", "calculus",
        "algebra", "geometry", "thermodynamics", "quantum", "organic",
        "inorganic", "molecular", "equation", "formula", "theorem",
        "proof", "derivative", "integral", "vector", "matrix",
    ],
    "pharmacy": [
        "drug", "medication", "pharmaceutical", "pharmacology", "dosage",
        "prescription", "side effect", "contraindication", "mechanism of action",
        "bioavailability", "half-life", "metabolite", "enzyme inhibitor",
        "receptor", "agonist", "antagonist", "therapeutic", "clinical trial",
    ],
    "finance": [
        "revenue", "profit", "loss", "balance sheet", "income statement",
        "cash flow", "valuation", "dcf", "p/e ratio", "ebitda",
        "depreciation", "amortization", "equity", "debt", "bond",
        "interest rate", "dividend", "portfolio", "hedge", "derivative",
    ],
}

CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful domain expert assistant. Provide accurate, detailed answers grounded in domain knowledge.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>"""


@dataclass
class FinetuneConfig:
    """Fine-tuning configuration loaded from YAML."""
    # Model
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 2048
    fp16: bool = True
    bf16: bool = False

    # Data
    dataset: str = "oasst1"
    domain: str = "stem"
    max_samples: int = 5000

    # Output
    output_dir: str = "./checkpoints"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


def load_domain_data(
    dataset_name: str,
    domain: str,
    max_samples: int = 5000,
) -> Dataset:
    """
    Load and filter instruction-tuning data for a specific domain.

    Supports: oasst1, dolly, combined
    """
    datasets_to_load = []

    if dataset_name in ("oasst1", "combined"):
        logger.info("Loading OpenAssistant OASST1...")
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        # Filter to English, top-level prompter messages with assistant replies
        oasst_filtered = oasst.filter(
            lambda x: x["lang"] == "en" and x["role"] == "prompter" and x["parent_id"] is None
        )
        datasets_to_load.append(("oasst1", oasst_filtered))

    if dataset_name in ("dolly", "combined"):
        logger.info("Loading Databricks Dolly 15K...")
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        datasets_to_load.append(("dolly", dolly))

    # Apply domain keyword filtering
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        logger.warning(f"No keywords for domain '{domain}', using all data")

    filtered_items = []
    for source_name, ds in datasets_to_load:
        for row in ds:
            text = ""
            if source_name == "oasst1":
                text = row.get("text", "").lower()
            elif source_name == "dolly":
                text = (row.get("instruction", "") + " " + row.get("response", "")).lower()

            if not keywords or any(kw in text for kw in keywords):
                if source_name == "oasst1":
                    filtered_items.append({
                        "instruction": row["text"],
                        "response": "",  # will need pairing
                        "source": "oasst1",
                    })
                elif source_name == "dolly":
                    instruction = row["instruction"]
                    if row.get("context"):
                        instruction = f"{instruction}\n\nContext: {row['context']}"
                    filtered_items.append({
                        "instruction": instruction,
                        "response": row["response"],
                        "source": "dolly",
                    })

    # Filter out items without responses (OASST needs pairing separately)
    filtered_items = [item for item in filtered_items if item["response"]]

    if len(filtered_items) > max_samples:
        import random
        random.seed(42)
        filtered_items = random.sample(filtered_items, max_samples)

    logger.info(
        f"Loaded {len(filtered_items)} domain-filtered samples "
        f"(domain={domain}, source={dataset_name})"
    )

    return Dataset.from_list(filtered_items)


def format_instruction(example: dict) -> dict:
    """Format a single example into the chat template."""
    text = CHAT_TEMPLATE.format(
        instruction=example["instruction"],
        response=example["response"],
    )
    return {"text": text}


def create_model_and_tokenizer(config: FinetuneConfig):
    """Initialize quantized model and tokenizer."""
    bnb_config = None
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def create_lora_model(model, config: FinetuneConfig):
    """Apply LoRA adapter to the model."""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied — Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model


def train(config: FinetuneConfig):
    """Run the full fine-tuning pipeline."""
    # Load data
    dataset = load_domain_data(config.dataset, config.domain, config.max_samples)
    dataset = dataset.map(format_instruction)

    # Split train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load model
    model, tokenizer = create_model_and_tokenizer(config)
    model = create_lora_model(model, config)

    # Training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # set to "wandb" for logging
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Model saved to {final_path}")

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PEFT/LoRA Fine-Tuning")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    parser.add_argument("--dataset", default="dolly", choices=["oasst1", "dolly", "combined"])
    parser.add_argument("--domain", default="stem", choices=["stem", "pharmacy", "finance"])
    parser.add_argument("--output_dir", default="./checkpoints/llama3-lora")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    if Path(args.config).exists():
        config = FinetuneConfig.from_yaml(args.config)
    else:
        config = FinetuneConfig()

    # Override with CLI args
    config.model_name = args.model_name
    config.dataset = args.dataset
    config.domain = args.domain
    config.output_dir = args.output_dir
    config.max_samples = args.max_samples

    train(config)
