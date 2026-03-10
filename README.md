# Adversarial Evaluation & PEFT-Based Fine-Tuning for Domain-Specific LLM QA

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/)

An end-to-end pipeline for evaluating and improving LLM performance on domain-specific tasks through adversarial prompt engineering, parameter-efficient fine-tuning (PEFT/LoRA), and rubric-driven quality assessment.

## Overview

Standard LLM benchmarks often fail to capture real-world performance in specialized domains. This project addresses three key gaps:

1. **Adversarial Evaluation** — Systematic identification of failure modes in frontier models (GPT-4, Claude, Llama-3) through controlled perturbation of domain-specific prompts
2. **PEFT/LoRA Fine-Tuning** — Parameter-efficient adaptation of open-source models using curated instruction-tuning data to improve domain accuracy in STEM, pharmacy, and financial analysis
3. **Rubric-Driven QA** — Structured scoring framework (D-B-A-S rubric) to benchmark model outputs against expert baselines across correctness, clarity, depth, and safety dimensions

## Key Results

| Model | Domain | Base Accuracy | Post-LoRA Accuracy | Adversarial Robustness |
|-------|--------|:---:|:---:|:---:|
| Llama-3-8B | STEM (Physics) | 62.4% | 78.1% (+15.7) | 71.3% |
| Mistral-7B-v0.3 | Pharmacy | 54.8% | 73.6% (+18.8) | 66.9% |
| Llama-3-8B | Financial Analysis | 58.2% | 74.5% (+16.3) | 68.7% |
| GPT-4 (baseline) | STEM (Physics) | 89.1% | — | 76.2% |
| Claude-3 (baseline) | STEM (Physics) | 87.6% | — | 79.4% |

> **Adversarial Robustness** = accuracy on perturbed variants of the same questions (surface-level changes preserving logical structure). Even GPT-4 shows a 12.9% drop, confirming that high benchmark scores can mask fragile reasoning.

## Project Structure

```
llm-domain-eval/
├── src/
│   ├── evaluation/
│   │   ├── benchmark_runner.py      # Run models against standard benchmarks
│   │   ├── rubric_scorer.py         # D-B-A-S rubric-based evaluation
│   │   └── metrics.py               # Custom metrics (robustness score, etc.)
│   ├── fine_tuning/
│   │   ├── lora_trainer.py          # PEFT/LoRA training pipeline
│   │   ├── data_prep.py             # Dataset loading and preprocessing
│   │   └── merge_adapter.py         # Merge LoRA weights into base model
│   └── adversarial/
│       ├── perturbation_engine.py   # Controlled prompt perturbation generator
│       ├── failure_taxonomy.py      # Classify and categorize failure modes
│       └── templates.py             # Adversarial template definitions
├── configs/
│   ├── lora_config.yaml             # LoRA hyperparameters
│   ├── eval_config.yaml             # Evaluation settings
│   └── adversarial_config.yaml      # Perturbation parameters
├── results/
│   ├── evaluation_summary.json      # Aggregated results
│   └── adversarial_analysis.md      # Detailed failure mode analysis
├── notebooks/
│   └── analysis.ipynb               # Result visualization and analysis
├── data/
│   └── README.md                    # Dataset sources and download instructions
├── requirements.txt
├── setup.py
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/llm-domain-eval.git
cd llm-domain-eval
pip install -r requirements.txt
```

## Quick Start

### 1. Run Evaluation on a Benchmark

```bash
python -m src.evaluation.benchmark_runner \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset truthfulqa \
    --config configs/eval_config.yaml
```

### 2. Generate Adversarial Perturbations

```bash
python -m src.adversarial.perturbation_engine \
    --input data/stem_physics_questions.jsonl \
    --output data/stem_physics_perturbed.jsonl \
    --config configs/adversarial_config.yaml
```

### 3. Fine-Tune with LoRA

```bash
python -m src.fine_tuning.lora_trainer \
    --base-model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset oasst1 \
    --domain stem \
    --config configs/lora_config.yaml \
    --output-dir checkpoints/llama3-stem-lora
```

### 4. Score Outputs with D-B-A-S Rubric

```bash
python -m src.evaluation.rubric_scorer \
    --predictions results/model_outputs.jsonl \
    --rubric dbas \
    --output results/rubric_scores.json
```

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| OpenAssistant OASST1 | 88,838 human-annotated messages, 35 languages | [HuggingFace](https://huggingface.co/datasets/OpenAssistant/oasst1) |
| Databricks Dolly 15K | 15K human-authored instruction-following pairs | [HuggingFace](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| TruthfulQA | 817 questions testing factual accuracy / hallucination | [HuggingFace](https://huggingface.co/datasets/truthfulqa/truthful_qa) |
| MMLU | 57-subject multiple-choice knowledge evaluation | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |
| GSM8K | 8.5K grade-school math word problems | [HuggingFace](https://huggingface.co/datasets/openai/gsm8k) |

## D-B-A-S Rubric Framework

| Dimension | Score Range | Description |
|-----------|:-----------:|-------------|
| **D**epth | 1–5 | Thoroughness of reasoning, edge case coverage |
| **B**readth | 1–5 | Coverage of relevant concepts and context |
| **A**ccuracy | 1–5 | Factual correctness and logical consistency |
| **S**afety | 1–5 | Absence of harmful, misleading, or biased content |

## Adversarial Perturbation Types

| Type | Description | Example |
|------|-------------|---------|
| **Numerical Swap** | Change numerical values | "5 kg" → "7.3 kg" |
| **Context Injection** | Add irrelevant plausible context | Insert unrelated backstory |
| **Paraphrase** | Rephrase preserving semantics | Active ↔ passive voice |
| **Negation Flip** | Invert boolean conditions | "which IS" → "which is NOT" |
| **Domain Transfer** | Same structure, different vocabulary | Physics → Chemistry framing |

## Fine-Tuning Configuration

- **Method:** LoRA (rank=16, alpha=32, dropout=0.05)
- **Target Modules:** q_proj, v_proj, k_proj, o_proj
- **Training:** 3 epochs, lr=2e-4, cosine scheduler with warmup
- **Quantization:** 4-bit QLoRA via bitsandbytes
- **Effective Batch Size:** 32 (batch=4, gradient accumulation=8)

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
- Mirzadeh et al., "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs" ([arXiv:2410.05229](https://arxiv.org/abs/2410.05229))
- Köpf et al., "OpenAssistant Conversations — Democratizing LLM Alignment" ([arXiv:2304.07327](https://arxiv.org/abs/2304.07327))
- Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods" ([arXiv:2109.07958](https://arxiv.org/abs/2109.07958))

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

**Shubham Sunder Kadam** — Data Scientist @ KPMG India | IIT Bombay (Dual Degree, Aerospace Engineering) | AI Evaluation & NLP
