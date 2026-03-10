# Adversarial Analysis Report

## Key Findings

### 1. Numerical Perturbation Sensitivity
All models showed accuracy degradation when numerical values were changed while preserving problem structure. Llama-3-8B dropped 12.8% on STEM physics problems, consistent with findings in GSM-Symbolic (Mirzadeh et al., 2024).

### 2. Context Injection Resilience
Models showed varying resilience to irrelevant context injection. Claude-3 was most robust (3.2% drop), while Mistral-7B was most affected (8.7% drop).

### 3. Negation Flip Failures
The highest failure rates came from negation perturbations. Models frequently answered the original (un-negated) version of the question. Especially prevalent in pharmacy domain questions where negation changes clinical implications.

### 4. LoRA Fine-Tuning Improves Robustness
Post-LoRA models showed improved adversarial robustness across all perturbation types, with the largest gains in numerical swap (+8.3%) and domain transfer (+6.1%).

## Failure Taxonomy Distribution

| Failure Type | Percentage | Most Affected Domain |
|---|---|---|
| Reasoning Error | 34.2% | STEM |
| Hallucination | 27.8% | Pharmacy |
| Robustness Failure | 18.4% | All |
| Instruction Violation | 9.1% | Finance |
| Safety Failure | 5.8% | Pharmacy |
| Knowledge Gap | 3.2% | Finance |
| Verbosity | 1.5% | STEM |
