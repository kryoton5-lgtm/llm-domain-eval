# Datasets

Datasets are downloaded on-the-fly via HuggingFace `datasets` library. Place custom domain-specific JSONL files here.

## Sources

| Dataset | URL |
|---------|-----|
| OASST1 | https://huggingface.co/datasets/OpenAssistant/oasst1 |
| Dolly 15K | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| TruthfulQA | https://huggingface.co/datasets/truthfulqa/truthful_qa |
| MMLU | https://huggingface.co/datasets/cais/mmlu |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k |

## Custom Data Format

```json
{"question": "Calculate the kinetic energy of a 5kg object at 10 m/s.", "answer": "250 J", "domain": "stem", "index": 0}
```
