"""
Merge LoRA Adapter — Merge trained LoRA weights back into the base model.
"""

import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_adapter(base_model_path: str, adapter_path: str, output_path: str,
                  push_to_hub: bool = False, hub_name: str = None):
    """Merge LoRA adapter into base model and save."""
    logger.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    logger.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if push_to_hub and hub_name:
        logger.info(f"Pushing to Hub: {hub_name}")
        model.push_to_hub(hub_name)
        tokenizer.push_to_hub(hub_name)

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-name", type=str, default=None)
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter, args.output,
                  args.push_to_hub, args.hub_name)


if __name__ == "__main__":
    main()
