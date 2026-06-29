"""
Train Llama-3.1-8B with DPO using its own generated samples.

This implements Direct Preference Optimization to improve the model's
reasoning by learning from preference pairs.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/work/markhsp/hsp_pddl/checkpoints/Meta-Llama-3.1-8B-Instruct-llama4-cot",
        metadata={"help": "Path to pretrained model or checkpoint"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"}
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default="/work/markhsp/hsp_pddl/data/training/dpo_data/llama31_8b/train_dpo.jsonl",
        metadata={"help": "Path to training data"}
    )
    eval_file: str = field(
        default="/work/markhsp/hsp_pddl/data/training/dpo_data/llama31_8b/eval_dpo.jsonl",
        metadata={"help": "Path to evaluation data"}
    )


def format_dpo_example(example, tokenizer):
    """
    Format DPO example for training.

    DPO requires:
    - prompt: The input prompt
    - chosen: The preferred response
    - rejected: The less preferred response
    """
    # The data already has prompt, chosen, rejected
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("="*80)
    print("Llama-3.1-8B DPO Training")
    print("="*80)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Train data: {data_args.train_file}")
    print(f"Eval data: {data_args.eval_file}")
    print(f"Output: {training_args.output_dir}")
    print("="*80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for DPO

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_args.train_file,
            "test": data_args.eval_file
        }
    )

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Eval examples: {len(dataset['test'])}")

    # Format dataset for DPO
    print("\nFormatting dataset...")
    dataset = dataset.map(
        lambda x: format_dpo_example(x, tokenizer),
        remove_columns=[col for col in dataset["train"].column_names
                       if col not in ["prompt", "chosen", "rejected"]],
        desc="Formatting for DPO"
    )

    # Load model
    print("\nLoading model...")

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if model_args.use_flash_attention_2:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            print("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Load reference model (for DPO)
    print("\nLoading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Setup trainer
    print("\nInitializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting DPO training...")
    print("="*80)
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    print(f"\nTraining complete! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
