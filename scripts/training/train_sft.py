#!/usr/bin/env python3
"""SFT Training script for PDDL models - Simple PyTorch version."""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=15)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--disable_eval", action="store_true", help="Disable evaluation to save memory")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_name}")
    print(f"Output dir: {args.output_dir}")

    # Model setup - use bfloat16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from {args.train_file}")

    # Dataset
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "test": args.eval_file},
    )

    print(f'Train samples: {len(dataset["train"])}')
    print(f'Eval samples: {len(dataset["test"])}')

    # Training config - simple version without deepspeed
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no" if args.disable_eval else "steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        max_length=args.max_seq_length,  # Max sequence length for training
        packing=False,  # Disable packing for simplicity
        push_to_hub=False,
        save_safetensors=True,
        report_to=["tensorboard"],
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
    )

    def formatting_func(example):
        """Format messages into a single string for training."""
        messages = example["messages"]
        # Use the chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            # Fallback: simple concatenation
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text += f"<|{role}|>\n{content}\n"
            return text

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
