#!/usr/bin/env python3
"""
Extract training data from Pass@4 evaluation results.

This script extracts successful samples (state_score == 1.0) from Pass@4 evaluation
results on training data and formats them for SFT training.

Usage:
    python scripts/extract_pass4_training_data.py \
        --experiment_dir /shared_work/markhsp/se_pddl/experiments/Qwen3-4B-Instruct-2507-base-passk4-train-YYYYMMDD \
        --output_dir /shared_work/markhsp/se_pddl/data/training/pass4_base
"""

import json
import random
import argparse
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent


def format_actions_for_output(actions):
    """Format action list as string for <FINAL> tag."""
    # Actions are already a list like ['pick-up(9, 1, 0)', 'rotate(9, 3, 1, 0)', ...]
    # Convert to the expected format: [[action1], [action2], ...]
    formatted = []
    for action in actions:
        # Parse action string like "pick-up(9, 1, 0)" into ["pick-up", "9", "1", "0"]
        if '(' in action:
            name = action.split('(')[0]
            args = action.split('(')[1].rstrip(')').split(', ')
            formatted.append([name] + args)
        else:
            formatted.append([action])
    return str(formatted)


def extract_successful_samples(experiment_dir, datasets):
    """
    Extract samples with state_score == 1.0 from processed_response.json files.

    Returns:
        List of dicts with: input_prompt, raw_response (full reasoning), dataset
    """
    samples = []

    for dataset in datasets:
        processed_file = os.path.join(experiment_dir, dataset, "processed_response.json")
        if not os.path.exists(processed_file):
            print(f"SKIPPED: {dataset} (processed_response.json not found)")
            continue

        with open(processed_file) as f:
            data = json.load(f)

        dataset_samples = 0
        for problem in data:
            state_scores = problem.get('state_scores', [])
            raw_responses = problem.get('raw_responses', [])
            extracted_answers = problem.get('extracted_answers', [])
            input_prompt = problem.get('input_prompt', '')

            for i, score in enumerate(state_scores):
                # Perfect success and has both raw response and extracted answer
                if score == 1.0 and i < len(raw_responses) and i < len(extracted_answers):
                    if raw_responses[i] and extracted_answers[i]:
                        samples.append({
                            'input_prompt': input_prompt,
                            'raw_response': raw_responses[i],  # Full reasoning + answer
                            'dataset': dataset
                        })
                        dataset_samples += 1

        print(f"{dataset}: {len(data)} problems, {dataset_samples} successful samples")

    return samples


def format_training_sample(sample, system_prompt):
    """
    Format a sample as messages structure for SFT training.

    Args:
        sample: Dict with input_prompt, raw_response (full reasoning)
        system_prompt: System prompt string

    Returns:
        Dict with messages list
    """
    # Use the full raw response (includes reasoning + <FINAL> tags)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample['input_prompt']},
            {"role": "assistant", "content": sample['raw_response']}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Extract training data from Pass@4 results")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to Pass@4 experiment directory containing processed_response.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "training" / "pass4_base"),
        help="Output directory for training data"
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # System prompt (same as in training data generation)
    system_prompt = "You are a robot assistant. Your task is to generate a plan given the initial and goal state. A plan is a sequence of actions."

    # Training datasets (from cot_prep_json)
    datasets = [
        "align_data_cot",
        "stack_data_cot",
        "unstack_data_cot",
        "reorder_data_cot",
        "suboptimal_data_cot",
        "suboptimal_data_extra"
    ]

    print("=" * 60)
    print("Extracting successful samples from Pass@4 results")
    print(f"Experiment dir: {args.experiment_dir}")
    print("=" * 60)

    # Extract successful samples
    samples = extract_successful_samples(args.experiment_dir, datasets)

    if not samples:
        print("ERROR: No successful samples found!")
        return

    print(f"\nTotal successful samples: {len(samples)}")

    # Shuffle and split
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - args.eval_ratio))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")

    # Format samples
    train_data = [format_training_sample(s, system_prompt) for s in train_samples]
    eval_data = [format_training_sample(s, system_prompt) for s in eval_samples]

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train_qwen3_4b_pass4_base.jsonl")
    eval_path = os.path.join(args.output_dir, "eval_qwen3_4b_pass4_base.jsonl")

    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(eval_path, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved training data to: {train_path}")
    print(f"Saved eval data to: {eval_path}")

    # Print dataset distribution in training data
    print("\n" + "=" * 60)
    print("Dataset distribution in training data:")
    print("=" * 60)
    from collections import Counter
    dataset_counts = Counter(s['dataset'] for s in train_samples)
    for ds, count in sorted(dataset_counts.items()):
        print(f"  {ds}: {count}")


if __name__ == "__main__":
    main()
