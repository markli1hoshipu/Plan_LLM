"""
Prepare DPO training data from generated samples.

Creates preference pairs where:
- Chosen: Sample with highest reward
- Rejected: Sample with lower reward

For DPO training format.
"""

import os
import sys
import json
import jsonlines
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# Allow running without `pip install -e .` by exposing src/ on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


@dataclass
class Args:
    samples_file: str = field(
        default="/work/markhsp/hsp_pddl/data/processed/rl_samples/qwen3_4b/samples.jsonl",
        metadata={"help": "Path to samples file"}
    )
    output_dir: str = field(
        default="/work/markhsp/hsp_pddl/data/processed/dpo_data/qwen3_4b",
        metadata={"help": "Output directory for DPO data"}
    )
    min_reward_diff: float = field(
        default=0.0,
        metadata={"help": "Minimum reward difference for pairs"}
    )


def create_preference_pairs(problem_data, min_reward_diff=0.0):
    """
    Create preference pairs from samples.

    Strategy:
    - Best sample (highest reward) as chosen
    - All worse samples as rejected
    """
    pairs = []
    samples = problem_data["samples"]

    if len(samples) < 2:
        return pairs

    # Get best sample
    best_sample = samples[0]  # Already sorted by reward
    best_reward = best_sample["reward"]

    # Create pairs with all worse samples
    for i in range(1, len(samples)):
        rejected_sample = samples[i]
        rejected_reward = rejected_sample["reward"]

        # Check if reward difference is sufficient
        if best_reward - rejected_reward >= min_reward_diff:
            pairs.append({
                "prompt": problem_data["prompt"],
                "chosen": best_sample["generated_text"],
                "rejected": rejected_sample["generated_text"],
                "chosen_reward": best_reward,
                "rejected_reward": rejected_reward,
                "reward_margin": best_reward - rejected_reward,
                "problem_id": problem_data["problem_id"],
                "static_state": problem_data["static_state"],
                "dynamic_states": problem_data["dynamic_states"]
            })

    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", type=str,
                       default="/work/markhsp/hsp_pddl/data/processed/rl_samples/qwen3_4b/samples.jsonl")
    parser.add_argument("--output_dir", type=str,
                       default="/work/markhsp/hsp_pddl/data/processed/dpo_data/qwen3_4b")
    parser.add_argument("--min_reward_diff", type=float, default=0.0)
    args = parser.parse_args()

    print("="*80)
    print("Preparing DPO Training Data")
    print("="*80)
    print(f"Samples file: {args.samples_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Min reward diff: {args.min_reward_diff}")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load samples
    print("\nLoading samples...")
    with jsonlines.open(args.samples_file, 'r') as f:
        samples_data = list(f)

    print(f"Loaded {len(samples_data)} problems")

    # Create preference pairs
    print("\nCreating preference pairs...")
    all_pairs = []

    for problem_data in samples_data:
        pairs = create_preference_pairs(problem_data, args.min_reward_diff)
        all_pairs.extend(pairs)

    print(f"Created {len(all_pairs)} preference pairs")

    # Statistics
    reward_margins = [p["reward_margin"] for p in all_pairs]
    avg_margin = sum(reward_margins) / len(reward_margins) if reward_margins else 0

    correct_chosen = sum(1 for p in all_pairs if p["chosen_reward"] == 1.0)
    correct_rejected = sum(1 for p in all_pairs if p["rejected_reward"] == 1.0)

    print(f"\n{'='*80}")
    print("DPO Data Statistics")
    print(f"{'='*80}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Average reward margin: {avg_margin:.4f}")
    print(f"Correct chosen: {correct_chosen} ({correct_chosen/len(all_pairs)*100:.2f}%)")
    print(f"Correct rejected: {correct_rejected} ({correct_rejected/len(all_pairs)*100:.2f}%)")
    print(f"{'='*80}")

    # Split into train/eval (90/10)
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]

    print(f"\nTrain pairs: {len(train_pairs)}")
    print(f"Eval pairs: {len(eval_pairs)}")

    # Save DPO data
    train_file = os.path.join(args.output_dir, "train_dpo.jsonl")
    eval_file = os.path.join(args.output_dir, "eval_dpo.jsonl")

    print(f"\nSaving train data to {train_file}...")
    with jsonlines.open(train_file, 'w') as f:
        f.write_all(train_pairs)

    print(f"Saving eval data to {eval_file}...")
    with jsonlines.open(eval_file, 'w') as f:
        f.write_all(eval_pairs)

    print(f"\nDone! DPO data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
