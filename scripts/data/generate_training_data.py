#!/usr/bin/env python3
"""
Generate training data from CoT-enriched JSONL files.

This script converts CoT-enriched planning problems into training format
for fine-tuning language models. Based on filter_train.py from pddl_sft_rl.

Usage:
    python scripts/generate_training_data.py --help
    python scripts/generate_training_data.py --cot-model Qwen3-30B-A3B-Thinking-2507
"""

import json
import random
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts" / "assets"))

from prompt import (
    system_prompt,
    user_prompt_dynamic,
    user_prompt_short,
    response_ans,
    response_cot
)


def get_random_problem_setting():
    """Return one random rephrased problem setting."""
    jsonl_file = PROJECT_ROOT / "scripts" / "assets" / "problem_setting.jsonl"

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = json.loads(random.choice(lines))
    random_entry = random.choice(data["trephrase_problem_setting"])
    return random_entry["rephrase_problem_setting"]


def canonicalise_state(state):
    """
    Given a state as a list of facts, e.g.
    [["on-table","7","1"],["top","2"],...]
    returns a deterministically sorted version
    where the outer list of facts is sorted
    but each fact's internal order is preserved.
    """
    # convert each fact to a tuple so it can be compared
    facts_as_tuples = [tuple(f) for f in state]
    # sort by tuple lexicographically
    facts_sorted = sorted(facts_as_tuples)
    # convert back to lists for JSON dump
    return [list(f) for f in facts_sorted]


def split_and_format_combined_filtered(
    input_paths,
    train_output_path,
    eval_output_path,
    test_paths=None,
    existing_train_paths=None,
    eval_ratio=0.1,
    seed=42
):
    """
    Combine multiple JSONL input files, shuffle, split into train/eval once,
    exclude any example whose (dynamic_states[0], dynamic_states[-1]) appears in test_paths,
    format and save.

    Args:
        input_paths: List of CoT-enriched JSONL files to use for training
        train_output_path: Path to save training data
        eval_output_path: Path to save evaluation data
        test_paths: List of JSONL files to exclude from training (test set)
        existing_train_paths: List of existing training JSONL files to merge in
        eval_ratio: Fraction of data to use for evaluation
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # 1. Build exclusion set from test_paths
    exclude_keys = set()
    if test_paths:
        for tpath in test_paths:
            if not os.path.exists(tpath):
                print(f"Warning: Test path does not exist: {tpath}")
                continue

            with open(tpath, "r") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        # Canonicalise states
                        state0 = canonicalise_state(d["dynamic_states"][0])
                        state_last = canonicalise_state(d["dynamic_states"][-1])

                        key = (
                            json.dumps(state0, ensure_ascii=False),
                            json.dumps(state_last, ensure_ascii=False),
                        )

                        exclude_keys.add(key)
                    except Exception as e:
                        # skip malformed
                        continue
        print(f"Loaded {len(exclude_keys)} keys to exclude from train data.")

    # 2. Load and combine input data
    data = []
    filtered_count = 0
    kept_count = 0

    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Warning: Input path does not exist: {input_path}")
            continue

        with open(input_path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    state0 = canonicalise_state(d["dynamic_states"][0])
                    state_last = canonicalise_state(d["dynamic_states"][-1])

                    key = (
                        json.dumps(state0, ensure_ascii=False),
                        json.dumps(state_last, ensure_ascii=False),
                    )
                    if key not in exclude_keys:
                        data.append(d)
                        kept_count += 1
                    else:
                        filtered_count += 1
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue

    print(f"Kept {kept_count} examples, filtered out {filtered_count}.")

    # Shuffle data
    random.shuffle(data)

    # 3. Split train/eval
    split_idx = int(len(data) * (1 - eval_ratio))
    train_data, eval_data = data[:split_idx], data[split_idx:]

    def format_entry(d):
        """Format a single example as training data."""
        problem_setting = get_random_problem_setting()
        cot = d.get("cot")
        if not cot:
            return None

        actions = d.get("actions", [])
        if actions and actions[-1] is None:
            actions = actions[:-1]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt_dynamic(
                    problem_setting,
                    d["static_state"],
                    d["dynamic_states"][0],
                    d["dynamic_states"][-1],
                ).replace("'", "").replace('"', "")
            },
            {
                "role": "assistant",
                "content": response_cot(
                    d["dynamic_states"][0],
                    cot,
                    d["dynamic_states"][-1],
                    actions,
                ).replace("'", "").replace('"', "")
            }
        ]
        return {"messages": messages}

    train_data_fmt = [entry for d in train_data if (entry := format_entry(d))]
    eval_data_fmt = [entry for d in eval_data if (entry := format_entry(d))]

    # 3b. Optionally merge in existing training JSONL(s)
    if existing_train_paths:
        merged_existing = []
        for epath in existing_train_paths:
            if not os.path.exists(epath):
                print(f"Warning: Existing train path does not exist: {epath}")
                continue

            with open(epath, "r") as ef:
                for line in ef:
                    try:
                        # existing files may already be formatted (messages)
                        d = json.loads(line)
                        if "messages" in d:  # already formatted
                            merged_existing.append(d)
                        else:  # raw data → format
                            if entry := format_entry(d):
                                merged_existing.append(entry)
                    except Exception:
                        continue
        print(f"Added {len(merged_existing)} existing training examples.")
        train_data_fmt.extend(merged_existing)

    # Create output directories
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)

    # Write output files
    with open(train_output_path, "w") as f:
        for entry in train_data_fmt:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(eval_output_path, "w") as f:
        for entry in eval_data_fmt:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"✅ Combined from {len(input_paths)} files "
        f"→ Train: {len(train_data_fmt)} | Eval: {len(eval_data_fmt)} "
        f"(excluded {len(exclude_keys)} test keys)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from CoT-enriched JSONL files"
    )
    parser.add_argument(
        "--cot-model",
        type=str,
        default="Qwen3-30B-A3B-Thinking-2507",
        help="Name of the CoT model used for generation (directory name)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["align_data", "reorder_data", "stack_data", "unstack_data", "suboptimal_data"],
        help="List of dataset names to include in training"
    )
    parser.add_argument(
        "--test-datasets",
        type=str,
        nargs="+",
        default=["align_data_eval", "reorder_data_eval", "stack_data_eval", "unstack_data_eval"],
        help="List of test dataset names to exclude from training"
    )
    parser.add_argument(
        "--existing-train",
        type=str,
        nargs="*",
        default=None,
        help="Existing training JSONL files to merge in"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="v1",
        help="Suffix for output files (e.g., v1, yx0, yx1)"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    # Direct path arguments (override directory-based logic when specified)
    parser.add_argument(
        "--input-paths",
        type=str,
        nargs="+",
        default=None,
        help="Direct paths to input COT JSONL files (overrides --cot-model/--datasets)"
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default=None,
        help="Direct path for train output JSONL (overrides --output-suffix)"
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Direct path for eval output JSONL (overrides --output-suffix)"
    )
    parser.add_argument(
        "--test-paths",
        type=str,
        nargs="*",
        default=None,
        help="Direct paths to test JSONL files to exclude (overrides --test-datasets)"
    )

    args = parser.parse_args()

    # Determine paths - use direct paths if specified, otherwise use directory-based logic
    if args.input_paths:
        input_files = args.input_paths
    else:
        cot_dir = PROJECT_ROOT / "data" / "processed" / "cot_prep_json" / "cot_generated" / args.cot_model
        input_files = [
            str(cot_dir / f"{name}_cot_with_cot.jsonl")
            for name in args.datasets
        ]

    if args.test_paths is not None:
        test_files = args.test_paths
    else:
        eval_dir = PROJECT_ROOT / "data" / "processed" / "eval"
        test_files = [
            str(eval_dir / f"{name}.jsonl")
            for name in args.test_datasets
        ]

    if args.train_output:
        train_out = args.train_output
    else:
        train_dir = PROJECT_ROOT / "data" / "processed" / "train"
        train_out = str(train_dir / f"train_{args.output_suffix}.jsonl")

    if args.eval_output:
        eval_out = args.eval_output
    else:
        if 'train_dir' not in locals():
            train_dir = PROJECT_ROOT / "data" / "processed" / "train"
        eval_out = str(train_dir / f"eval_{args.output_suffix}.jsonl")

    print("="*80)
    print("PDDL Training Data Generation")
    print("="*80)
    print(f"CoT Model: {args.cot_model}")
    print(f"Input datasets: {args.datasets}")
    print(f"Test datasets (excluded): {args.test_datasets}")
    print(f"Output suffix: {args.output_suffix}")
    print(f"Eval ratio: {args.eval_ratio}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    print()

    # Generate training data
    split_and_format_combined_filtered(
        input_files,
        train_out,
        eval_out,
        test_paths=test_files,
        existing_train_paths=args.existing_train,
        eval_ratio=args.eval_ratio,
        seed=args.seed
    )

    print()
    print("="*80)
    print("Training data generated successfully!")
    print("="*80)
    print(f"Train: {train_out}")
    print(f"Eval: {eval_out}")
    print("="*80)


if __name__ == "__main__":
    main()
