#!/usr/bin/env python3
"""
Step 3: Validate extracted actions using PDDL solver and compute analytics.

This script reads raw_response.json and extracted_response.json, validates
the extracted action sequences using the PDDL simulator, and computes
metrics. No GPU required.

Input: raw_response.json, extracted_response.json
Output: processed_response.json with fields:
    - extracted (1.0 or 0.0): whether a valid list was extracted
    - legal (1.0 or 0.0): whether all actions are valid format
    - valid_plan (1.0 or 0.0): whether plan executes fully in simulator
    - lcs_score: longest common subsequence score
    - progress_score: correct prefix score
    - state_score: state-based goal achievement score
    - success (1.0 or 0.0): goal state achieved (state_score == 1.0)
"""

import argparse
import os
import json
import re
import sys
from pathlib import Path

# Allow running without `pip install -e .` by exposing src/ on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from plan_llm.evaluation import (
    compute_lcs_score,
    compute_action_score,
    process_actions,
    compute_logical_divergence,
)


def is_valid_action(action):
    """Check if action string is a valid action format: action_name(arg1, arg2, ...)"""
    if not isinstance(action, str):
        return False
    pattern = r'^[a-z_\-]+\([^)]*\)$'
    return bool(re.match(pattern, action.strip(), re.IGNORECASE))


def is_extracted(extracted_answer):
    """Check if answer was successfully extracted (non-empty list of strings)"""
    if not isinstance(extracted_answer, list):
        return False
    if len(extracted_answer) == 0:
        return False
    return all(isinstance(a, str) for a in extracted_answer)


def is_legal(extracted_answer):
    """Check if all actions in the sequence are valid action format"""
    if not is_extracted(extracted_answer):
        return False
    return all(is_valid_action(a) for a in extracted_answer)


def parse_action_to_list(action_str):
    """Convert action string like 'pick-up(A, B)' to list ['pick-up', 'A', 'B']"""
    if not isinstance(action_str, str):
        return None

    # Match action_name(args)
    match = re.match(r'^([a-z_\-]+)\(([^)]*)\)$', action_str.strip(), re.IGNORECASE)
    if not match:
        return None

    action_name = match.group(1)
    args_str = match.group(2)

    if args_str.strip():
        args = [a.strip().strip('"\'') for a in args_str.split(',')]
        args = [a for a in args if a]
    else:
        args = []

    return [action_name] + args


def validate_and_score(args):
    """
    Validate extracted actions and compute metrics.
    """
    # Load data from extracted_response.json (which contains all raw fields + extracted_answer)
    extracted_file = os.path.join(args.input_dir, "extracted_response.json")

    with open(extracted_file, 'r') as f:
        extracted_data = json.load(f)

    print(f"Loaded {len(extracted_data)} samples")

    # Process each sample
    processed_results = []
    stats = {
        'total': len(extracted_data),
        'extracted': 0,
        'legal': 0,
        'valid_plan': 0,
        'success': 0,
        'total_lcs': 0,
        'total_progress': 0,
        'total_state_score': 0,
    }

    for i, item in enumerate(extracted_data):
        gt_actions = item.get('gt_actions', [])
        extracted_answer = item.get('extracted_answer', [])
        initial_state = item.get('initial_state', [])
        goal_state = item.get('goal_state', [])

        # Check extraction and legality
        extracted = 1.0 if is_extracted(extracted_answer) else 0.0
        legal = 1.0 if is_legal(extracted_answer) else 0.0

        if extracted == 1.0:
            stats['extracted'] += 1
        if legal == 1.0:
            stats['legal'] += 1

        # Convert extracted_answer (list of strings) to list of lists format
        # e.g., ['pick-up(9, 1, 0)'] -> [['pick-up', '9', '1', '0']]
        extracted_as_lists = []
        if legal == 1.0 and extracted_answer:
            for action_str in extracted_answer:
                action_list = parse_action_to_list(action_str)
                if action_list:
                    extracted_as_lists.append(action_list)

        # Compute LCS and progress scores against ground truth (both in list of lists format)
        if extracted_as_lists:
            try:
                lcs_score = compute_lcs_score(gt_actions, extracted_as_lists)
                progress_score = compute_action_score(gt_actions, extracted_as_lists)
            except Exception as e:
                print(f"Warning: Error computing metrics for sample {i}: {e}")
                lcs_score = 0.0
                progress_score = 0.0
        else:
            lcs_score = 0.0
            progress_score = 0.0

        stats['total_lcs'] += lcs_score
        stats['total_progress'] += progress_score

        # Validate plan execution using PDDL simulator
        valid_plan = 0.0
        state_score = 0.0

        if extracted_as_lists:
            try:
                # Convert states to sets of tuples for simulator
                initial_state_set = {tuple(p) for p in initial_state}
                goal_state_set = {tuple(p) for p in goal_state}

                # Run simulator with extracted_as_lists (already in list of lists format)
                action_valid, len_executed, len_total, last_state, reason = process_actions(
                    initial_state_set, extracted_as_lists
                )

                if action_valid is not False and len_executed == len_total:
                    valid_plan = 1.0
                    stats['valid_plan'] += 1
                    # Compute state-based score
                    state_score = compute_logical_divergence(goal_state_set, last_state)
                elif action_valid is not False and len_executed > 0:
                    # Partial execution - compute progress
                    state_score = compute_logical_divergence(goal_state_set, last_state)

            except Exception as e:
                print(f"Warning: Simulator error for sample {i}: {e}")
                valid_plan = 0.0
                state_score = 0.0

        stats['total_state_score'] += state_score

        # Success = goal state achieved (state_score == 1.0)
        success = 1.0 if state_score == 1.0 else 0.0
        if success == 1.0:
            stats['success'] += 1

        # Include all fields from extracted_response.json plus new metrics
        result = dict(item)  # Copy all fields from extracted data
        result["extracted"] = extracted
        result["legal"] = legal
        result["valid_plan"] = valid_plan
        result["lcs_score"] = lcs_score
        result["progress_score"] = progress_score
        result["state_score"] = state_score
        result["success"] = success
        processed_results.append(result)

    # Save results
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    processed_file = os.path.join(output_dir, "processed_response.json")
    with open(processed_file, 'w') as f:
        json.dump(processed_results, f, indent=2)

    # Display statistics
    n = stats['total']
    print(f"\n{'='*60}")
    print(f"Step 3: Validation & Analytics Complete")
    print(f"{'='*60}")
    print(f"Samples: {n}")
    print(f"Extracted: {stats['extracted']}/{n} ({stats['extracted']/n*100:.1f}%)")
    print(f"Legal: {stats['legal']}/{n} ({stats['legal']/n*100:.1f}%)")
    print(f"Valid Plan: {stats['valid_plan']}/{n} ({stats['valid_plan']/n*100:.1f}%)")
    print(f"Success (goal achieved): {stats['success']}/{n} ({stats['success']/n*100:.1f}%)")
    print(f"Avg LCS Score: {stats['total_lcs']/n:.4f}")
    print(f"Avg Progress Score: {stats['total_progress']/n:.4f}")
    print(f"Avg State Score: {stats['total_state_score']/n:.4f}")
    print(f"Output: {processed_file}")
    print(f"{'='*60}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Validate actions with PDDL solver and compute metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw_response.json and extracted_response.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input_dir)")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    validate_and_score(args)
