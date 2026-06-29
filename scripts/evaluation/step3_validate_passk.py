#!/usr/bin/env python3
"""
Step 3 (Pass@k): Validate k extracted actions per problem and compute pass@k metrics.

This script reads extracted_response.json with k extracted answers per problem,
validates each using the PDDL simulator, and computes pass@k metrics.

Input: extracted_response.json with extracted_answers (list of k answers per problem)
Output: processed_response.json with fields:
    - successes (list of k success values, 1.0 or 0.0)
    - state_scores (list of k state scores)
    - pass_at_1: success of first sample
    - pass_at_k: success if any of k samples succeeds
    - num_successes: count of successful samples out of k
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


def validate_single_answer(extracted_answer, initial_state, goal_state, gt_actions):
    """
    Validate a single extracted answer and return metrics.
    Returns: (extracted, legal, valid_plan, lcs_score, progress_score, state_score, success)
    """
    # Check extraction and legality
    extracted = 1.0 if is_extracted(extracted_answer) else 0.0
    legal = 1.0 if is_legal(extracted_answer) else 0.0

    # Convert to list of lists format
    extracted_as_lists = []
    if legal == 1.0 and extracted_answer:
        for action_str in extracted_answer:
            action_list = parse_action_to_list(action_str)
            if action_list:
                extracted_as_lists.append(action_list)

    # Compute LCS and progress scores
    if extracted_as_lists:
        try:
            lcs_score = compute_lcs_score(gt_actions, extracted_as_lists)
            progress_score = compute_action_score(gt_actions, extracted_as_lists)
        except Exception:
            lcs_score = 0.0
            progress_score = 0.0
    else:
        lcs_score = 0.0
        progress_score = 0.0

    # Validate plan execution using PDDL simulator
    valid_plan = 0.0
    state_score = 0.0

    if extracted_as_lists:
        try:
            initial_state_set = {tuple(p) for p in initial_state}
            goal_state_set = {tuple(p) for p in goal_state}

            action_valid, len_executed, len_total, last_state, reason = process_actions(
                initial_state_set, extracted_as_lists
            )

            if action_valid is not False and len_executed == len_total:
                valid_plan = 1.0
                state_score = compute_logical_divergence(goal_state_set, last_state)
            elif action_valid is not False and len_executed > 0:
                state_score = compute_logical_divergence(goal_state_set, last_state)
        except Exception:
            valid_plan = 0.0
            state_score = 0.0

    success = 1.0 if state_score == 1.0 else 0.0

    return extracted, legal, valid_plan, lcs_score, progress_score, state_score, success


def validate_and_score(args):
    """
    Validate k extracted answers per problem and compute pass@k metrics.
    """
    # Load extracted responses
    extracted_file = os.path.join(args.input_dir, "extracted_response.json")
    with open(extracted_file, 'r') as f:
        extracted_data = json.load(f)

    # Determine k from data
    k = len(extracted_data[0].get('extracted_answers', []))
    print(f"Loaded {len(extracted_data)} problems with {k} samples each")

    # Process each problem
    processed_results = []
    stats = {
        'total_problems': len(extracted_data),
        'total_samples': len(extracted_data) * k,
        'pass_at_1': 0,
        'pass_at_k': 0,
        'total_successes': 0,
        'total_extracted': 0,
        'total_legal': 0,
        'total_state_score': 0,
    }

    for prob_idx, item in enumerate(extracted_data):
        gt_actions = item.get('gt_actions', [])
        initial_state = item.get('initial_state', [])
        goal_state = item.get('goal_state', [])
        extracted_answers = item.get('extracted_answers', [])

        # Validate each of the k samples
        sample_results = []
        successes = []
        state_scores = []

        for sample_idx, extracted_answer in enumerate(extracted_answers):
            extracted, legal, valid_plan, lcs_score, progress_score, state_score, success = \
                validate_single_answer(extracted_answer, initial_state, goal_state, gt_actions)

            sample_results.append({
                'extracted': extracted,
                'legal': legal,
                'valid_plan': valid_plan,
                'lcs_score': lcs_score,
                'progress_score': progress_score,
                'state_score': state_score,
                'success': success,
            })
            successes.append(success)
            state_scores.append(state_score)

            stats['total_extracted'] += extracted
            stats['total_legal'] += legal
            stats['total_state_score'] += state_score
            if success == 1.0:
                stats['total_successes'] += 1

        # Compute pass@1 and pass@k for this problem
        pass_at_1 = 1.0 if successes[0] == 1.0 else 0.0
        pass_at_k = 1.0 if any(s == 1.0 for s in successes) else 0.0
        num_successes = sum(1 for s in successes if s == 1.0)

        if pass_at_1 == 1.0:
            stats['pass_at_1'] += 1
        if pass_at_k == 1.0:
            stats['pass_at_k'] += 1

        # Build result
        result = dict(item)
        result['sample_results'] = sample_results
        result['successes'] = successes
        result['state_scores'] = state_scores
        result['pass_at_1'] = pass_at_1
        result['pass_at_k'] = pass_at_k
        result['num_successes'] = num_successes
        processed_results.append(result)

    # Save results
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    processed_file = os.path.join(output_dir, "processed_response.json")
    with open(processed_file, 'w') as f:
        json.dump(processed_results, f, indent=2)

    # Display statistics
    n_problems = stats['total_problems']
    n_samples = stats['total_samples']

    print(f"\n{'='*70}")
    print(f"Step 3 (Pass@{k}): Validation & Analytics Complete")
    print(f"{'='*70}")
    print(f"Problems: {n_problems}")
    print(f"Samples per problem: {k}")
    print(f"Total samples: {n_samples}")
    print(f"")
    print(f"Per-Sample Metrics:")
    print(f"  Extracted: {stats['total_extracted']:.0f}/{n_samples} ({stats['total_extracted']/n_samples*100:.1f}%)")
    print(f"  Legal: {stats['total_legal']:.0f}/{n_samples} ({stats['total_legal']/n_samples*100:.1f}%)")
    print(f"  Success: {stats['total_successes']}/{n_samples} ({stats['total_successes']/n_samples*100:.1f}%)")
    print(f"  Avg State Score: {stats['total_state_score']/n_samples:.4f}")
    print(f"")
    print(f"Pass@k Metrics:")
    print(f"  Pass@1: {stats['pass_at_1']}/{n_problems} ({stats['pass_at_1']/n_problems*100:.1f}%)")
    print(f"  Pass@{k}: {stats['pass_at_k']}/{n_problems} ({stats['pass_at_k']/n_problems*100:.1f}%)")
    print(f"")
    print(f"Output: {processed_file}")
    print(f"{'='*70}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Step 3 (Pass@k): Validate k answers per problem and compute pass@k metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing extracted_response.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input_dir)")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    validate_and_score(args)
