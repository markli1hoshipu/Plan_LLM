#!/usr/bin/env python3
"""
Step 2 (Pass@k): Extract action sequences from k raw responses using REGEX only.

This script uses regex patterns to extract structured action sequences from all k raw
responses per problem. No LLM is used - just pattern matching.

Input: raw_response.json with raw_responses (list of k responses)
Output: extracted_response.json with extracted_answers (list of k extracted answers)
"""

import argparse
import os
import json
import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_final_tag(response):
    """Extract content from <FINAL>...</FINAL> tag."""
    # Try to find <FINAL>...</FINAL> pattern
    pattern = r'<FINAL>\s*(.*?)\s*</FINAL>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_action_list(content):
    """
    Parse action list from various formats:
    - [[action, arg1, arg2], ...]
    - [action(arg1, arg2), ...]
    - action(arg1, arg2)\naction(arg1, arg2)
    """
    if not content:
        return []

    actions = []

    # Try parsing as list of lists: [[action, arg1, arg2], ...]
    try:
        # Handle Python-style lists (no quotes on elements)
        # Convert to valid JSON by adding quotes
        normalized = content.strip()

        # Check if it looks like a list of lists
        if normalized.startswith('[[') or normalized.startswith('['):
            # Try to parse action(arg1, arg2) style first
            action_pattern = r'([a-z_\-]+)\s*\(\s*([^)]*)\s*\)'
            func_matches = re.findall(action_pattern, normalized, re.IGNORECASE)
            if func_matches:
                for action_name, args_str in func_matches:
                    if args_str.strip():
                        args = [a.strip().strip('"\'') for a in args_str.split(',')]
                        args = [a for a in args if a]
                        action_str = f"{action_name}({', '.join(args)})"
                    else:
                        action_str = f"{action_name}()"
                    actions.append(action_str)
                return actions

            # Try to parse [[action, arg1, arg2], ...] style
            # Find all inner brackets
            inner_pattern = r'\[\s*([^\[\]]+?)\s*\]'
            inner_matches = re.findall(inner_pattern, normalized)

            for inner in inner_matches:
                # Split by comma, handling potential commas in strings
                parts = [p.strip().strip('"\'') for p in inner.split(',')]
                parts = [p for p in parts if p]

                if parts:
                    action_name = parts[0]
                    args = parts[1:] if len(parts) > 1 else []
                    if args:
                        action_str = f"{action_name}({', '.join(args)})"
                    else:
                        action_str = f"{action_name}()"
                    actions.append(action_str)

            return actions
    except Exception:
        pass

    # Try parsing line by line for action(arg1, arg2) format
    action_pattern = r'([a-z_\-]+)\s*\(\s*([^)]*)\s*\)'
    matches = re.findall(action_pattern, content, re.IGNORECASE)

    for action_name, args_str in matches:
        if args_str.strip():
            args = [a.strip().strip('"\'') for a in args_str.split(',')]
            args = [a for a in args if a]
            action_str = f"{action_name}({', '.join(args)})"
        else:
            action_str = f"{action_name}()"
        actions.append(action_str)

    return actions


def extract_from_response(response):
    """Extract action sequence from a single response using regex."""
    if not response:
        return []

    # First try to find <FINAL> tag
    final_content = extract_final_tag(response)
    if final_content:
        actions = parse_action_list(final_content)
        if actions:
            return actions

    # If no FINAL tag, try to find action sequences in the response
    # Look for common patterns at the end of responses

    # Try to find a list near the end
    list_pattern = r'\[\s*\[.*?\]\s*\]'
    lists = re.findall(list_pattern, response, re.DOTALL)
    if lists:
        # Use the last list found (usually the final answer)
        actions = parse_action_list(lists[-1])
        if actions:
            return actions

    # Try to find action sequence patterns
    action_pattern = r'([a-z_\-]+)\s*\(\s*([^)]*)\s*\)'
    matches = re.findall(action_pattern, response, re.IGNORECASE)

    if matches:
        actions = []
        for action_name, args_str in matches:
            # Filter out common non-action patterns
            if action_name.lower() in ['print', 'len', 'range', 'str', 'int', 'list', 'dict', 'set', 'tuple', 'type', 'format']:
                continue
            if args_str.strip():
                args = [a.strip().strip('"\'') for a in args_str.split(',')]
                args = [a for a in args if a]
                action_str = f"{action_name}({', '.join(args)})"
            else:
                action_str = f"{action_name}()"
            actions.append(action_str)
        return actions

    return []


def extract_answers(args):
    """
    Extract action sequences from k raw responses per problem using regex.
    """
    # Load raw responses
    raw_file = os.path.join(args.input_dir, "raw_response.json")
    with open(raw_file, 'r') as f:
        raw_data = json.load(f)

    # Determine k from data
    k = len(raw_data[0].get('raw_responses', []))
    total_extractions = len(raw_data) * k

    print(f"Loaded {len(raw_data)} problems from {raw_file}")
    print(f"Each problem has {k} responses, total extractions: {total_extractions}")
    print(f"Using REGEX extraction (no LLM)")

    # Process each problem
    extracted_results = []

    for prob_idx, item in enumerate(raw_data):
        extracted_answers = []

        for sample_idx, raw_response in enumerate(item.get('raw_responses', [])):
            extracted = extract_from_response(raw_response)
            extracted_answers.append(extracted)

        # Copy all fields from raw data and add extracted_answers
        result = dict(item)
        result["extracted_answers"] = extracted_answers
        extracted_results.append(result)

    # Save results
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    extracted_file = os.path.join(output_dir, "extracted_response.json")
    with open(extracted_file, 'w') as f:
        json.dump(extracted_results, f, indent=2)

    # Display statistics
    total_extracted = sum(
        1 for r in extracted_results
        for ans in r['extracted_answers']
        if ans
    )
    extraction_rate = total_extracted / total_extractions if total_extractions else 0

    # Problems with at least one successful extraction
    problems_with_extraction = sum(
        1 for r in extracted_results
        if any(ans for ans in r['extracted_answers'])
    )

    print(f"\n{'='*60}")
    print(f"Step 2 (Pass@{k}): REGEX Extraction Complete")
    print(f"{'='*60}")
    print(f"Problems: {len(extracted_results)}")
    print(f"Samples per problem: {k}")
    print(f"Total extractions: {total_extracted}/{total_extractions} ({extraction_rate*100:.1f}%)")
    print(f"Problems with >=1 extraction: {problems_with_extraction}/{len(extracted_results)} ({problems_with_extraction/len(extracted_results)*100:.1f}%)")
    print(f"Output: {extracted_file}")
    print(f"{'='*60}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Step 2 (Pass@k): Extract answers from k raw responses using REGEX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw_response.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input_dir)")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    extract_answers(args)
