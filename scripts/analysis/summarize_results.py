#!/usr/bin/env python3
"""
Summarize experiment results across all datasets and models.

Usage:
    python summarize_results.py --experiments_dir experiments
    python summarize_results.py --experiments_dir experiments --model Qwen3-4B-Instruct-2507-base
    python summarize_results.py --experiments_dir experiments --output results_summary.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_json(filepath: str) -> Optional[List[Dict]]:
    """Load JSON file, return None if not found or invalid."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def summarize_dataset(dataset_dir: str) -> Optional[Dict]:
    """Summarize results for a single dataset directory."""
    raw_file = os.path.join(dataset_dir, 'raw_response.json')
    extracted_file = os.path.join(dataset_dir, 'extracted_response.json')
    processed_file = os.path.join(dataset_dir, 'processed_response.json')

    raw_data = load_json(raw_file)
    extracted_data = load_json(extracted_file)
    processed_data = load_json(processed_file)

    if not raw_data:
        return None

    n_samples = len(raw_data)

    # Token statistics from raw_response.json
    tokens = [item.get('num_tokens', 0) for item in raw_data]
    avg_tokens = sum(tokens) / n_samples if n_samples > 0 else 0
    min_tokens = min(tokens) if tokens else 0
    max_tokens = max(tokens) if tokens else 0

    # Performance metrics from processed_response.json
    success_rate = 0.0
    avg_lcs = 0.0
    avg_progress = 0.0
    extraction_rate = 0.0
    legal_rate = 0.0
    n_success = 0
    n_extracted = 0
    n_legal = 0

    if processed_data:
        n_success = sum(1 for item in processed_data if item.get('success', 0) == 1.0)
        n_extracted = sum(1 for item in processed_data if item.get('extracted', 0) == 1.0)
        n_legal = sum(1 for item in processed_data if item.get('legal', 0) == 1.0)
        success_rate = n_success / n_samples if n_samples > 0 else 0
        extraction_rate = n_extracted / n_samples if n_samples > 0 else 0
        legal_rate = n_legal / n_samples if n_samples > 0 else 0
        avg_lcs = sum(item.get('lcs_score', 0) for item in processed_data) / n_samples if n_samples > 0 else 0
        avg_progress = sum(item.get('progress_score', 0) for item in processed_data) / n_samples if n_samples > 0 else 0

    return {
        'n_samples': n_samples,
        'n_extracted': n_extracted,
        'n_legal': n_legal,
        'n_success': n_success,
        'extraction_rate': extraction_rate,
        'legal_rate': legal_rate,
        'success_rate': success_rate,
        'avg_lcs': avg_lcs,
        'avg_progress': avg_progress,
        'avg_tokens': avg_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
    }


def summarize_model(model_dir: str) -> Dict[str, Dict]:
    """Summarize results for all datasets under a model directory."""
    results = {}

    if not os.path.isdir(model_dir):
        return results

    for dataset_name in sorted(os.listdir(model_dir)):
        dataset_dir = os.path.join(model_dir, dataset_name)
        if os.path.isdir(dataset_dir):
            summary = summarize_dataset(dataset_dir)
            if summary:
                results[dataset_name] = summary

    return results


def print_summary(model_name: str, results: Dict[str, Dict]):
    """Print formatted summary for a model."""
    if not results:
        print(f"\n{model_name}: No results found")
        return

    print(f"\n{'='*120}")
    print(f"Model: {model_name}")
    print(f"{'='*120}")

    # Header
    print(f"\n{'Dataset':<35} {'Samples':>8} {'Extract':>9} {'Legal':>8} {'Success':>12} {'LCS':>8} {'Progress':>10} {'Tokens':>10}")
    print(f"{'-'*35} {'-'*8} {'-'*9} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")

    # Per-dataset results
    total_samples = 0
    total_extracted = 0
    total_legal = 0
    total_success = 0
    total_lcs = 0
    total_progress = 0
    total_tokens = 0

    for dataset_name, stats in results.items():
        n = stats['n_samples']
        extract_str = f"{stats['extraction_rate']*100:.1f}%"
        legal_str = f"{stats['legal_rate']*100:.1f}%"
        success_str = f"{stats['n_success']}/{n} ({stats['success_rate']*100:.1f}%)"

        print(f"{dataset_name:<35} {n:>8} {extract_str:>9} {legal_str:>8} {success_str:>12} {stats['avg_lcs']:>8.3f} {stats['avg_progress']:>10.3f} {stats['avg_tokens']:>10.0f}")

        total_samples += n
        total_extracted += stats['n_extracted']
        total_legal += stats['n_legal']
        total_success += stats['n_success']
        total_lcs += stats['avg_lcs'] * n
        total_progress += stats['avg_progress'] * n
        total_tokens += stats['avg_tokens'] * n

    # Overall summary
    if total_samples > 0:
        print(f"{'-'*35} {'-'*8} {'-'*9} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
        extract_str = f"{total_extracted/total_samples*100:.1f}%"
        legal_str = f"{total_legal/total_samples*100:.1f}%"
        overall_success = f"{total_success}/{total_samples} ({total_success/total_samples*100:.1f}%)"
        print(f"{'OVERALL':<35} {total_samples:>8} {extract_str:>9} {legal_str:>8} {overall_success:>12} {total_lcs/total_samples:>8.3f} {total_progress/total_samples:>10.3f} {total_tokens/total_samples:>10.0f}")


def save_csv(all_results: Dict[str, Dict[str, Dict]], output_path: str):
    """Save results to CSV file."""
    with open(output_path, 'w') as f:
        # Header
        f.write("model,dataset,n_samples,n_extracted,n_legal,n_success,extraction_rate,legal_rate,success_rate,avg_lcs,avg_progress,avg_tokens,min_tokens,max_tokens\n")

        for model_name, datasets in all_results.items():
            for dataset_name, stats in datasets.items():
                f.write(f"{model_name},{dataset_name},{stats['n_samples']},{stats['n_extracted']},{stats['n_legal']},{stats['n_success']},{stats['extraction_rate']:.4f},{stats['legal_rate']:.4f},{stats['success_rate']:.4f},{stats['avg_lcs']:.4f},{stats['avg_progress']:.4f},{stats['avg_tokens']:.1f},{stats['min_tokens']},{stats['max_tokens']}\n")

    print(f"CSV saved to: {output_path}")


def save_markdown(all_results: Dict[str, Dict[str, Dict]], output_path: str):
    """Save results to markdown file."""
    with open(output_path, 'w') as f:
        f.write("# Evaluation Results\n\n")

        for model_name, datasets in all_results.items():
            f.write(f"## {model_name}\n\n")

            # Table header
            f.write("| Dataset | Samples | Extracted | Legal | Success | LCS | Progress | Tokens |\n")
            f.write("|---------|--------:|----------:|------:|--------:|----:|---------:|-------:|\n")

            # Per-dataset rows
            total_samples = 0
            total_extracted = 0
            total_legal = 0
            total_success = 0
            total_lcs = 0
            total_progress = 0
            total_tokens = 0

            for dataset_name, stats in datasets.items():
                n = stats['n_samples']
                extract_pct = stats['extraction_rate'] * 100
                legal_pct = stats['legal_rate'] * 100
                success_pct = stats['success_rate'] * 100

                f.write(f"| {dataset_name} | {n} | {extract_pct:.1f}% | {legal_pct:.1f}% | {stats['n_success']}/{n} ({success_pct:.1f}%) | {stats['avg_lcs']:.3f} | {stats['avg_progress']:.3f} | {stats['avg_tokens']:.0f} |\n")

                total_samples += n
                total_extracted += stats['n_extracted']
                total_legal += stats['n_legal']
                total_success += stats['n_success']
                total_lcs += stats['avg_lcs'] * n
                total_progress += stats['avg_progress'] * n
                total_tokens += stats['avg_tokens'] * n

            # Overall row
            if total_samples > 0:
                extract_pct = total_extracted / total_samples * 100
                legal_pct = total_legal / total_samples * 100
                success_pct = total_success / total_samples * 100
                avg_lcs = total_lcs / total_samples
                avg_progress = total_progress / total_samples
                avg_tokens = total_tokens / total_samples

                f.write(f"| **OVERALL** | **{total_samples}** | **{extract_pct:.1f}%** | **{legal_pct:.1f}%** | **{total_success}/{total_samples} ({success_pct:.1f}%)** | **{avg_lcs:.3f}** | **{avg_progress:.3f}** | **{avg_tokens:.0f}** |\n")

            f.write("\n")

    print(f"Markdown saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Path to experiments directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to summarize (default: all models)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base name for CSV/MD files (without extension)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Output markdown file path"
    )

    args = parser.parse_args()

    experiments_dir = args.experiments_dir
    if not os.path.isdir(experiments_dir):
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return

    all_results = {}

    if args.model:
        # Summarize specific model
        model_dir = os.path.join(experiments_dir, args.model)
        results = summarize_model(model_dir)
        if results:
            all_results[args.model] = results
            print_summary(args.model, results)
        else:
            print(f"No results found for model: {args.model}")
    else:
        # Summarize all models
        for model_name in sorted(os.listdir(experiments_dir)):
            model_dir = os.path.join(experiments_dir, model_name)
            if os.path.isdir(model_dir):
                results = summarize_model(model_dir)
                if results:
                    all_results[model_name] = results
                    print_summary(model_name, results)

    # Determine output paths
    csv_path = args.csv
    md_path = args.markdown

    if args.output and all_results:
        # Use base name to generate both outputs
        csv_path = csv_path or f"{args.output}.csv"
        md_path = md_path or f"{args.output}.md"

    # Save outputs
    if csv_path and all_results:
        save_csv(all_results, csv_path)
    if md_path and all_results:
        save_markdown(all_results, md_path)

    print(f"\n{'='*100}")
    print("Summary complete!")


if __name__ == "__main__":
    main()
