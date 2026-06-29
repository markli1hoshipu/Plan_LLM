"""
Evaluation script for PDDL planning models.

This script evaluates trained models on PDDL planning tasks using vLLM for efficient inference.
"""

import argparse
import os
import json
import jsonlines
import re
import sys
from pathlib import Path

# Allow running without `pip install -e .` by exposing src/ on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from transformers.utils import logging as transformers_logging

# Import evaluation utilities from our library
from plan_llm.evaluation import (
    compute_lcs_score,
    compute_action_score,
    compute_vague_action_score,
    extract_answer,
    parse_actions,
    get_eval_prompts
)


def is_valid_action(action):
    """Check if action string is a valid action format: action_name(arg1, arg2, ...)"""
    if not isinstance(action, str):
        return False
    pattern = r'^[a-z_]+\([^)]*\)$'
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

# Suppress transformer warnings
transformers_logging.set_verbosity_error()


def eval_model(args):
    """
    Evaluate a model on PDDL planning tasks.

    Args:
        args: Parsed command-line arguments containing:
            - data_file: Path to evaluation data (JSONL format)
            - model: Model name or path
            - max_completion_length: Maximum tokens to generate
            - temperature: Sampling temperature
            - sampling_n: Number of completions per prompt
            - vllm_tensor_parallel_size: Tensor parallelism size
            - experiment_folder: Output directory for results
    """
    # Load evaluation data
    with open(args.data_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} evaluation examples from {args.data_file}")

    # Load prompts from configs
    system_prompt, user_prompt_template = get_eval_prompts()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True
    )

    # Initialize vLLM engine
    print(f"Initializing vLLM with model: {args.model}")
    engine = LLM(
        model=args.model,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_completion_length,
        quantization=args.quantization if hasattr(args, 'quantization') and args.quantization else None,
    )

    # Prepare prompts with chat formatting
    prompts = []
    for d in data:
        user_content = user_prompt_template(
            d['static_state'],
            d['dynamic_states'][0],
            d['dynamic_states'][-1]
        )
        # Remove quotes from user content to avoid parsing issues
        user_content = user_content.replace("'", "").replace('"', "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompts.append(messages)

    # Apply chat template
    input_texts = [
        tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]

    # Extract ground truth actions (remove last None if present)
    ground_truths = [d['actions'][:-1] if d['actions'][-1] is None else d['actions'] for d in data]

    # Generation parameters
    sampling_params = SamplingParams(
        n=args.sampling_n,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
        stop=tokenizer.eos_token
    )

    # Generate outputs
    print(f"Generating predictions for {len(input_texts)} examples...")
    outputs = engine.generate(input_texts, sampling_params)

    # Process results - separate into raw_response, extracted_response, processed_response
    raw_results = []
    extracted_results = []
    processed_results = []

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        initial_state = data[i]['static_state'] + data[i]['dynamic_states'][0]
        goal_state = data[i]['static_state'] + data[i]['dynamic_states'][-1]

        # Raw response data
        raw_results.append({
            "num_tokens": len(output.outputs[0].token_ids),
            "initial_state": initial_state,
            "goal_state": goal_state,
            "gt_actions": ground_truths[i],
            "input_prompt": prompts[i][1]['content'],
            "raw_response": generated_text,
        })

        # Extract and parse answer
        extracted_answer_str = extract_answer(generated_text)
        extracted_answer = parse_actions(extracted_answer_str)

        # Extracted response data (just extracted_answer)
        extracted_results.append({
            "extracted_answer": extracted_answer,
        })

        # Check extraction and legality
        extracted = 1.0 if is_extracted(extracted_answer) else 0.0
        legal = 1.0 if is_legal(extracted_answer) else 0.0

        # Compute metrics
        if extracted_answer:
            try:
                lcs_score = compute_lcs_score(ground_truths[i], extracted_answer)
                progress_score = compute_action_score(ground_truths[i], extracted_answer)
                # Success if extracted matches ground truth exactly
                success = 1.0 if extracted_answer == ground_truths[i] else 0.0
            except Exception as e:
                print(f"Warning: Error computing metrics for example {i}: {e}")
                lcs_score = 0.0
                progress_score = 0.0
                success = 0.0
        else:
            lcs_score = 0.0
            progress_score = 0.0
            success = 0.0

        # Processed response data (metrics)
        processed_results.append({
            "extracted": extracted,
            "legal": legal,
            "lcs_score": lcs_score,
            "progress_score": progress_score,
            "success": success,
        })

    # Save results
    os.makedirs(args.experiment_folder, exist_ok=True)

    # Save raw_response.json
    raw_file = os.path.join(args.experiment_folder, "raw_response.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_results, f, indent=2)

    # Save extracted_response.json
    extracted_file = os.path.join(args.experiment_folder, "extracted_response.json")
    with open(extracted_file, 'w') as f:
        json.dump(extracted_results, f, indent=2)

    # Save processed_response.json
    processed_file = os.path.join(args.experiment_folder, "processed_response.json")
    with open(processed_file, 'w') as f:
        json.dump(processed_results, f, indent=2)

    # Calculate and display statistics
    n = len(processed_results)
    avg_extracted = sum(r['extracted'] for r in processed_results) / n if n else 0
    avg_legal = sum(r['legal'] for r in processed_results) / n if n else 0
    avg_lcs = sum(r['lcs_score'] for r in processed_results) / n if n else 0
    avg_progress = sum(r['progress_score'] for r in processed_results) / n if n else 0
    avg_success = sum(r['success'] for r in processed_results) / n if n else 0
    avg_tokens = sum(r['num_tokens'] for r in raw_results) / len(raw_results) if raw_results else 0

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data_file}")
    print(f"Examples: {n}")
    print(f"Extracted Rate: {avg_extracted:.4f} ({int(avg_extracted * n)}/{n})")
    print(f"Legal Rate: {avg_legal:.4f} ({int(avg_legal * n)}/{n})")
    print(f"Average LCS Score: {avg_lcs:.4f}")
    print(f"Average Progress Score: {avg_progress:.4f}")
    print(f"Success Rate: {avg_success:.4f} ({int(avg_success * n)}/{n})")
    print(f"Average Generated Tokens: {avg_tokens:.2f}")
    print(f"Results saved to: {args.experiment_folder}")
    print(f"  - raw_response.json")
    print(f"  - extracted_response.json")
    print(f"  - processed_response.json")
    print(f"{'='*60}\n")


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate PDDL planning models using vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to evaluation data file (JSONL format)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen3-4B-Instruct-2507')"
    )

    # Generation arguments
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=8192,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--sampling_n",
        type=int,
        default=1,
        help="Number of completions to sample per prompt"
    )

    # GPU arguments
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=8,
        help="Tensor parallel size for vLLM (must be power of 2)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0 to 1.0)"
    )

    # Optional arguments
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", None],
        help="Quantization method for vLLM"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not specified)"
    )

    args = parser.parse_args()

    # Set GPU environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Auto-generate experiment folder if not provided
    if args.experiment_folder is None:
        model_name = args.model.replace("/", "_")
        data_name = Path(args.data_file).stem
        args.experiment_folder = f"experiments/{model_name}/{data_name}"

    os.makedirs(args.experiment_folder, exist_ok=True)

    return args


def run_batch_evaluation(model_names, file_names, base_args):
    """
    Run evaluation across multiple models and datasets.

    Args:
        model_names: List of model names to evaluate
        file_names: List of dataset filenames (without directory path)
        base_args: Base arguments object to copy settings from
    """
    data_dir = "/work/markhsp/hsp_pddl/data/processed/eval"

    for model in model_names:
        base_args.model = model

        for fname in file_names:
            # Update args for this run
            base_args.data_file = os.path.join(data_dir, f"{fname}.jsonl")
            base_args.experiment_folder = f"experiments/{model.replace('/', '_')}/{fname}"

            os.makedirs(base_args.experiment_folder, exist_ok=True)

            print(f"\n{'='*80}")
            print(f"Running evaluation: {model} on {fname}")
            print(f"{'='*80}\n")

            eval_model(base_args)


if __name__ == "__main__":
    args = get_args()

    # Single model/dataset evaluation
    if args.data_file and args.model:
        eval_model(args)

    # Uncomment below for batch evaluation across multiple models/datasets
    # model_names = [
    #     "Qwen/Qwen3-4B-Instruct-2507",
    #     "meta-llama/Meta-Llama-3-8B-Instruct",
    # ]
    #
    # file_names = [
    #     "align_data_eval",
    #     "stack_data_eval",
    #     "unstack_data_eval",
    #     "reorder_data_eval",
    # ]
    #
    # run_batch_evaluation(model_names, file_names, args)
