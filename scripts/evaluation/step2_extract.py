#!/usr/bin/env python3
"""
Step 2: Extract action sequences from raw responses using LLM.

This script uses vLLM to extract structured action sequences from the raw
model responses. It reads raw_response.json and outputs extracted_response.json.

Input: raw_response.json
Output: extracted_response.json with fields:
    - extracted_answer (list of action strings)
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

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from transformers.utils import logging as transformers_logging

# Suppress transformer warnings
transformers_logging.set_verbosity_error()


EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting structured information from text.

Your task is to extract the final action sequence from a planning response and convert it to a specific format.

Rules:
1. Look for the <FINAL>...</FINAL> tag which contains the action sequence
2. The content inside <FINAL> is typically a list of lists like: [[action_name, arg1, arg2], [action_name, arg1], ...]
3. Convert each action to string format: "action_name(arg1, arg2, ...)"
4. Return ONLY a JSON array of action strings, nothing else
5. If no valid action sequence is found, return: []

Example input:
"Here's my plan to solve this problem...
<FINAL>[[pick-up, 9, 1, 0], [rotate, 9, 3, 1, 0], [put-down, 9, 1, 0]]</FINAL>"

Example output:
["pick-up(9, 1, 0)", "rotate(9, 3, 1, 0)", "put-down(9, 1, 0)"]"""


EXTRACTION_USER_PROMPT = """Extract the final action sequence from this response and convert to the required format:

{response}

Return ONLY a valid JSON array of action strings. Convert [[action, arg1, arg2], ...] to ["action(arg1, arg2)", ...].
If no valid actions found, return: []"""


def truncate_response(response, max_chars=12000):
    """Truncate response to fit within model context, keeping the end."""
    if len(response) <= max_chars:
        return response
    start_chars = 1000
    end_chars = max_chars - start_chars - 100
    return response[:start_chars] + "\n\n...[truncated]...\n\n" + response[-end_chars:]


def extract_answers(args):
    """
    Extract action sequences from raw responses using LLM.
    """
    # Load raw responses
    raw_file = os.path.join(args.input_dir, "raw_response.json")
    with open(raw_file, 'r') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw responses from {raw_file}")

    # Check if this is a Mistral model that needs special tokenizer handling
    is_mistral = 'mistral' in args.model.lower() or 'ministral' in args.model.lower()

    if is_mistral:
        # For Mistral models, use vLLM's built-in tokenizer mode
        print(f"Detected Mistral model, using tokenizer_mode='mistral'")
        engine = LLM(
            model=args.model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
        )
        tokenizer = engine.get_tokenizer()
        eos_token = "</s>"
    else:
        # Initialize tokenizer for non-Mistral models
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
            max_model_len=args.max_model_len,
        )
        eos_token = tokenizer.eos_token

    # Sampling params for extraction (low temperature for consistency)
    sampling_params = SamplingParams(
        n=1,
        max_tokens=2048,
        temperature=0.1,
        top_p=0.95,
        stop=eos_token
    )

    # Build extraction prompts
    prompts = []
    for item in raw_data:
        raw_response = truncate_response(item.get('raw_response', ''))
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACTION_USER_PROMPT.format(response=raw_response)}
        ]
        prompts.append(messages)

    # Apply chat template
    if is_mistral:
        # For Mistral tokenizer from vLLM, apply_chat_template returns token IDs
        from vllm import TokensPrompt
        input_token_ids = []
        for prompt in prompts:
            token_ids = tokenizer.apply_chat_template(
                messages=prompt,
                add_generation_prompt=True,
            )
            input_token_ids.append(token_ids)
        # Generate extractions using token IDs
        print(f"Extracting answers from {len(input_token_ids)} responses...")
        token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_token_ids]
        outputs = engine.generate(token_prompts, sampling_params)
    else:
        input_texts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            ) for prompt in prompts
        ]
        # Generate extractions
        print(f"Extracting answers from {len(input_texts)} responses...")
        outputs = engine.generate(input_texts, sampling_params)

    # Parse extracted answers and merge with raw data
    extracted_results = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()

        # Try to parse as JSON list
        extracted_answer = []
        try:
            # Clean up markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            # Find JSON array in response
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                extracted_answer = json.loads(match.group())
                if not isinstance(extracted_answer, list):
                    extracted_answer = []
                # Ensure all items are strings
                extracted_answer = [str(a) for a in extracted_answer if a]
        except (json.JSONDecodeError, Exception):
            extracted_answer = []

        # Include all fields from raw_response.json plus extracted_answer
        result = dict(raw_data[i])  # Copy all fields from raw data
        result["extracted_answer"] = extracted_answer
        extracted_results.append(result)

    # Save results
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    extracted_file = os.path.join(output_dir, "extracted_response.json")
    with open(extracted_file, 'w') as f:
        json.dump(extracted_results, f, indent=2)

    # Display statistics
    n_extracted = sum(1 for r in extracted_results if r['extracted_answer'])
    extraction_rate = n_extracted / len(extracted_results) if extracted_results else 0

    print(f"\n{'='*60}")
    print(f"Step 2: LLM Extraction Complete")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(extracted_results)}")
    print(f"Extracted: {n_extracted}/{len(extracted_results)} ({extraction_rate*100:.1f}%)")
    print(f"Output: {extracted_file}")
    print(f"{'='*60}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Extract answers from raw responses using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw_response.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input_dir)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model to use for extraction")

    # GPU arguments
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=8,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="Maximum model context length")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    return args


if __name__ == "__main__":
    args = get_args()
    extract_answers(args)
