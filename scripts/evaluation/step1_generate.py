#!/usr/bin/env python3
"""
Step 1: Generate raw responses from LLM.

This script uses vLLM to generate raw responses from a model and saves them
to raw_response.json. No extraction or validation is done here.

Output: raw_response.json with fields:
    - num_tokens
    - initial_state
    - goal_state
    - gt_actions
    - input_prompt
    - raw_response
"""

import argparse
import os
import json
import sys
from pathlib import Path

# Allow running without `pip install -e .` by exposing src/ on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from transformers.utils import logging as transformers_logging

from plan_llm.evaluation import get_eval_prompts

# Suppress transformer warnings
transformers_logging.set_verbosity_error()


def generate_responses(args):
    """
    Generate raw responses from a model using vLLM.

    Only saves raw_response.json - no extraction or processing.
    """
    # Load evaluation data
    with open(args.data_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} evaluation examples from {args.data_file}")

    # Load prompts from configs
    system_prompt, user_prompt_template = get_eval_prompts()

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
            quantization=args.quantization if args.quantization else None,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
        )
        tokenizer = engine.get_tokenizer()
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
            quantization=args.quantization if args.quantization else None,
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
    if is_mistral:
        # For Mistral tokenizer from vLLM, apply_chat_template returns token IDs
        # We need to pass the prompts directly to generate() which handles tokenization
        input_token_ids = []
        for prompt in prompts:
            token_ids = tokenizer.apply_chat_template(
                messages=prompt,
                add_generation_prompt=True,
            )
            input_token_ids.append(token_ids)
        input_texts = None  # Will use token IDs instead
        eos_token = "</s>"
    else:
        input_texts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            ) for prompt in prompts
        ]
        input_token_ids = None
        eos_token = tokenizer.eos_token

    # Extract ground truth actions (remove last None if present)
    ground_truths = [d['actions'][:-1] if d['actions'][-1] is None else d['actions'] for d in data]

    # Generation parameters
    sampling_params = SamplingParams(
        n=args.sampling_n,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=eos_token
    )

    # Generate outputs
    if is_mistral:
        # For Mistral, use token IDs with TokensPrompt
        from vllm import TokensPrompt
        print(f"Generating predictions for {len(input_token_ids)} examples...")
        token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_token_ids]
        outputs = engine.generate(token_prompts, sampling_params)
    else:
        print(f"Generating predictions for {len(input_texts)} examples...")
        outputs = engine.generate(input_texts, sampling_params)

    # Process results - only save raw response data
    raw_results = []

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        initial_state = data[i]['static_state'] + data[i]['dynamic_states'][0]
        goal_state = data[i]['static_state'] + data[i]['dynamic_states'][-1]

        raw_results.append({
            "num_tokens": len(output.outputs[0].token_ids),
            "initial_state": initial_state,
            "goal_state": goal_state,
            "gt_actions": ground_truths[i],
            "input_prompt": prompts[i][1]['content'],
            "raw_response": generated_text,
        })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    raw_file = os.path.join(args.output_dir, "raw_response.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_results, f, indent=2)

    # Display statistics
    avg_tokens = sum(r['num_tokens'] for r in raw_results) / len(raw_results)

    print(f"\n{'='*60}")
    print(f"Step 1: Raw Response Generation Complete")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data_file}")
    print(f"Examples: {len(raw_results)}")
    print(f"Average Generated Tokens: {avg_tokens:.2f}")
    print(f"Output: {raw_file}")
    print(f"{'='*60}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate raw responses using vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to evaluation data file (JSONL format)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for raw_response.json")

    # Generation arguments
    parser.add_argument("--max_completion_length", type=int, default=8192,
                        help="Maximum tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="Maximum model context length")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top-p")
    parser.add_argument("--sampling_n", type=int, default=1,
                        help="Number of completions per prompt")

    # GPU arguments
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=8,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", None],
                        help="Quantization method for vLLM")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    return args


if __name__ == "__main__":
    args = get_args()
    generate_responses(args)
