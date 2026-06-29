#!/usr/bin/env python3
"""
Step 1 (Pass@k): Generate k raw responses per problem using OpenAI API.

Uses curl subprocess to avoid httpx/HTTP2 issues on some clusters.
Output format is identical to step1_generate_passk.py: raw_response.json with fields:
    - num_tokens (list of k token counts)
    - initial_state
    - goal_state
    - gt_actions
    - input_prompt
    - raw_responses (list of k responses)
"""

import argparse
import os
import json
import sys
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow running without `pip install -e .` by exposing src/ on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from plan_llm.evaluation import get_eval_prompts


def call_openai_curl(api_key, model, system_prompt, user_content, temperature, max_tokens):
    """Make a single OpenAI API call via curl with retry."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    })

    for attempt in range(5):
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "--max-time", "120",
                    "https://api.openai.com/v1/chat/completions",
                    "-H", f"Authorization: Bearer {api_key}",
                    "-H", "Content-Type: application/json",
                    "-d", payload,
                ],
                capture_output=True, text=True, timeout=130,
            )
            resp = json.loads(result.stdout)
            if "error" in resp:
                raise Exception(resp["error"].get("message", str(resp["error"])))
            text = resp["choices"][0]["message"]["content"] or ""
            tokens = resp["usage"]["completion_tokens"]
            return text, tokens
        except Exception as e:
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    return "", 0


def generate_responses(args):
    """Generate k raw responses per problem using OpenAI API."""
    # Load evaluation data
    with open(args.data_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} evaluation examples from {args.data_file}", flush=True)
    print(f"Generating {args.num_samples} samples per problem for pass@{args.num_samples}", flush=True)

    # Load prompts from configs
    system_prompt, user_prompt_template = get_eval_prompts()

    # Extract ground truth actions
    ground_truths = [d['actions'][:-1] if d['actions'][-1] is None else d['actions'] for d in data]

    # Check for existing partial results to allow resume
    raw_file = os.path.join(args.output_dir, "raw_response.json")
    raw_results = []
    start_idx = 0
    if os.path.exists(raw_file):
        with open(raw_file, 'r') as f:
            raw_results = json.load(f)
        start_idx = len(raw_results)
        if start_idx > 0:
            print(f"Resuming from problem {start_idx}/{len(data)}", flush=True)

    total_calls = len(data) * args.num_samples
    completed = start_idx * args.num_samples

    for i in range(start_idx, len(data)):
        d = data[i]
        user_content = user_prompt_template(
            d['static_state'],
            d['dynamic_states'][0],
            d['dynamic_states'][-1]
        )
        user_content = user_content.replace("'", "").replace('"', "")

        initial_state = d['static_state'] + d['dynamic_states'][0]
        goal_state = d['static_state'] + d['dynamic_states'][-1]

        # Generate k samples for this problem
        raw_responses = []
        num_tokens = []

        # Use threads for concurrent API calls within each problem
        with ThreadPoolExecutor(max_workers=min(args.num_samples, 4)) as executor:
            futures = [
                executor.submit(
                    call_openai_curl, args.api_key, args.model, system_prompt,
                    user_content, args.temperature, args.max_tokens
                )
                for _ in range(args.num_samples)
            ]
            for future in as_completed(futures):
                text, tok = future.result()
                raw_responses.append(text)
                num_tokens.append(tok)
                completed += 1

        raw_results.append({
            "num_tokens": num_tokens,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "gt_actions": ground_truths[i],
            "input_prompt": user_content,
            "raw_responses": raw_responses,
        })

        # Save periodically (every 10 problems) for resume capability
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(raw_file, 'w') as f:
                json.dump(raw_results, f, indent=2)
            print(f"  Progress: {i+1}/{len(data)} problems ({completed}/{total_calls} calls) [saved]", flush=True)

    # Stats
    all_tokens = [t for r in raw_results for t in r['num_tokens']]
    avg_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0

    print(f"\n{'='*60}", flush=True)
    print(f"Step 1 (Pass@{args.num_samples}): Raw Response Generation Complete", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Data: {args.data_file}", flush=True)
    print(f"Problems: {len(raw_results)}", flush=True)
    print(f"Samples per problem: {args.num_samples}", flush=True)
    print(f"Total responses: {len(raw_results) * args.num_samples}", flush=True)
    print(f"Average Generated Tokens: {avg_tokens:.2f}", flush=True)
    print(f"Output: {raw_file}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Generate responses via OpenAI API (curl-based)")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    generate_responses(args)
