import argparse
import os
import json
import jsonlines
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import SamplingParams, LLM

# Load prompts from config files
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "configs" / "prompts"

def load_prompt(filename):
    """Load prompt from config file."""
    with open(PROMPTS_DIR / filename, 'r') as f:
        return f.read().strip()

# Load system and user prompts from config
system_prompt = load_prompt("cot_system_prompt.md")
user_prompt_template = load_prompt("cot_user_prompt_template.md")


def extract_answer(output):
    # This pattern ensures no nested <REASON> inside the capture
    matches = re.findall(r'<REASON>((?:(?!<REASON>).)*?)</REASON>', output, re.DOTALL)
    if matches:
        return matches[-1]  # last match only
    return None

def eval_model(args):
    with open(args.data_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Check if this is a Mistral model that needs special tokenizer handling
    is_mistral = 'mistral' in args.model.lower() or 'ministral' in args.model.lower()

    if is_mistral:
        print(f"Detected Mistral model, using tokenizer_mode='mistral'")
        engine = LLM(
            model=args.model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            quantization=args.quantization if hasattr(args, 'quantization') else None,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
        )
        tokenizer = engine.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
            trust_remote_code=True
        )
        engine = LLM(
            model=args.model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            quantization=args.quantization if hasattr(args, 'quantization') else None,
        )

    # Build prompts
    prompts = []
    for d in data:
        # Get actions, removing trailing None if present (final state has no action)
        actions = d['actions'][:-1] if d['actions'][-1] is None else d['actions']

        # Format user prompt with data
        user_prompt = user_prompt_template.format(
            static_state=d['static_state'],
            initial_state=d['dynamic_states'][0],
            goal_state=d['dynamic_states'][-1],
            actions=actions
        ).replace("\'", "").replace("\"", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompts.append(messages)

    # Apply chat template
    if is_mistral:
        input_token_ids = []
        for prompt in prompts:
            token_ids = tokenizer.apply_chat_template(
                messages=prompt,
                add_generation_prompt=True,
            )
            input_token_ids.append(token_ids)
        eos_token = "</s>"
    else:
        input_texts = [tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts]
        eos_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        n=args.sampling_n,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=eos_token
    )

    # Run generation
    if is_mistral:
        from vllm import TokensPrompt
        print(f"Generating for {len(input_token_ids)} examples...")
        token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_token_ids]
        outputs = engine.generate(token_prompts, sampling_params)
    else:
        print(f"Generating for {len(input_texts)} examples...")
        outputs = engine.generate(input_texts, sampling_params)

    results = []
    for d, out in zip(data, outputs):
        # each out can have multiple generations (n>1), take first
        generated_text = out.outputs[0].text if out.outputs else ""
        extracted_answer = extract_answer(generated_text)

        d["cot"] = extracted_answer  # add chain of thought/explanation
        d["generated_text"] = generated_text
        results.append(d)


    # Determine output file name
    data_dir = os.path.dirname(args.data_file)
    base_name = os.path.basename(args.data_file).replace(".jsonl", "")

    # If output_file is specified, use it; otherwise create default name
    if args.output_file:
        output_file = args.output_file
    else:
        # Remove '_cot' suffix if present, then add '_with_cot'
        if base_name.endswith('_cot'):
            base_name = base_name[:-4]  # Remove '_cot'
        output_file = os.path.join(data_dir, f"{base_name}_with_cot.jsonl")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with jsonlines.open(output_file, "w") as f:
        f.write_all(results)

    print(f"Saved {len(results)} results to {output_file}")




def get_args():

    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Thought reasoning for PDDL planning tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the input jsonl data file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output file (default: auto-generated based on input)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
    )

    parser.add_argument("--max_completion_length", type=int, default=8192, help="Max generation tokens")
    parser.add_argument("--max_model_len", type=int, default=16384, help="Max model context length for vLLM engine")

    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--sampling_n", type=int, default=1, help="Number of completions to sample")

    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1')")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")

    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--quantization", type=str, default=None,
                      choices=["awq", "gptq", None],
                      help="Quantization method for vLLM")


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    return args

if __name__ == "__main__":
    args = get_args()

    print("="*70)
    print("Chain-of-Thought Generation")
    print("="*70)
    print(f"Data file: {args.data_file}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_completion_length}")
    print(f"GPUs: {args.gpus}")
    print(f"Tensor parallel size: {args.vllm_tensor_parallel_size}")
    print("="*70)

    eval_model(args)
