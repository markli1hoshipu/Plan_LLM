# Plan_LLM: Self-CriTeach for PDDL Planning

Official implementation of "Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation"

**Paper**: [arXiv:2509.21543](https://arxiv.org/abs/2509.21543)

## Overview

Self-CriTeach is a self-teaching framework that improves LLM performance on PDDL planning tasks through:
- Chain-of-Thought (CoT) generation from ground truth plans
- Supervised fine-tuning (SFT) on self-generated reasoning
- Preference optimization (CPO/DPO) for plan refinement

## Key Results

| Model | Base | Pass@1 | Pass@4 | Improvement |
|-------|------|--------|--------|-------------|
| Qwen3-4B | 25.1% | 32.9% | 50.0% | +31% |
| Llama-3.1-8B | baseline | 27.7% | 42.7% | - |

## Installation

```bash
# Clone repository
git clone https://github.com/markli1hoshipu/Plan_LLM.git
cd Plan_LLM

# Create environment
conda create -n plan_llm python=3.11
conda activate plan_llm

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Download Data

```bash
# Download training and evaluation data
huggingface-cli download Self-CriTeach/pddl-planning-data --repo-type dataset --local-dir ./data
```

### Evaluate Pre-trained Model

Trained checkpoints are released at [`Self-CriTeach/SCT`](https://huggingface.co/Self-CriTeach/SCT) — both backbones share one repo with subfolders:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load by backbone (use subfolder=):
model = AutoModelForCausalLM.from_pretrained("Self-CriTeach/SCT", subfolder="Qwen3-4B")
tokenizer = AutoTokenizer.from_pretrained("Self-CriTeach/SCT", subfolder="Qwen3-4B")
# or: subfolder="Llama-3.1-8B"

# Format PDDL problem
problem = """
Static predicates: [[box, 1], [box, 2], [table, 0], [robot, r]]
Initial state: [[on-table, 1, 0], [on-table, 2, 0], [hand_free, r], [top, 1], [top, 2]]
Goal state: [[on-table, 1, 0], [above, 2, 1], [hand_free, r], [top, 2]]
"""

# Generate plan
inputs = tokenizer(problem, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048)
plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(plan)
```

## Training

### 1. Generate Chain-of-Thought Data

```bash
python scripts/data/generate_cot.py \
    --model Qwen/Qwen3-4B-Instruct \
    --data_dir data/raw \
    --output_dir data/generated/cot
```

### 2. Prepare Training Data

```bash
python scripts/data/generate_training_data.py \
    --cot_dir data/generated/cot \
    --output_dir data/processed
```

### 3. Supervised Fine-Tuning

```bash
python scripts/training/train_sft.py \
    --model_name Qwen/Qwen3-4B-Instruct \
    --train_file data/processed/train.jsonl \
    --eval_file data/processed/eval.jsonl \
    --output_dir checkpoints/sft \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 4
```

### 4. Reinforcement Learning with Structured Rewards (CPO)

The full Self-CriTeach pipeline post-trains the SFT model with online Constrained Policy Optimization, using the self-generated planning domain as a step-level reward signal. Paper hyperparameters are encoded in the runner script:

```bash
# Start from the released SFT checkpoint (or pass MODEL_PATH=./checkpoints/sft/final to use your own)
bash scripts/training/run_cpo_online.sh
```

This runs `train_cpo_online.py` with `lr=1e-6`, `β=0.1`, `d=0.25`, `λ=0.5`, 3 epochs on the reorder-2000 subset, 8 GPUs, gradient-accumulation effective batch of 4 — matching the paper. Override `MODEL_PATH`, `TRAIN_DATA`, `OUTPUT_DIR`, or `NUM_GPUS` via environment variables.

### 5. Preference Optimization (Alternative)

```bash
python scripts/training/train_dpo_qwen.py \
    --model_name checkpoints/sft/final \
    --train_file data/processed/dpo_pairs.jsonl \
    --output_dir checkpoints/dpo
```

## Evaluation

```bash
python scripts/evaluation/eval.py \
    --model checkpoints/sft/final \
    --data_file data/eval/test.jsonl \
    --experiment_folder results/sft_eval \
    --gpus 0,1,2,3 \
    --vllm_tensor_parallel_size 4
```

For Pass@k evaluation, run the three pipeline steps in order:
`step1_generate_passk.py` → `step2_extract_passk.py` → `step3_validate_passk.py`.

## Repository Structure

```
Plan_LLM/
├── src/plan_llm/              # Core library
│   ├── evaluation/            # Evaluation metrics and validation
│   ├── training/              # Training utilities
│   ├── data/                  # Data processing
│   └── utils/                 # Helper functions
├── scripts/                   # Executable scripts
│   ├── data/                  # Data preparation
│   ├── training/              # Training scripts
│   ├── evaluation/            # Evaluation pipeline
│   └── analysis/              # Result analysis
├── configs/                   # Configuration files
│   ├── prompts/               # Prompt templates
│   ├── deepspeed/             # DeepSpeed configs
│   ├── training/              # Training configs
│   └── evaluation/            # Evaluation configs
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

## Documentation

- [Getting Started](docs/guides/GETTING_STARTED.md) — Installation and setup
- [CoT Quality Analysis](docs/guides/COT_QUALITY_ANALYSIS.md) — Bug-fix history and quality checklist for chain-of-thought generation
- [Baseline Results](docs/guides/BASELINE_RESULTS.md) — Reference numbers for baseline models

## Resources

- **HuggingFace Org**: [Self-CriTeach](https://huggingface.co/Self-CriTeach)
- **Dataset**: [Self-CriTeach/pddl-planning-data](https://huggingface.co/datasets/Self-CriTeach/pddl-planning-data) — 7,476 train + 2,800 eval PDDL planning problems
- **Models**: [Self-CriTeach/SCT](https://huggingface.co/Self-CriTeach/SCT) — SCT-Qwen3-4B + SCT-Llama-3.1-8B (+ intermediate training-curve checkpoints), loaded via `subfolder=`
- **Paper**: [arXiv:2509.21543](https://arxiv.org/abs/2509.21543)

## Citation

```bibtex
@article{huang2025selfcriteach,
  title         = {Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation},
  author        = {Huang, Jinbang and Li, Zhiyuan and Hu, Yuanzhao and Zhang, Zhanguang and Coates, Mark and Quan, Xingyue and Zhang, Yingxue},
  journal       = {arXiv preprint arXiv:2509.21543},
  year          = {2025},
  url           = {https://arxiv.org/abs/2509.21543}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [Qwen3](https://github.com/QwenLM/Qwen) and [Llama 3.1](https://github.com/meta-llama/llama3) models
- Evaluation framework adapted from PDDL benchmarks
- Training infrastructure powered by [TRL](https://github.com/huggingface/trl) and [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## Contact

- **Issues**: [GitHub Issues](https://github.com/markli1hoshipu/Plan_LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/markli1hoshipu/Plan_LLM/discussions)
- **Email**: [Contact via GitHub](https://github.com/markli1hoshipu)

---

**Note**: This repository contains a partial implementation focusing on the SFT and preference optimization training pipeline. The automated domain generation component described in the paper will be integrated from separate work in a future release.
