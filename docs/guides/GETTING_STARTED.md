# Getting Started with Plan_LLM

This guide will help you set up and run your first PDDL planning experiment with Self-CriTeach.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 50GB+ disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/markli1hoshipu/Plan_LLM.git
cd Plan_LLM
```

### 2. Create Virtual Environment

Using conda (recommended):
```bash
conda create -n plan_llm python=3.11
conda activate plan_llm
```

Or using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import plan_llm; print('Installation successful!')"
```

## Quick Test

### Download Sample Data

```bash
# Create data directory
mkdir -p data/sample

# Download a small sample (you'll need to set up HuggingFace CLI first)
huggingface-cli download self-criteach/pddl-planning-data \
    --repo-type dataset \
    --include "test/align_data_eval.jsonl" \
    --local-dir data/sample
```

### Run Evaluation on Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model_name = "self-criteach/SCT-4B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Format a simple problem
problem = """
Your task is to predict a set of actions that arrive at the goal state starting from the initial state.

Static predicates: [[box, 1], [box, 2], [table, 0], [robot, r]]

Initial state: [[on-table, 1, 0], [on-table, 2, 0], [hand_free, r], [top, 1], [top, 2]]

Goal state: [[on-table, 1, 0], [above, 2, 1], [hand_free, r], [top, 2]]

IMPORTANT: Always output the final plan inside <FINAL> ... </FINAL> tags.
"""

# Generate
messages = [
    {"role": "system", "content": "You are a robot assistant. Your task is to generate a plan given the initial and goal state."},
    {"role": "user", "content": problem}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print("Generated Plan:")
print(response)
```

## Next Steps

- **Training**: See [TRAINING.md](TRAINING.md) for training your own models
- **Evaluation**: See [EVALUATION.md](EVALUATION.md) for comprehensive evaluation
- **Data Preparation**: See [DATA_PREPARATION.md](DATA_PREPARATION.md) for custom datasets

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
```python
# Use smaller batch size
--per_device_batch_size 1 --gradient_accumulation_steps 8

# Or use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)
```

### Import Errors

Make sure you've installed the package:
```bash
pip install -e .
```

Or add the src directory to your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Support

- **Issues**: [GitHub Issues](https://github.com/markli1hoshipu/Plan_LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/markli1hoshipu/Plan_LLM/discussions)
