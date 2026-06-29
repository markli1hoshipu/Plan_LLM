# COT Generation Quality Evaluation Documentation

This document records the findings from COT quality evaluation and provides a checklist for validating future COT generations.

## Summary of Findings (January 2026)

### Bug Fixes Applied
1. **Actions Truncation Bug** (`scripts/generate_cot.py:80`)
   - **Issue**: `actions=d['actions'][:-1]` always removed the last action from ground truth
   - **Fix**: `actions = d['actions'][:-1] if d['actions'][-1] is None else d['actions']`

2. **Typo in Prompt Template** (`configs/prompts/cot_user_prompt_template.md:140`)
   - **Issue**: "actwhions" instead of "actions"
   - **Fix**: Corrected spelling

### COT Quality Comparison

| COT Source Model | Problematic Samples | Main Issues |
|------------------|---------------------|-------------|
| Qwen3-4B-Instruct | 90% (9/10) | Confusion ("Wait —"), contradictions, inconsistency notes |
| Qwen3-30B-A3B-Thinking | 50% (5/10) | Mainly "invalid state" mentions |

### Training Results (Pass@1)

| Model | Average Accuracy | Notes |
|-------|------------------|-------|
| Base Qwen3-4B | 28.6% | No COT training |
| Trained on 30B COT | 14.8% | Degraded vs base |
| Trained on 4B COT (old) | 10.6% | Worst performance |

**Key Finding**: COT training currently degrades model performance compared to baseline.

---

## Quality Checklist for New COT Generation

### 1. Pre-Generation Checks

- [ ] Verify `generate_cot.py` has the correct actions handling:
  ```python
  actions = d['actions'][:-1] if d['actions'][-1] is None else d['actions']
  ```
- [ ] Verify prompt template has no typos (check "actions" spelling)
- [ ] Confirm all 6 datasets are included:
  - align_data
  - stack_data
  - unstack_data
  - reorder_data
  - suboptimal_data
  - suboptimal_data_extra

### 2. Post-Generation Quality Checks

Run these checks on the generated COT files:

#### Check 1: Problematic Language Patterns
```bash
# Count samples with confusion/uncertainty language
python3 << 'EOF'
import json
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/cot_file.jsonl"

problematic_patterns = [
    'contradiction',
    'violates',
    'inconsisten',
    'wait —',
    'wait -',
    'something is wrong',
    'cannot achieve',
    'impossible to',
    'error in the',
    'problem appears',
]

total = 0
issues = 0

with open(file_path) as f:
    for line in f:
        total += 1
        d = json.loads(line)
        cot = (d.get('cot', '') or '').lower()

        for pattern in problematic_patterns:
            if pattern in cot:
                issues += 1
                break

print(f"File: {file_path}")
print(f"Total samples: {total}")
print(f"Samples with issues: {issues} ({issues/total*100:.1f}%)")
print(f"Clean samples: {total - issues} ({(total-issues)/total*100:.1f}%)")
EOF
```

**Target**: < 10% samples with problematic patterns

#### Check 2: Final Answer Validity
```bash
# Verify final answers match ground truth actions
python3 << 'EOF'
import json
import re

file_path = "path/to/cot_file.jsonl"

with open(file_path) as f:
    correct = 0
    total = 0

    for line in f:
        total += 1
        d = json.loads(line)
        cot = d.get('cot', '') or ''
        actions = d.get('actions', [])

        # Remove trailing None
        if actions and actions[-1] is None:
            actions = actions[:-1]

        # Check if last action appears in COT
        if actions:
            last_action_str = str(actions[-1])
            if last_action_str in cot or str(actions[-1][0]) in cot:
                correct += 1

print(f"Samples with last action in COT: {correct}/{total} ({correct/total*100:.1f}%)")
EOF
```

**Target**: > 95% samples should have last action in COT

#### Check 3: Reasoning Length Distribution
```bash
# Check COT length distribution
python3 << 'EOF'
import json
import statistics

file_path = "path/to/cot_file.jsonl"
lengths = []

with open(file_path) as f:
    for line in f:
        d = json.loads(line)
        cot = d.get('cot', '') or ''
        lengths.append(len(cot))

print(f"COT Length Statistics:")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Mean: {statistics.mean(lengths):.0f}")
print(f"  Median: {statistics.median(lengths):.0f}")
print(f"  Samples with empty COT: {lengths.count(0)}")
EOF
```

**Target**: No empty COTs, reasonable length distribution (5000-20000 chars typical)

### 3. Training Data Validation

After generating training JSONL files:

#### Check 4: Message Format Validation
```bash
python3 << 'EOF'
import json

file_path = "path/to/train_file.jsonl"

with open(file_path) as f:
    for i, line in enumerate(f):
        d = json.loads(line)

        if 'messages' not in d:
            print(f"Line {i}: Missing 'messages' key")
            continue

        messages = d['messages']
        roles = [m['role'] for m in messages]

        # Should have system, user, assistant
        if 'system' not in roles:
            print(f"Line {i}: Missing system message")
        if 'user' not in roles:
            print(f"Line {i}: Missing user message")
        if 'assistant' not in roles:
            print(f"Line {i}: Missing assistant message")

print("Validation complete")
EOF
```

#### Check 5: Sample Inspection
Manually inspect 10 random samples:
```bash
python3 << 'EOF'
import json
import random

file_path = "path/to/train_file.jsonl"

with open(file_path) as f:
    lines = f.readlines()

random.seed(42)
for line in random.sample(lines, min(10, len(lines))):
    d = json.loads(line)
    for msg in d['messages']:
        if msg['role'] == 'assistant':
            content = msg['content']
            print("="*80)
            print(content[:2000])
            print()
            break
EOF
```

Look for:
- [ ] Clear step-by-step reasoning
- [ ] No expressions of confusion ("Wait —", "contradiction")
- [ ] Logical flow from initial state to goal
- [ ] Final answer in correct format

---

## Recommendations for Better COT Generation

### Option 1: Use Stronger Model
- Use Qwen3-30B-A3B-Thinking or larger model for COT generation
- 30B model shows 50% less problematic patterns than 4B

### Option 2: Filter Bad Samples
- Post-process generated COT to remove samples with problematic patterns
- Only keep samples with clean, confident reasoning

### Option 3: Improve Prompt
- Add explicit instruction to avoid uncertainty language
- Example addition to system prompt:
  ```
  IMPORTANT: Provide confident, clear reasoning. Do not express uncertainty
  or confusion. If a step is correct, state it definitively.
  ```

### Option 4: Two-Stage Validation
1. Generate COT
2. Use a validator model to score COT quality
3. Only keep high-quality samples

---

## File Locations

- COT Generation Script: `scripts/generate_cot.py`
- Prompt Templates: `configs/prompts/`
- Raw COT Output: `data/training/cot_generated/{MODEL_NAME}/`
- Training Data: `data/processed/train/`

---

## Version History

| Date | Changes |
|------|---------|
| 2026-01-13 | Initial documentation, bug fixes, quality analysis |
