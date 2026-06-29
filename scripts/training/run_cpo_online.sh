#!/usr/bin/env bash
# Online Constrained Policy Optimization (CPO) for Self-CriTeach.
# Hyperparameters and training subset match the camera-ready paper:
#   "For the reinforcement learning stage, we further optimize the SFT model
#    using Constrained Policy Optimization (CPO) with a learning rate of 1e-6,
#    a KL penalty coefficient β = 0.1, constraint threshold d = 0.25, and
#    constraint weight λ = 0.5. Training is conducted for 3 epochs on the
#    reorder-2000 subset (2,000 problems) with an effective batch size of 4
#    (via gradient accumulation) across 8 GPUs."
#
# Usage (run from the repository root):
#   bash scripts/training/run_cpo_online.sh
#
# Override anything by exporting before running, e.g.:
#   MODEL_PATH=Self-CriTeach/SCT MODEL_SUBFOLDER=Qwen3-4B bash scripts/training/run_cpo_online.sh
#   MODEL_PATH=./checkpoints/sft/final bash scripts/training/run_cpo_online.sh

set -e

# ---------- paths (override-able) ----------
# Default starts from the released SFT checkpoint on HF.
MODEL_PATH="${MODEL_PATH:-Self-CriTeach/SCT}"
MODEL_SUBFOLDER="${MODEL_SUBFOLDER:-Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-./data/train/reorder_data.jsonl}"   # reorder-2000 subset per the paper
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/cpo}"

# ---------- training hyperparameters (paper-aligned) ----------
NUM_EPOCHS=3
BATCH_SIZE=1            # per-GPU
GRAD_ACCUM=4            # → effective batch of 4 with 8 GPUs accumulating
LEARNING_RATE=1e-6
BETA=0.1
CONSTRAINT_THRESHOLD=0.25
LAMBDA=0.5
MAX_NEW_TOKENS=16384
TEMPERATURE=0.7

# ---------- runtime ----------
NUM_GPUS="${NUM_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT_DIR"

SUBFOLDER_ARG=""
if [ -n "$MODEL_SUBFOLDER" ]; then
  SUBFOLDER_ARG="--model_subfolder ${MODEL_SUBFOLDER}"
fi

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    scripts/training/train_cpo_online.py \
    --model_path "$MODEL_PATH" $SUBFOLDER_ARG \
    --train_data_path "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --beta $BETA \
    --constraint_threshold $CONSTRAINT_THRESHOLD \
    --lambda_constraint $LAMBDA \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE
