"""
Online Constrained Policy Optimization (CPO) for PDDL Planning.

This implements the CPO algorithm from Achiam et al. (2017) with:
- Step-level rewards based on goal predicate satisfaction (Eq. 2)
- Constraint costs for precondition violations (Eq. 3)
- Online trajectory sampling from the current policy

No pre-collected preference data needed - this is true online RL.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm

# Allow running without `pip install -e .` by exposing src/ on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from plan_llm.evaluation.validator import (
    can_perform_action,
    apply_action,
    process_actions,
    compute_logical_divergence,
)


@dataclass
class CPOConfig:
    """Configuration for CPO training."""
    # Model — pass via --model_path; default is the released SFT checkpoint on HF
    model_path: str = "Self-CriTeach/SCT"  # use subfolder= via --model_subfolder, or local path
    model_subfolder: Optional[str] = None  # e.g., "Qwen3-4B" when loading from Self-CriTeach/SCT

    # Data — pass via --train_data_path; default expects local jsonl downloaded from HF
    train_data_path: str = "./data/train"

    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_samples_per_problem: int = 1
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0

    # CPO specific
    beta: float = 0.1  # KL penalty coefficient
    constraint_threshold: float = 0.25  # d in Eq. 1
    lambda_constraint: float = 0.5  # λ in Eq. 3

    # Generation
    max_new_tokens: int = 16384
    temperature: float = 0.7
    top_p: float = 0.95

    # Output
    output_dir: str = "./checkpoints/cpo"
    save_steps: int = 100
    logging_steps: int = 10


def load_training_problems(data_path: str) -> List[Dict]:
    """Load training problems from JSONL file(s)."""
    problems = []
    path = Path(data_path)

    # Handle both single file and directory
    if path.is_file():
        jsonl_files = [path]
    else:
        jsonl_files = list(path.glob("*.jsonl"))

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line)
                    # Convert to standard format
                    problem["initial_state"] = problem["dynamic_states"][0]
                    problem["goal_state"] = problem["dynamic_states"][-1]
                    problems.append(problem)

    print(f"Loaded {len(problems)} training problems from {data_path}")
    return problems


def load_prompt_templates(base_dir: str) -> Tuple[str, str]:
    """Load system and user prompt templates from eval prompts (same as inference)."""
    # Use eval prompts - same as used during inference
    system_path = Path(base_dir) / "configs/prompts/eval_system_prompt.md"
    user_path = Path(base_dir) / "configs/prompts/eval_user_prompt_template.md"

    if not system_path.exists():
        raise FileNotFoundError(f"System prompt not found: {system_path}")
    if not user_path.exists():
        raise FileNotFoundError(f"User prompt template not found: {user_path}")

    system_prompt = system_path.read_text().strip()
    user_template = user_path.read_text().strip()

    return system_prompt, user_template


def format_prompt(problem: Dict, tokenizer, system_prompt: str, user_template: str) -> str:
    """Format a problem into a chat prompt (same format as inference)."""
    static_state = problem["static_state"]
    initial_state = problem["initial_state"]
    goal_state = problem["goal_state"]

    # Format user prompt using eval template
    user_prompt = user_template.format(
        static_state=str(static_state),
        initial_state=str(initial_state),
        goal_state=str(goal_state)
    )

    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text


def parse_action_sequence(response: str) -> List[List[str]]:
    """
    Parse action sequence from model response.
    Looks for <FINAL>[[action, arg1, ...], ...]</FINAL> format.
    """
    import re

    # Try to find <FINAL> tag
    pattern = r'<FINAL>\s*(.*?)\s*</FINAL>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if not match:
        return []

    content = match.group(1).strip()
    actions = []

    # Parse [[action, arg1, arg2], ...] format
    inner_pattern = r'\[\s*([^\[\]]+?)\s*\]'
    inner_matches = re.findall(inner_pattern, content)

    for inner in inner_matches:
        parts = [p.strip().strip('"\'') for p in inner.split(',')]
        parts = [p for p in parts if p]
        if parts:
            actions.append(parts)

    return actions


def compute_step_reward(current_state: Set[Tuple], goal_state: Set[Tuple]) -> float:
    """
    Compute step-level reward based on goal predicate satisfaction.
    R(X_t, a_t) = |X^goal ∩ X_t| / |X^goal|  (Eq. 2)
    """
    if len(goal_state) == 0:
        return 1.0

    satisfied = goal_state.intersection(current_state)
    return len(satisfied) / len(goal_state)


def compute_constraint_cost(
    current_state: Set[Tuple],
    action: List[str],
    lambda_coef: float = 0.5
) -> float:
    """
    Compute constraint cost for precondition violations.
    C(X_t, a_t) = 1[prec(a_t) ⊄ X_t] + λ * 1[¬valid(X_{t+1})]  (Eq. 3)
    """
    if not action or len(action) == 0:
        return 1.0  # Invalid action format

    action_name = action[0]
    args = action[1:]

    # Check precondition violation
    can_perform, reason = can_perform_action(current_state, action_name, *args)
    precond_violation = 0.0 if can_perform else 1.0

    # Check if resulting state would be valid
    validity_violation = 0.0
    if can_perform:
        try:
            new_state = apply_action(current_state, action_name, *args)
            # Basic validity check - state should not be empty
            if len(new_state) == 0:
                validity_violation = 1.0
        except Exception:
            validity_violation = 1.0

    return precond_violation + lambda_coef * validity_violation


def compute_trajectory_rewards(
    initial_state: Set[Tuple],
    goal_state: Set[Tuple],
    actions: List[List[str]],
    lambda_coef: float = 0.5
) -> Tuple[List[float], List[float], float, float]:
    """
    Compute rewards and constraints for a trajectory.

    Returns:
        step_rewards: List of R(X_t, a_t) for each step
        step_constraints: List of C(X_t, a_t) for each step
        total_reward: Sum of step rewards
        total_constraint: Sum of constraint costs
    """
    step_rewards = []
    step_constraints = []

    current_state = initial_state.copy()

    for action in actions:
        # Compute constraint cost before action
        constraint = compute_constraint_cost(current_state, action, lambda_coef)
        step_constraints.append(constraint)

        # Try to apply action
        can_perform, _ = can_perform_action(current_state, action[0], *action[1:])
        if can_perform:
            try:
                current_state = apply_action(current_state, action[0], *action[1:])
            except Exception:
                pass

        # Compute reward after action
        reward = compute_step_reward(current_state, goal_state)
        step_rewards.append(reward)

    total_reward = sum(step_rewards) if step_rewards else 0.0
    total_constraint = sum(step_constraints) if step_constraints else 0.0

    return step_rewards, step_constraints, total_reward, total_constraint


class CPOTrainer:
    """Online CPO Trainer for PDDL Planning."""

    def __init__(self, config: CPOConfig):
        self.config = config
        self.accelerator = Accelerator(mixed_precision="bf16")

        # Load model and tokenizer
        print(f"Loading model from {config.model_path}"
              + (f" (subfolder={config.model_subfolder})" if config.model_subfolder else "")
              + " ...")
        load_kwargs = {"trust_remote_code": True}
        if config.model_subfolder:
            load_kwargs["subfolder"] = config.model_subfolder

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            use_fast=True,
            **load_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        )
        # Enable gradient checkpointing permanently (no toggling to avoid DDP conflicts)
        self.model.gradient_checkpointing_enable()

        # Reference model for KL penalty (frozen copy)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Prepare with accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.ref_model = self.accelerator.prepare(self.ref_model)

        # Load prompt templates
        base_dir = str(project_root)
        self.system_prompt, self.user_template = load_prompt_templates(base_dir)
        print(f"Loaded prompt templates from {base_dir}/configs/prompts/")

        # Load training data
        self.problems = load_training_problems(config.train_data_path)

        # Metrics tracking
        self.global_step = 0
        self.metrics_history = defaultdict(list)

    def generate_trajectory(self, problem: Dict) -> Tuple[str, str, List[List[str]]]:
        """Generate a trajectory (action sequence) for a problem."""
        # Format prompt using templates
        prompt = format_prompt(
            problem, self.tokenizer,
            self.system_prompt, self.user_template
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.accelerator.device)

        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            # Disable gradient checkpointing for generation (enables KV cache)
            unwrapped_model.gradient_checkpointing_disable()
            # Also set the flag explicitly
            if hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'gradient_checkpointing'):
                unwrapped_model.model.gradient_checkpointing = False
            unwrapped_model.eval()
            outputs = unwrapped_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,  # Enable KV cache for fast generation
            )
            unwrapped_model.train()
            # Re-enable gradient checkpointing for training
            unwrapped_model.gradient_checkpointing_enable()
            if hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'gradient_checkpointing'):
                unwrapped_model.model.gradient_checkpointing = True

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):]  # Remove prompt from response

        actions = parse_action_sequence(response)
        return prompt, response, actions

    def compute_policy_loss(
        self,
        prompt: str,
        response: str,
        actions: List[List[str]],
        rewards: List[float],
        constraints: List[float],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute CPO policy loss with reward maximization and constraint satisfaction.
        """
        full_text = prompt + response

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=16384
        ).to(self.accelerator.device)

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.accelerator.device)

        prompt_len = prompt_inputs.input_ids.shape[1]

        # Get log probabilities from policy and reference
        with torch.no_grad():
            unwrapped_ref = self.accelerator.unwrap_model(self.ref_model)
            ref_outputs = unwrapped_ref(**inputs)
            ref_logits = ref_outputs.logits

        policy_outputs = self.model(**inputs)
        policy_logits = policy_outputs.logits

        # Compute log probs for response tokens only
        response_logits = policy_logits[:, prompt_len-1:-1, :]
        response_labels = inputs.input_ids[:, prompt_len:]

        ref_response_logits = ref_logits[:, prompt_len-1:-1, :]

        # Log probabilities
        policy_log_probs = F.log_softmax(response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)

        # Gather log probs for actual tokens
        policy_token_log_probs = torch.gather(
            policy_log_probs, 2, response_labels.unsqueeze(-1)
        ).squeeze(-1)
        ref_token_log_probs = torch.gather(
            ref_log_probs, 2, response_labels.unsqueeze(-1)
        ).squeeze(-1)

        # KL divergence penalty
        kl_div = (policy_token_log_probs - ref_token_log_probs).mean()

        # Compute advantage-weighted policy gradient
        total_reward = sum(rewards) if rewards else 0.0
        total_constraint = sum(constraints) if constraints else 0.0

        # Advantage = reward - constraint penalty
        advantage = total_reward - self.config.constraint_threshold * total_constraint

        # Policy gradient loss (negative because we maximize reward)
        pg_loss = -advantage * policy_token_log_probs.mean()

        # Total loss = policy gradient + KL penalty
        loss = pg_loss + self.config.beta * kl_div

        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "total_reward": total_reward,
            "total_constraint": total_constraint,
            "advantage": advantage,
            "num_actions": len(actions),
        }

        return loss, metrics

    def train_step(self, problem: Dict) -> Tuple[Optional[torch.Tensor], Dict]:
        """Perform forward pass on a single problem, return loss for accumulation."""
        self.model.train()

        # Get initial and goal states
        initial_state = set(tuple(p) for p in problem.get("initial_state", []))
        goal_state = set(tuple(p) for p in problem.get("goal_state", []))

        # Generate trajectory
        prompt, response, actions = self.generate_trajectory(problem)

        if not actions:
            return None, {}

        # Compute rewards and constraints
        step_rewards, step_constraints, total_reward, total_constraint = \
            compute_trajectory_rewards(
                initial_state, goal_state, actions,
                self.config.lambda_constraint
            )

        # Compute loss
        loss, metrics = self.compute_policy_loss(
            prompt, response, actions,
            step_rewards, step_constraints
        )

        return loss, metrics

    def train(self):
        """Main training loop with gradient accumulation."""
        print(f"\n{'='*60}")
        print("Starting Online CPO Training")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_path}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Beta (KL penalty): {self.config.beta}")
        print(f"Constraint threshold: {self.config.constraint_threshold}")
        print(f"{'='*60}\n")

        os.makedirs(self.config.output_dir, exist_ok=True)
        grad_accum = self.config.gradient_accumulation_steps

        for epoch in range(self.config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")

            # Shuffle problems
            random.shuffle(self.problems)

            num_steps = len(self.problems) // grad_accum
            epoch_metrics = defaultdict(list)

            progress_bar = tqdm(range(num_steps), desc=f"Epoch {epoch + 1}")

            for step_idx in progress_bar:
                self.optimizer.zero_grad()
                accumulated_loss = 0.0
                step_metrics = defaultdict(list)

                # Accumulate gradients over multiple problems
                for accum_idx in range(grad_accum):
                    problem_idx = step_idx * grad_accum + accum_idx
                    if problem_idx >= len(self.problems):
                        break

                    problem = self.problems[problem_idx]
                    loss, metrics = self.train_step(problem)

                    if loss is not None:
                        # Scale loss for accumulation
                        scaled_loss = loss / grad_accum
                        self.accelerator.backward(scaled_loss)
                        accumulated_loss += loss.item()

                        for k, v in metrics.items():
                            step_metrics[k].append(v)

                # Optimizer step after accumulation
                if accumulated_loss > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                # Track metrics
                for k, v in step_metrics.items():
                    avg_v = sum(v) / len(v) if v else 0.0
                    epoch_metrics[k].append(avg_v)
                    self.metrics_history[k].append(avg_v)

                epoch_metrics["loss"].append(accumulated_loss / grad_accum)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    recent_loss = epoch_metrics["loss"][-self.config.logging_steps:]
                    recent_reward = epoch_metrics.get("total_reward", [0])[-self.config.logging_steps:]
                    avg_loss = sum(recent_loss) / len(recent_loss) if recent_loss else 0
                    avg_reward = sum(recent_reward) / len(recent_reward) if recent_reward else 0
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "reward": f"{avg_reward:.3f}"
                    })

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

            # Epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            for k, v in epoch_metrics.items():
                avg_v = sum(v) / len(v) if v else 0.0
                print(f"  {k}: {avg_v:.4f}")

        # Save final model
        self.save_checkpoint("final")
        print(f"\nTraining complete! Model saved to {self.config.output_dir}")

    def save_checkpoint(self, name: str):
        """Save a checkpoint."""
        save_path = os.path.join(self.config.output_dir, name)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save config
        with open(os.path.join(save_path, "cpo_config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=2)

        print(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Online CPO Training for PDDL Planning")

    # Model args (defaults assume the released SFT checkpoint loaded as a local dir)
    parser.add_argument("--model_path", type=str,
                        default="Self-CriTeach/SCT",
                        help="HF repo id or local path to the SFT-trained model to initialize CPO from. "
                             "If using Self-CriTeach/SCT, pair with --model_subfolder.")
    parser.add_argument("--model_subfolder", type=str, default=None,
                        help="Subfolder within model_path (e.g., 'Qwen3-4B'). Optional.")
    parser.add_argument("--train_data_path", type=str,
                        default="./data/train",
                        help="Local path to training JSONLs (download via Self-CriTeach/pddl-planning-data).")
    parser.add_argument("--output_dir", type=str,
                        default="./checkpoints/cpo",
                        help="Where to save trained CPO checkpoints.")

    # Training args
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_samples_per_problem", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)

    # CPO args
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--constraint_threshold", type=float, default=0.25)
    parser.add_argument("--lambda_constraint", type=float, default=0.5)

    # Generation args
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Logging
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()

    config = CPOConfig(
        model_path=args.model_path,
        model_subfolder=args.model_subfolder,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_samples_per_problem=args.num_samples_per_problem,
        learning_rate=args.learning_rate,
        beta=args.beta,
        constraint_threshold=args.constraint_threshold,
        lambda_constraint=args.lambda_constraint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

    trainer = CPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
