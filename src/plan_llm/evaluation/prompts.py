"""
Prompt loading utilities for evaluation.
"""

from pathlib import Path
from typing import Optional


def load_prompt(prompt_name: str, prompts_dir: Optional[Path] = None) -> str:
    """
    Load a prompt from the configs/prompts directory.

    Args:
        prompt_name: Name of the prompt file (e.g., "eval_system_prompt.md")
        prompts_dir: Optional custom prompts directory path.
                     If None, uses project default.

    Returns:
        Prompt content as string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    if prompts_dir is None:
        # Default to project configs/prompts directory
        project_root = Path(__file__).parent.parent.parent
        prompts_dir = project_root / "configs" / "prompts"

    prompt_path = prompts_dir / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Looking in directory: {prompts_dir}"
        )

    return prompt_path.read_text()


def get_eval_prompts():
    """
    Load evaluation system and user prompt templates.

    Returns:
        Tuple of (system_prompt, user_prompt_template_function)
    """
    system_prompt_content = load_prompt("eval_system_prompt.md")
    user_prompt_content = load_prompt("eval_user_prompt_template.md")

    # Extract the actual prompt content (skip variable assignments if present)
    # The user prompt template contains a .format placeholder function
    system_prompt = extract_prompt_content(system_prompt_content)
    user_prompt_template = extract_user_prompt_function(user_prompt_content)

    return system_prompt, user_prompt_template


def extract_prompt_content(content: str) -> str:
    """
    Extract prompt text from file content, handling Python variable assignments.

    Args:
        content: Raw file content

    Returns:
        Cleaned prompt text
    """
    # Remove Python variable assignment if present (e.g., 'system_prompt = """..."""')
    content = content.strip()

    # Check if content starts with a variable assignment
    if '=' in content and ('"""' in content or "'''" in content):
        # Extract content between triple quotes
        if '"""' in content:
            parts = content.split('"""')
            if len(parts) >= 2:
                return parts[1].strip()
        elif "'''" in content:
            parts = content.split("'''")
            if len(parts) >= 2:
                return parts[1].strip()

    return content


def extract_user_prompt_function(content: str):
    """
    Extract user prompt template function from file content.

    The template should have a .format method for string substitution.

    Args:
        content: Raw file content with template

    Returns:
        Function that formats the template with static_state, initial_state, goal_state
    """
    prompt_text = extract_prompt_content(content)

    # Return a formatting function
    def format_user_prompt(static_state, initial_state, goal_state):
        return prompt_text.format(
            static_state=static_state,
            initial_state=initial_state,
            goal_state=goal_state
        )

    return format_user_prompt
