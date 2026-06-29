"""
Prompt Template Manager for HSP PDDL

This module provides utilities for loading and managing prompt templates.

Usage:
    from plan_llm.utils.prompt_manager import PromptManager

    # Initialize
    pm = PromptManager()

    # Load base template
    template = pm.get_base_template("planning_template.md")

    # Load task-specific template
    reorder_prompt = pm.get_task_template("reorder", "default.md")

    # Format template with variables
    prompt = pm.format_template(
        template,
        static_state="...",
        dynamic_state="...",
        goal="..."
    )
"""

from pathlib import Path
from typing import Dict, Optional, List
import os


class PromptManager:
    """Manager for loading and formatting prompt templates."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the PromptManager.

        Args:
            base_path: Base path to configs directory. If None, uses project default.
        """
        if base_path is None:
            # Default to project configs directory
            project_root = Path(__file__).parent.parent
            base_path = project_root / "configs"

        self.base_path = Path(base_path)
        self.prompts_path = self.base_path / "prompts"
        self.eval_path = self.base_path / "evaluation"
        self.training_path = self.base_path / "training"

    def get_base_template(self, template_name: str) -> str:
        """
        Load a base template from configs/prompts/base_templates/.

        Args:
            template_name: Name of the template file (e.g., "planning_template.md")

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.prompts_path / "base_templates" / template_name
        return self._load_template(template_path)

    def get_task_template(self, task: str, variant: str = "default.md") -> str:
        """
        Load a task-specific template from configs/prompts/task_templates/{task}/.

        Args:
            task: Task name (e.g., "reorder", "stack")
            variant: Template variant name (default: "default.md")

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.prompts_path / "task_templates" / task / variant
        return self._load_template(template_path)

    def get_eval_template(self, template_name: str) -> str:
        """
        Load an evaluation template from configs/evaluation/.

        Args:
            template_name: Name of the template file

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.eval_path / template_name
        return self._load_template(template_path)

    def get_training_template(self, template_name: str) -> str:
        """
        Load a training template from configs/training/.

        Args:
            template_name: Name of the template file

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.training_path / template_name
        return self._load_template(template_path)

    def list_base_templates(self) -> List[str]:
        """List all available base templates."""
        return self._list_templates(self.prompts_path / "base_templates")

    def list_task_templates(self, task: str) -> List[str]:
        """List all template variants for a specific task."""
        task_path = self.prompts_path / "task_templates" / task
        if not task_path.exists():
            return []
        return self._list_templates(task_path)

    def list_tasks(self) -> List[str]:
        """List all available tasks with templates."""
        task_templates_path = self.prompts_path / "task_templates"
        if not task_templates_path.exists():
            return []
        return [d.name for d in task_templates_path.iterdir() if d.is_dir()]

    def format_template(self, template: str, **kwargs) -> str:
        """
        Format a template string with provided variables.

        Args:
            template: Template string with {variable} placeholders
            **kwargs: Variables to substitute

        Returns:
            Formatted template string

        Example:
            >>> pm = PromptManager()
            >>> template = pm.get_base_template("planning_template.md")
            >>> prompt = pm.format_template(
            ...     template,
            ...     static_state="obj1, obj2",
            ...     dynamic_state="on(obj1, table)",
            ...     goal="on(obj2, obj1)"
            ... )
        """
        return template.format(**kwargs)

    def create_task_template(self, task: str, variant: str, content: str) -> Path:
        """
        Create a new task-specific template.

        Args:
            task: Task name
            variant: Template variant name (e.g., "variant_1.md")
            content: Template content

        Returns:
            Path to created template
        """
        task_dir = self.prompts_path / "task_templates" / task
        task_dir.mkdir(parents=True, exist_ok=True)

        template_path = task_dir / variant
        template_path.write_text(content)
        return template_path

    @staticmethod
    def _load_template(template_path: Path) -> str:
        """Load a template file."""
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}\n"
                f"Available templates can be listed using list_*_templates() methods."
            )
        return template_path.read_text()

    @staticmethod
    def _list_templates(directory: Path) -> List[str]:
        """List all template files in a directory."""
        if not directory.exists():
            return []
        return sorted([
            f.name for f in directory.iterdir()
            if f.is_file() and f.suffix in ['.md', '.txt']
        ])


# Convenience function for quick access
def load_template(template_type: str, name: str, task: Optional[str] = None) -> str:
    """
    Convenience function to quickly load a template.

    Args:
        template_type: Type of template ("base", "task", "eval", "training")
        name: Template name
        task: Task name (required if template_type is "task")

    Returns:
        Template content

    Example:
        >>> template = load_template("base", "planning_template.md")
        >>> reorder_template = load_template("task", "default.md", task="reorder")
    """
    pm = PromptManager()

    if template_type == "base":
        return pm.get_base_template(name)
    elif template_type == "task":
        if task is None:
            raise ValueError("task argument is required for task templates")
        return pm.get_task_template(task, name)
    elif template_type == "eval":
        return pm.get_eval_template(name)
    elif template_type == "training":
        return pm.get_training_template(name)
    else:
        raise ValueError(
            f"Unknown template_type: {template_type}. "
            f"Must be 'base', 'task', 'eval', or 'training'"
        )


if __name__ == "__main__":
    # Example usage
    pm = PromptManager()

    print("Available base templates:")
    for template in pm.list_base_templates():
        print(f"  - {template}")

    print("\nAvailable tasks:")
    for task in pm.list_tasks():
        print(f"  - {task}")
        variants = pm.list_task_templates(task)
        for variant in variants:
            print(f"    - {variant}")

    # Load and display a template
    print("\n" + "="*60)
    print("Example: Loading planning template")
    print("="*60)
    try:
        template = pm.get_base_template("planning_template.md")
        print(template[:500] + "...")
    except FileNotFoundError as e:
        print(f"Template not found: {e}")
