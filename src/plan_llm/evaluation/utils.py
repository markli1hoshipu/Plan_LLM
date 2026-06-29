"""
Utility functions for evaluation.
"""

import re
import ast
from typing import Optional, List


def extract_answer(output: str) -> Optional[str]:
    """
    Extract the action sequence from model output enclosed in <FINAL>...</FINAL> tags.

    Args:
        output: Raw model output text

    Returns:
        Extracted answer string with proper quote formatting, or None if no match found

    Example:
        >>> output = "Here's my plan: <FINAL>[pick, A, 1], [stack, A, B, 0]</FINAL>"
        >>> extract_answer(output)
        '["pick", "A", "1"], ["stack", "A", "B", "0"]'
    """
    # Pattern ensures no nested <FINAL> inside the capture
    matches = re.findall(r'<FINAL>((?:(?!<FINAL>).)*?)</FINAL>', output, re.DOTALL)
    if matches:
        # Add quotes around unquoted words for valid JSON parsing
        return re.sub(r'([^,\[\]\s]+)', r'"\1"', matches[-1])  # last match only
    return None


def parse_actions(answer_str: Optional[str]) -> List[List[str]]:
    """
    Parse action string into list of action lists.

    Args:
        answer_str: String representation of actions (e.g., from extract_answer)

    Returns:
        List of actions, where each action is a list of strings.
        Returns empty list if parsing fails.

    Example:
        >>> answer_str = '["pick", "A", "1"], ["stack", "A", "B", "0"]'
        >>> parse_actions(answer_str)
        [['pick', 'A', '1'], ['stack', 'A', 'B', '0']]
    """
    if answer_str is None:
        return []

    try:
        return list(ast.literal_eval(answer_str))
    except (ValueError, SyntaxError):
        return []
