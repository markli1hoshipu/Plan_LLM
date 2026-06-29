"""
Evaluation metrics for PDDL planning tasks.

This module provides metrics for evaluating predicted action sequences
against ground truth actions.
"""

from typing import List, Tuple
from collections import Counter


def compute_lcs_score(
    ground_truth_actions: List[List[str]],
    prediction_actions: List[List[str]]
) -> float:
    """
    Compute Longest Common Subsequence (LCS) score between ground truth and predicted actions.

    The LCS score measures the order-preserving overlap between two action sequences,
    normalized by the maximum length.

    Args:
        ground_truth_actions: Ground truth action sequence, where each action is a list of strings
        prediction_actions: Predicted action sequence, where each action is a list of strings

    Returns:
        LCS score in range [0.0, 1.0], where 1.0 indicates perfect order-preserving match

    Example:
        >>> gt = [["pick", "A"], ["stack", "A", "B"]]
        >>> pred = [["pick", "A"], ["put", "A"], ["stack", "A", "B"]]
        >>> compute_lcs_score(gt, pred)
        0.667  # 2 out of 3 actions match in order
    """
    actions1 = [tuple(action) for action in ground_truth_actions]
    actions2 = [tuple(action) for action in prediction_actions]
    max_len = max(len(actions1), len(actions2)) or 1  # Avoid division by zero

    def lcs_length(X: List[Tuple[str, ...]], Y: List[Tuple[str, ...]]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs_score = lcs_length(actions1, actions2) / max_len
    return lcs_score


def compute_action_score(
    ground_truth_actions: List[List[str]],
    prediction_actions: List[List[str]]
) -> float:
    """
    Compute exact action overlap score (order-agnostic).

    This metric counts how many actions appear in both sequences, treating
    actions as multisets (order doesn't matter, but duplicates count separately).

    Args:
        ground_truth_actions: Ground truth action sequence
        prediction_actions: Predicted action sequence

    Returns:
        Action overlap score in range [0.0, 1.0], normalized by max sequence length

    Example:
        >>> gt = [["pick", "A"], ["stack", "A", "B"]]
        >>> pred = [["stack", "A", "B"], ["pick", "A"]]
        >>> compute_action_score(gt, pred)
        1.0  # Both actions match, order doesn't matter
    """
    actions1 = [tuple(action) for action in ground_truth_actions]
    actions2 = [tuple(action) for action in prediction_actions]
    max_len = max(len(actions1), len(actions2)) or 1  # Avoid division by zero

    # Count exact action overlap using multiset intersection
    exact_common = Counter(actions1) & Counter(actions2)
    exact_score = sum(exact_common.values()) / max_len

    return exact_score


def compute_vague_action_score(
    ground_truth_actions: List[List[str]],
    prediction_actions: List[List[str]]
) -> float:
    """
    Compute keyword-level overlap score.

    This is a lenient metric that measures overlap at the individual keyword level,
    useful for partial credit when action structure is close but not exact.

    Args:
        ground_truth_actions: Ground truth action sequence
        prediction_actions: Predicted action sequence

    Returns:
        Keyword overlap score in range [0.0, 1.0]

    Example:
        >>> gt = [["pick", "A"], ["stack", "A", "B"]]
        >>> pred = [["pick", "B"], ["stack", "B", "C"]]
        >>> compute_vague_action_score(gt, pred)
        0.5  # "pick" and "stack" match, but objects differ
    """
    actions1 = [tuple(action) for action in ground_truth_actions]
    actions2 = [tuple(action) for action in prediction_actions]

    def flatten_keywords(actions: List[Tuple[str, ...]]) -> List[str]:
        """Flatten all keywords from actions into a single list."""
        return [word for action in actions for word in action]

    keywords1 = flatten_keywords(actions1)
    keywords2 = flatten_keywords(actions2)
    total_keywords = len(keywords1) + len(keywords2)

    if total_keywords > 0:
        common_keywords = Counter(keywords1) & Counter(keywords2)
        vague_score = 2 * sum(common_keywords.values()) / total_keywords
    else:
        vague_score = 0.0

    return vague_score
