"""
PDDL Action Validation Engine

This module implements a PDDL simulator that validates action sequences
by checking preconditions and applying effects. It's used for post-processing
evaluation results to determine if generated plans are actually executable.
"""

import inspect
from typing import Set, Tuple, Dict, Callable, Any, List


# Parameters per action
ACTION_PARAM_MAP = {
    'pick-up':           ['b1', 't1', 'r1'],
    'stack':             ['b1', 'b2', 'r1'],
    'unstack':           ['b1', 'b2', 'r1'],
    'put-down':          ['b1', 't1', 'r1'],
    'align':             ['b1', 'b2', 't1', 'r1'],
    'place_smaller_obj': ['b1', 'b2', 'r1'],
    'remove_smaller_obj':['b1', 'b2', 'r1'],
    'pick-and-rotate':   ['b1', 't1', 'r1'],
    'rotate':            ['b1', 'b2', 't1', 'r1'],
}

# Preconditions per action
ACTION_PRECONDITIONS = {
    'pick-up': lambda b1, t1, r1: {
        ('obj', b1), ('robot', r1), ('on-table', b1, t1),
        ('top', b1), ('hand_free', r1), ('table', t1)
    },
    'stack': lambda b1, b2, r1: {
        ('obj', b1), ('top', b1), ('holding', b1, r1),
        ('robot', r1), ('top', b2), ('obj', b2)
    },
    'unstack': lambda b1, b2, r1: {
        ('obj', b1), ('robot', r1), ('top', b1),
        ('above', b1, b2), ('hand_free', r1), ('obj', b2)
    },
    'put-down': lambda b1, t1, r1: {
        ('obj', b1), ('holding', b1, r1),
        ('top', b1), ('robot', r1), ('table', t1)
    },
    'place_smaller_obj': lambda b1, b2, r1: {
        ('obj', b1), ('top', b1), ('holding', b1, r1),
        ('robot', r1), ('top', b2), ('obj', b2), ('smaller', b1, b2)
    },
    'remove_smaller_obj': lambda b1, b2, r1: {
        ('obj', b1), ('robot', r1), ('top', b1),
        ('above', b1, b2), ('hand_free', r1), ('obj', b2), ('smaller', b1, b2)
    },
    'pick-and-rotate': lambda b1, t1, r1: {
        ('obj', b1), ('robot', r1), ('on-table', b1, t1),
        ('top', b1), ('nothing_beside', b1), ('hand_free', r1), ('table', t1)
    },
    'rotate': lambda b1, b2, t1, r1: {
        ('table', t1), ('robot', r1), ('obj', b2), ('obj', b1),
        ('on-table', b2, t1), ('holding', b1, r1)
    },
}

# Effects per action
ACTION_EFFECTS = {
    'pick-up': lambda b1, t1, r1: {
        'add': {('holding', b1, r1)},
        'del': {('hand_free', r1), ('on-table', b1, t1)}
    },
    'stack': lambda b1, b2, r1: {
        'add': {('above', b1, b2), ('hand_free', r1)},
        'del': {('top', b2), ('holding', b1, r1)}
    },
    'unstack': lambda b1, b2, r1: {
        'add': {('top', b2), ('holding', b1, r1)},
        'del': {('hand_free', r1), ('above', b1, b2)}
    },
    'put-down': lambda b1, t1, r1: {
        'add': {('hand_free', r1), ('on-table', b1, t1)},
        'del': {('holding', b1, r1)}
    },
    'place_smaller_obj': lambda b1, b2, r1: {
        'add': {('above', b1, b2), ('hand_free', r1)},
        'del': {('top', b2), ('holding', b1, r1)}
    },
    'remove_smaller_obj': lambda b1, b2, r1: {
        'add': {('top', b2), ('holding', b1, r1)},
        'del': {('hand_free', r1), ('above', b1, b2)}
    },
    'rotate': lambda b1, b2, t1, r1: {
        'add': {('aligned', b1, b2)},
        'del': set()
    },
}


def find_related_predicates(preconditions: Set[Tuple], state: Set[Tuple]) -> Tuple[Set[Tuple], Dict]:
    """
    Find predicates in state that share the first parameter (after predicate name)
    with any missing predicates from preconditions.

    Args:
        preconditions: Set of required predicates
        state: Current world state

    Returns:
        Tuple of (missing predicates, related predicates dict)
    """
    missing = preconditions - state
    related = {}

    for m in missing:
        if len(m) > 1:  # ensure it has a parameter to match
            first_param = m[1]  # first parameter after predicate name
            matches = [s for s in state if len(s) > 1 and s[1] == first_param]
            matches2 = [s for s in state if len(s) > 2 and s[2] == first_param]
            matches = matches + matches2
            related[first_param] = matches

    return missing, related


def call_with_used_args(fn: Callable, param_dict: Dict[str, Any]) -> Any:
    """
    Call lambda with only the arguments it uses.

    Args:
        fn: Lambda function to call
        param_dict: Dictionary of parameter name -> value

    Returns:
        Result of calling fn with matching parameters
    """
    sig = inspect.signature(fn)
    used_keys = sig.parameters.keys()
    args = {k: v for k, v in param_dict.items() if k in used_keys}
    return fn(**args)


def can_perform_action(state: Set[Tuple], action: str, *args) -> Tuple[bool, str]:
    """
    Check if an action can be performed in the current state.

    Args:
        state: Current world state (set of predicates)
        action: Action name
        *args: Action parameters

    Returns:
        Tuple of (can_perform, reason) where reason is empty if can_perform is True
    """
    if action not in ACTION_PRECONDITIONS:
        return False, "Action doesn't meet preconditions. (illegal action type)"

    param_names = ACTION_PARAM_MAP[action]
    if len(args) != len(param_names):
        return False, "Action not implemented. (incorrect number of params)"

    param_dict = dict(zip(param_names, args))
    preconditions = call_with_used_args(ACTION_PRECONDITIONS[action], param_dict)

    if not preconditions.issubset(state):
        missing_preds, related_preds = find_related_predicates(preconditions, state)
        return False, f"Missing predicates: {missing_preds} ; Reality: {related_preds}"
    else:
        return True, ""


def apply_action(state: Set[Tuple], action: str, *args) -> Set[Tuple]:
    """
    Apply an action to the current state, returning the new state.

    Args:
        state: Current world state
        action: Action name
        *args: Action parameters

    Returns:
        New state after applying action

    Raises:
        ValueError: If action cannot be performed in current state
    """
    can, reason = can_perform_action(state, action, *args)
    if not can:
        raise ValueError(f"Action {action}{args} is not applicable in the current state. {reason}")

    param_names = ACTION_PARAM_MAP[action]
    param_dict = dict(zip(param_names, args))

    effects = call_with_used_args(ACTION_EFFECTS[action], param_dict)
    return (state - effects['del']) | effects['add']


def process_actions(initial_state: Set[Tuple], actions: List[List[str]]) -> Tuple[bool, int, int, Set[Tuple], str]:
    """
    Process a sequence of actions and validate executability.

    This is the main function for validating plans. It simulates executing each action
    step-by-step and tracks where failures occur.

    Args:
        initial_state: Starting world state (set of tuples)
        actions: List of actions, where each action is [action_name, param1, param2, ...]

    Returns:
        Tuple of:
        - syntax_valid (bool): Whether the plan has valid syntax
        - num_executed (int): Number of actions successfully executed
        - num_total (int): Total number of actions in plan
        - final_state (Set[Tuple]): State after last successful action
        - fail_reason (str): Reason for failure (empty if fully successful)

    Example:
        >>> initial = {('obj', 'A'), ('hand_free', 'robot1')}
        >>> plan = [['pick-up', 'A', 'table1', 'robot1']]
        >>> syntax_valid, executed, total, state, reason = process_actions(initial, plan)
        >>> print(executed, total)  # 1, 1 if successful
    """
    # Input validation (returns False instead of asserting)
    if not isinstance(initial_state, set):
        return False, 0, -1, initial_state, "✗ initial_state must be a set"
    if not isinstance(actions, list):
        return False, 0, -1, initial_state, "✗ plan must be a list"
    if not all(isinstance(p, tuple) for p in initial_state):
        return False, 0, -1, initial_state, "✗ initial_state must contain tuples"
    if not all(isinstance(step, list) and len(step) > 0 and isinstance(step[0], str) for step in actions):
        return False, 0, -1, initial_state, "✗ plan must contain (action, [params]) lists"

    # Start simulation
    current_state = initial_state.copy()  # Avoid modifying the original
    for i, step in enumerate(actions):
        try:
            action_name = step[0]
            args = step[1:]

            can, reason = can_perform_action(current_state, action_name, *args)
            if can:
                current_state = apply_action(current_state, action_name, *args)
            else:
                return True, i, len(actions), current_state, f"✗ Action {i+1}: {action_name}{args} is not applicable ; " + reason
        except Exception as e:
            return True, i, len(actions), current_state, f"✗ Error exception processing action {i+1}: {step} - {str(e)}"

    # All actions executed successfully
    return True, len(actions), len(actions), current_state, ""


def compute_logical_divergence(ground_truth: Set[Tuple], prediction: Set[Tuple]) -> float:
    """
    Compute a similarity score between ground truth and predicted logical states.

    This metric measures how close the predicted state is to the ground truth.
    Score is 1.0 if they match perfectly, and decreases with every missing or redundant predicate.

    Args:
        ground_truth: Expected final state
        prediction: Actual final state

    Returns:
        Similarity score in range [0.0, 1.0]

    Example:
        >>> gt = {('on', 'A', 'B'), ('clear', 'A')}
        >>> pred = {('on', 'A', 'B')}  # Missing one predicate
        >>> score = compute_logical_divergence(gt, pred)
        >>> print(score)  # 0.667 (2 out of 3 total elements match)
    """
    gt_set = set(ground_truth)
    pred_set = set(prediction)

    # Calculate mismatches
    missing = gt_set - pred_set         # in GT but not in prediction (false negatives)
    redundant = pred_set - gt_set       # in prediction but not in GT (false positives)

    # Total unique elements across both sets (for normalization)
    total_elements = len(gt_set.union(pred_set))

    if total_elements == 0:
        return 1.0

    # Penalize both missing and redundant predicates equally
    divergence = len(missing) + len(redundant)
    score = 1 - divergence / total_elements
    return max(0.0, score)  # Ensure non-negative
