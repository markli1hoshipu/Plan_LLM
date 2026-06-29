"""
Prompt templates for PDDL planning training data generation.
"""

system_prompt = """You are a robot assistant. Your task is to generate a plan given the initial and goal state. A plan is a sequence of actions.
"""

user_prompt_short = """
Your task is to predict a set of actions that arrive at the goal state starting the initial state. A state is defined by a set of predicates. Predicates can be static (i.e. describe invariant properties of the environment that do not change over time) or dynamic.

Possible Predicates: robot, table, obj, on-table, holding, hand_free, top, above, aligned

Possible Actions:
- [pick-up, b, t, r]: pick obj b from table t with robot r; requires [top, b], [on-table, b, t], [hand_free, r]; effects: adds [holding, b, r], removes [on-table, b, t], [hand_free, r]
- [put-down, b, t, r]: place obj b on table t with robot r; requires [holding, b, r]; effects: adds [on-table, b, t], [hand_free, r], [top, b], removes [holding, b, r]
- [unstack, b1, b2, r]: remove obj b1 from obj b2 with robot r; requires [top, b1], [above, b1, b2], [hand_free, r]; effects: adds [holding, b1, r], [top, b2], removes [above, b1, b2], [hand_free, r]
- [stack, b1, b2, r]: place obj b1 on obj b2 with robot r; requires [holding, b1, r], [top, b2]; effects: adds [above, b1, b2], [hand_free, r], [top, b1], removes [holding, b1, r], [top, b2]
- [rotate, b1, b2, t, r]: rotate ibj b1 to align direction with obj b2 on table t with robot r; requires [holding, b1, r], [on-table, b2, t]; effects: adds [aligned, b1, b2], removes none

# Example Input/Output 1
Static predicates: [[box, 3], [table, 1], [box, 7], [box, 5], [box, 6], [box, 4], [box, 2], [robot, 0], [box, 8]]
Initial state: [[top, 4], [above, 3, 7], [top, 2], [on-table, 5, 1], [top, 8], [on-table, 6, 1], [hand_free, 0], [top, 3], [on-table, 8, 1], [on-table, 4, 1], [on-table, 7, 1], [on-table, 2, 1], [top, 5], [top, 6]]
Goal state: [[top, 2], [above, 7, 5], [above, 5, 8], [on-table, 6, 1], [hand_free, 0], [top, 3], [above, 3, 4], [on-table, 4, 1], [on-table, 8, 1], [top, 7], [on-table, 2, 1], [top, 6]]
Output actions: <FINAL>[[unstack, 3, 7, 0], [stack, 3, 4, 0], [pick-up, 5, 1, 0], [stack, 5, 8, 0], [pick-up, 7, 1, 0], [stack, 7, 5, 0]]</FINAL>

(Optional) You may optionally provide reasoning before the final plan.

# Problem to Solve:
Static predicates: {}
Initial State: {}
Goal State: {}

Always output the final plan inside <FINAL> ... </FINAL>
""".format


user_prompt_dynamic = """
{}

(Optional) You may optionally provide reasoning before the final plan.

# Problem to Solve:
Static predicates: {}
Initial State: {}
Goal State: {}

Always output the final plan inside <FINAL> ... </FINAL>
""".format


response_cot = """
The initial state is: {}

{}

Finally the goal state is: {}

Output actions: <FINAL>{}</FINAL>
""".format


response_ans = """
Output actions: <FINAL>{}</FINAL>
""".format
