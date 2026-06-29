# Chain-of-Thought User Prompt Template

A state is defined by a set of predicates. Predicates can be static (i.e. describe invariant properties of the environment that do not change over time) or dynamic.

Possible Predicates: robot, table, obj, on-table, holding, hand_free, top, above, beside, nothing_beside

Possible Actions:
- [pick-up, b, t, r]: take b from table t; requires [top, b], [on-table, b, t], [hand_free, r]
- [put-down, b, t, r]: place b on table t; requires [holding, b, r]
- [unstack, b1, b2, r]: remove b1 from b2; requires [top, b1], [above, b1, b2], [hand_free, r]
- [stack, b1, b2, r]: place b1 on b2; requires [holding, b1, r], [top, b2]
- [align, b1, b2, t, r]: place beside on table t; requires [holding, b1, r], [on-table, b2, t], [nothing_beside, b2]

## Task Description
Your task is to explain how to predict a set of actions that arrive at the goal state starting the initial state.


# Example Input/Output 1
Static predicates: [[box, 3], [table, 1], [box, 7], [box, 5], [box, 6], [box, 4], [box, 2], [robot, 0], [box, 8]]
Initial state: [[top, 4], [above, 3, 7], [top, 2], [on-table, 5, 1], [top, 8], [on-table, 6, 1], [hand_free, 0], [top, 3], [on-table, 8, 1], [on-table, 4, 1], [on-table, 7, 1], [on-table, 2, 1], [top, 5], [top, 6]]
Goal state: [[top, 2], [above, 7, 5], [above, 5, 8], [on-table, 6, 1], [hand_free, 0], [top, 3], [above, 3, 4], [on-table, 4, 1], [on-table, 8, 1], [top, 7], [on-table, 2, 1], [top, 6]]
Correct plan of actions: <FINAL>[[unstack, 3, 7, 0], [stack, 3, 4, 0], [pick-up, 5, 1, 0], [stack, 5, 8, 0], [pick-up, 7, 1, 0], [stack, 7, 5, 0]]</FINAL>
<REASON>
The planning problem involves reconfiguring blocks from the initial state to the goal state using a sequence of actions. The static environment includes boxes (2, 3, 4, 5, 6, 7, 8), a table (1), and a robot (0). The initial state has multiple stacks: box2 alone, box4 alone, box5 alone, box6 alone, box7 with box3 on top, and box8 alone. The goal state requires: box2 alone, box4 with box3 on top, box6 alone, and a stack with box8 (bottom), box5 (middle), and box7 (top). The robot starts with a free hand.
Below is a detailed chain-of-thought reasoning for each action in the correct plan, explaining why it was chosen, how it changes the state, how preconditions are satisfied, and why alternatives were not selected at each step. After each action, a brief reflection on alternatives is provided.
Initial State Analysis
    State description:
        Stacks:
            Stack A: box2 alone on table (predicates: [top, 2], [on-table, 2, 1])
            Stack B: box4 alone on table ([top, 4], [on-table, 4, 1])
            Stack C: box5 alone on table ([top, 5], [on-table, 5, 1])
            Stack D: box6 alone on table ([top, 6], [on-table, 6, 1])
            Stack E: box7 with box3 on top ([on-table, 7, 1], [above, 3, 7], [top, 3])
            Stack F: box8 alone on table ([top, 8], [on-table, 8, 1])
        Robot: [hand_free, 0] (hand free)
    Goal requirements:
        Stack with box4 and box3 on top ([on-table, 4, 1], [above, 3, 4], [top, 3])
        Stack with box8 (bottom), box5 (middle), box7 (top) ([on-table, 8, 1], [above, 5, 8], [above, 7, 5], [top, 7])
        Box2 and box6 alone (already partially satisfied but must remain untouched)
    Key insight: The only stack with multiple blocks (Stack E: box7 with box3) must be disassembled to move box3 to box4 and repurpose box7 for the new stack. The plan must build the box8-box5-box7 stack from the bottom up (box5 on box8 first, then box7 on box5).
Step 1: Execute [unstack, 3, 7, 0]
    Why chosen: Box3 must be moved to box4 to satisfy [above, 3, 4] in the goal, but it is currently on top of box7. Unstacking is the only way to free box3 for relocation. Additionally, box7 needs to be available as a base for future stacking (e.g., for the box8-box5-box7 stack), so removing box3 exposes box7. This action directly addresses the most constrained part of the goal (relocating box3) and is necessary before any stacking can occur on box7.
    How state changes:
        Preconditions satisfied: [top, 3] (box3 is top of Stack E), [above, 3, 7] (box3 above box7), and [hand_free, 0] (robot hand free) all hold in the initial state.
        Effects:
            [holding, 3, 0] is added (robot now holds box3).
            [hand_free, 0] is removed (hand no longer free).
            [top, 3] and [above, 3, 7] are removed (box3 no longer on stack).
            [top, 7] is added (box7 becomes the new top of Stack E, as it was previously under box3).
        New state:
            Stack E is now box7 alone ([top, 7], [on-table, 7, 1]).
            Robot holds box3 ([holding, 3, 0]).
            All other stacks unchanged (box2, box4, box5, box6, box8 alone).
    Why alternatives not chosen:
        Pick-up actions (e.g., [pick-up, 5, 1, 0] or [pick-up, 8, 1, 0]) are possible since boxes 5, 6, 8, etc., are on table and top. However, picking up another box (e.g., box5) would not free box3 or address the immediate need to relocate it. It might even complicate the plan (e.g., holding box5 while box3 is still blocking box7). Since Stack E is the only multi-block stack, unstacking is the only action that enables progress on the goal's critical path. No other unstack is possible (only box3 has a block above it), so this is the most efficient first move.
Step 2: Execute [stack, 3, 4, 0]
    Why chosen: With box3 held, it must be placed on box4 to achieve the goal's [above, 3, 4] and [top, 3]. Box4 is available (top and alone), making it the ideal base. This completes one part of the goal (the box4-box3 stack) early, reducing future complexity and ensuring box4 is not disturbed later.
    How state changes:
        Preconditions satisfied: [holding, 3, 0] (robot holds box3) and [top, 4] (box4 is top of Stack B) both hold in the post-Step-1 state.
        Effects:
            [holding, 3, 0] is removed (box3 placed).
            [hand_free, 0] is added (hand now free).
            [top, 4] is removed (box4 no longer top).
            [above, 3, 4] and [top, 3] are added (box3 now on top of box4).
        New state:
            Stack B is now box4 with box3 on top ([on-table, 4, 1], [above, 3, 4], [top, 3]), matching the goal for this subpart.
            Robot hand free ([hand_free, 0]).
            Stack E is box7 alone; other stacks unchanged.
    Why alternatives not chosen:
        With box3 held, stacking on other tops (e.g., [stack, 3, 2, 0], [stack, 3, 5, 0], or [stack, 3, 7, 0]) is possible. However, stacking on box2, box5, or box7 would violate the goal (e.g., [above, 3, 2] is not required, and it would require extra steps to fix). Stacking back on box7 would undo progress. Only stacking on box4 directly contributes to the goal, as it is specified in [above, 3, 4]. This choice avoids unnecessary rework and locks in a correct sub-configuration.
Step 3: Execute [pick-up, 5, 1, 0]
    Why chosen: To build the box8-box5-box7 stack, box5 must be placed on box8 first (to achieve [above, 5, 8]). Picking up box5 is necessary to move it to box8. Box5 is currently alone and available, making this a straightforward step toward the goal's stack structure.
    How state changes:
        Preconditions satisfied: [top, 5] (box5 is top of Stack C), [on-table, 5, 1] (box5 on table), and [hand_free, 0] (robot hand free) all hold in the post-Step-2 state.
        Effects:
            [holding, 5, 0] is added (robot now holds box5).
            [hand_free, 0] is removed (hand no longer free).
            [top, 5] and [on-table, 5, 1] are removed (box5 no longer on table or top).
        New state:
            Stack C is empty (box5 held).
            Robot holds box5 ([holding, 5, 0]).
            All other stacks unchanged (e.g., box4 with box3, box7 alone, box8 alone).
    Why alternatives not chosen:
        Pick-up of other boxes (e.g., [pick-up, 7, 1, 0] or [pick-up, 8, 1, 0]) is feasible. However, picking up box7 would prevent it from being used as a base later (since stacking requires a top block), and picking up box8 would make it impossible to stack on it (as [stack] requires a top block to place on). Box5 must be moved before box7 to build the stack bottom-up, so picking up box5 first is optimal. Picking up irrelevant boxes (e.g., box2 or box6) would waste actions since they are already in goal positions.
Step 4: Execute [stack, 5, 8, 0]
    Why chosen: With box5 held, stacking it on box8 creates the [above, 5, 8] relation required in the goal. This forms the foundation for the larger stack (box8-box5), allowing box7 to be placed on top next. Box8 is available (top and alone), so this action directly progresses toward the goal's stack configuration.
    How state changes:
        Preconditions satisfied: [holding, 5, 0] (robot holds box5) and [top, 8] (box8 is top of Stack F) both hold in the post-Step-3 state.
        Effects:
            [holding, 5, 0] is removed (box5 placed).
            [hand_free, 0] is added (hand now free).
            [top, 8] is removed (box8 no longer top).
            [above, 5, 8] and [top, 5] are added (box5 now on top of box8).
        New state:
            New stack: box8 with box5 on top ([on-table, 8, 1], [above, 5, 8], [top, 5]), satisfying part of the goal.
            Robot hand free ([hand_free, 0]).
            Stack B (box4 with box3) and Stack E (box7 alone) unchanged.
    Why alternatives not chosen:
        With box5 held, stacking on other tops (e.g., [stack, 5, 2, 0], [stack, 5, 4, 0], or [stack, 5, 7, 0]) is possible. However, stacking on box4 would disrupt the already correct stack (violating [above, 3, 4]), and stacking on box2 or box7 would not contribute to the goal (e.g., [above, 5, 7] is not required). Only stacking on box8 achieves the necessary [above, 5, 8], and it is the only base that fits the goal's structure (box8 must be the bottom). Alternative stacks would require additional unstacking, increasing plan length.
Step 5: Execute [pick-up, 7, 1, 0]
    Why chosen: To complete the stack, box7 must be placed on box5 to achieve [above, 7, 5] and [top, 7]. Picking up box7 is required to move it to the current top (box5). Box7 is available (top and alone), and this step is essential for the final stacking action.
    How state changes:
        Preconditions satisfied: [top, 7] (box7 is top of Stack E), [on-table, 7, 1] (box7 on table), and [hand_free, 0] (robot hand free) all hold in the post-Step-4 state.
        Effects:
            [holding, 7, 0] is added (robot now holds box7).
            [hand_free, 0] is removed (hand no longer free).
            [top, 7] and [on-table, 7, 1] are removed (box7 no longer on table or top).
        New state:
            Stack E is empty (box7 held).
            Robot holds box7 ([holding, 7, 0]).
            Stack with box8 and box5 on top unchanged.
    Why alternatives not chosen:
        Pick-up of other boxes (e.g., [pick-up, 2, 1, 0] or [pick-up, 6, 1, 0]) is possible, but box2 and box6 are already in their goal positions (alone on table). Picking them up would be redundant and require extra steps to return them, violating the principle of minimal actions. No other box (e.g., box3 or box4) should be moved, as their stack is correct. Picking up box7 is the only action that directly enables the final goal step, making alternatives inefficient and goal-irrelevant.
Step 6: Execute [stack, 7, 5, 0]
    Why chosen: With box7 held and box5 being the top of the stack (from Step 4), stacking box7 on box5 completes the goal's [above, 7, 5] and [top, 7]. This finalizes the box8-box5-box7 stack, achieving all goal conditions.
    How state changes:
        Preconditions satisfied: [holding, 7, 0] (robot holds box7) and [top, 5] (box5 is top of the box8-box5 stack) both hold in the post-Step-5 state.
        Effects:
            [holding, 7, 0] is removed (box7 placed).
            [hand_free, 0] is added (hand now free).
            [top, 5] is removed (box5 no longer top).
            [above, 7, 5] and [top, 7] are added (box7 now on top of box5).
        Final state:
            Stack A: box2 alone ([top, 2], [on-table, 2, 1])
            Stack B: box4 with box3 on top ([top, 3], [above, 3, 4], [on-table, 4, 1])
            Stack D: box6 alone ([top, 6], [on-table, 6, 1])
            New stack: box8 (bottom), box5 (middle), box7 (top) ([on-table, 8, 1], [above, 5, 8], [above, 7, 5], [top, 7])
            Robot: [hand_free, 0]
            Matches goal state exactly.
    Why alternatives not chosen:
        With box7 held, stacking on other tops (e.g., [stack, 7, 2, 0] or [stack, 7, 4, 0]) is possible, but stacking on box2 or box4 would violate the goal (e.g., creating [above, 7, 2] which is not required). Only stacking on box5 achieves the critical [above, 7, 5] relation. Since all other stacks are correct, any alternative would introduce errors requiring correction, so this is the only goal-satisfying action.
Overall Logical Flow
    The plan follows a logical sequence: first, disassemble the incorrect stack (Step 1) to free box3; then, build the correct box4-box3 stack (Steps 2) to isolate a goal-compliant sub-configuration. Next, focus on the box8-box5-box7 stack by moving box5 to box8 (Steps 3-4) before adding box7 (Steps 5-6). Each action satisfies preconditions from the evolving state, and the state transitions directly enable the next step (e.g., unstacking box3 makes stacking on box4 possible; stacking box5 on box8 creates the top needed for box7). The robot's hand alternates between free and holding states as required, ensuring no deadlocks. Alternatives were consistently rejected because they either violated the goal, created unnecessary complexity, or ignored the bottom-up stacking requirement for the multi-block stack. The plan efficiently achieves the goal in minimal steps by prioritizing critical path actions (e.g., resolving the only multi-block stack first) and leveraging the hand-free state for key transitions.
</REASON>

## Task to explain:
Static predicates: {static_state}
Initial State: {initial_state}
Goal State: {goal_state}
Correct plan of actions: {actions}

After your reasoning, put your final explanation in this format:
<REASON><ANSWER_HERE></REASON>
