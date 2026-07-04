from __future__ import annotations

from dataclasses import dataclass
from typing import List

import jax.numpy as jnp
import numpy as np

from JAxtar.annotate import ACTION_DTYPE
from JAxtar.solution_trace import SolutionTrace

ACTION_PAD_INT = int(np.iinfo(np.dtype(ACTION_DTYPE)).max)


@dataclass
class PathStep:
    state: object
    cost: float
    dist: float | None
    action: int | None


def build_path_steps_from_trace(
    puzzle,
    solve_config,
    initial_state,
    solution_trace: SolutionTrace,
    heuristic=None,
    q_fn=None,
) -> List[PathStep]:
    """Convert search-owned solution trace into display-oriented path steps."""
    if not solution_trace.solved:
        return []

    return build_path_steps_from_actions(
        puzzle=puzzle,
        solve_config=solve_config,
        initial_state=initial_state,
        actions=list(solution_trace.actions),
        heuristic=heuristic,
        q_fn=q_fn,
        states=list(solution_trace.states) if solution_trace.states is not None else None,
        costs=list(solution_trace.costs) if solution_trace.costs is not None else None,
        dists=list(solution_trace.dists) if solution_trace.dists is not None else None,
    )


def build_path_steps_from_actions(
    puzzle,
    solve_config,
    initial_state,
    actions: List[int],
    heuristic=None,
    q_fn=None,
    states=None,
    costs=None,
    dists=None,
) -> List[PathStep]:
    steps: List[PathStep] = []
    heuristic_params = (
        heuristic.prepare_heuristic_parameters(solve_config) if heuristic is not None else None
    )
    q_fn_params = q_fn.prepare_q_parameters(solve_config) if q_fn is not None else None

    action_sequence: List[int] = []
    for raw_action in actions:
        action_val = int(raw_action)
        if action_val == ACTION_PAD_INT:
            break
        action_sequence.append(action_val)

    states_list = list(states) if states is not None else None
    costs_arr = np.asarray(costs) if costs is not None else None
    can_use_trace = (
        states_list is not None
        and costs_arr is not None
        and len(costs_arr) >= (len(action_sequence) + 1)
    )
    state = states_list[0] if states_list else initial_state
    running_cost = 0.0

    def _trace_cost(idx: int, default_val: float) -> float:
        if costs is not None and idx < len(costs):
            return float(costs[idx])
        return default_val

    def _trace_dist(idx: int, current_state) -> float | None:
        if dists is not None and idx < len(dists):
            val = dists[idx]
            if val is None:
                return None
            if isinstance(val, jnp.ndarray):
                val = float(val)
            return float(val)
        if heuristic is not None:
            params = heuristic_params if heuristic_params is not None else solve_config
            return float(heuristic.distance(params, current_state))
        if q_fn is not None:
            params = q_fn_params if q_fn_params is not None else solve_config
            q_vals = q_fn.q_value(params, current_state)
            return float(jnp.min(jnp.asarray(q_vals)))
        return None

    start_action = action_sequence[0] if action_sequence else None
    start_cost = _trace_cost(0, running_cost)
    start_dist = _trace_dist(0, state)
    steps.append(PathStep(state=state, cost=start_cost, dist=start_dist, action=start_action))

    for depth, action_val in enumerate(action_sequence):
        if can_use_trace and depth + 1 < len(states_list):
            next_state = states_list[depth + 1]
            next_cost = float(costs_arr[depth + 1])
            running_cost = next_cost
        else:
            neighbours, transition_cost = puzzle.get_neighbours(solve_config, state, True)
            step_cost = float(transition_cost[action_val])
            if not np.isfinite(step_cost):
                break
            running_cost += step_cost
            next_state = neighbours[action_val]
            if states_list is not None and depth + 1 < len(states_list):
                next_state = states_list[depth + 1]

        cost_val = _trace_cost(depth + 1, running_cost)
        dist_val = _trace_dist(depth + 1, next_state)

        next_action: int | None = None
        if depth + 1 < len(action_sequence):
            next_action = action_sequence[depth + 1]

        steps.append(
            PathStep(
                state=next_state,
                cost=cost_val,
                dist=dist_val,
                action=next_action,
            )
        )
        state = next_state

    return steps
