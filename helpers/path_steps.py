from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import xtructure.numpy as xnp

from JAxtar.annotate import ACTION_DTYPE
from JAxtar.solution_trace import SolutionTrace

ACTION_PAD_INT = int(np.iinfo(np.dtype(ACTION_DTYPE)).max)

# Bucketed batch size for one-shot path dist evaluation; padding to the bucket
# keeps the jit cache at one entry across varying path lengths.
_PATH_DIST_BUCKET = 32


@lru_cache(maxsize=8)
def _jitted_batched_distance(heuristic):
    return jax.jit(heuristic.batched_distance)


@lru_cache(maxsize=8)
def _jitted_batched_q_value(q_fn):
    return jax.jit(q_fn.batched_q_value)


def _batched_path_dists(heuristic, heuristic_params, states: List) -> List[float]:
    """Evaluate h for a whole path in one jitted batched call.

    Per-state `heuristic.distance` calls re-trace the model (and bitpacked
    state unpacking) eagerly at every step; one compiled call replaces them.
    """
    n = len(states)
    pad = -n % _PATH_DIST_BUCKET
    batch = xnp.concatenate(list(states) + [states[-1]] * pad)
    dists = _jitted_batched_distance(heuristic)(heuristic_params, batch)
    return [float(v) for v in jax.device_get(dists)[:n]]


def _batched_path_q_dists(q_fn, q_params, states: List) -> List[float]:
    """Evaluate min-a Q(s, a) for a whole path in one jitted batched call."""
    n = len(states)
    pad = -n % _PATH_DIST_BUCKET
    batch = xnp.concatenate(list(states) + [states[-1]] * pad)
    q_vals = _jitted_batched_q_value(q_fn)(q_params, batch)
    mins = jnp.min(jnp.asarray(q_vals), axis=-1)
    return [float(v) for v in jax.device_get(mins)[:n]]


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


def _materialise_path(
    puzzle,
    solve_config,
    initial_state,
    action_sequence: List[int],
    states_list,
    costs_list,
) -> tuple[list, List[float], List[int]]:
    """Resolve the state and cost at every path index.

    Trace-supplied states / costs win wherever they reach; the remaining indices are
    replayed one action at a time through the puzzle. Replay stops at an unreachable
    transition, so the action sequence is truncated to the states actually produced.
    """

    def _traced(seq, idx):
        return seq[idx] if seq is not None and idx < len(seq) else None

    state = _traced(states_list, 0)
    if state is None:
        state = initial_state
    running_cost = _traced(costs_list, 0)
    if running_cost is None:
        running_cost = 0.0

    path_states = [state]
    path_costs = [running_cost]

    for depth, action_val in enumerate(action_sequence):
        next_state = _traced(states_list, depth + 1)
        traced_cost = _traced(costs_list, depth + 1)

        if next_state is None or traced_cost is None:
            neighbours, transition_cost = puzzle.get_neighbours(solve_config, state, True)
            step_cost = float(transition_cost[action_val])
            if not np.isfinite(step_cost):
                return path_states, path_costs, action_sequence[:depth]
            running_cost += step_cost
            if next_state is None:
                next_state = neighbours[action_val]
        if traced_cost is not None:
            running_cost = traced_cost

        path_states.append(next_state)
        path_costs.append(running_cost)
        state = next_state

    return path_states, path_costs, action_sequence


def _path_dists(
    dists,
    states: List,
    heuristic,
    heuristic_params,
    q_fn,
    q_fn_params,
) -> List[float | None] | None:
    """Resolve h (or min-a Q) for every path state in one batched, jitted call.

    Per-state `heuristic.distance` / `q_fn.q_value` are eager model applies -- ~62 ms
    each on the 14.5M-param rubikscube net -- so a search family whose trace carries no
    dists (ID-*, bi-*) paid ~1.4 s of pure dispatch per solved sample, an order of
    magnitude more than the search it was reporting.
    """
    if dists is not None and len(dists) >= len(states):
        return [None if v is None else float(v) for v in list(dists)[: len(states)]]
    if heuristic is not None:
        return _batched_path_dists(heuristic, heuristic_params, states)
    if q_fn is not None:
        return _batched_path_q_dists(q_fn, q_fn_params, states)
    return None


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

    path_states, path_costs, action_sequence = _materialise_path(
        puzzle=puzzle,
        solve_config=solve_config,
        initial_state=initial_state,
        action_sequence=action_sequence,
        states_list=list(states) if states is not None else None,
        costs_list=[float(c) for c in costs] if costs is not None else None,
    )

    path_dists = _path_dists(
        dists=dists,
        states=path_states,
        heuristic=heuristic,
        heuristic_params=heuristic_params,
        q_fn=q_fn,
        q_fn_params=q_fn_params,
    )

    return [
        PathStep(
            state=path_states[idx],
            cost=path_costs[idx],
            dist=path_dists[idx] if path_dists is not None else None,
            action=action_sequence[idx] if idx < len(action_sequence) else None,
        )
        for idx in range(len(path_states))
    ]
