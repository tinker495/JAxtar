import logging
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import xtructure.numpy as xnp

from helpers.path_steps import PathStep, build_path_steps_from_actions
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction

logger = logging.getLogger(__name__)


def extract_heuristic_accuracy_data(
    puzzle,
    solve_config,
    initial_state,
    action_sequence: Optional[List[Union[int, str]]] = None,
    path_states: Optional[List[Any]] = None,
    heuristic_model: Optional[Heuristic] = None,
    qfunction_model: Optional[QFunction] = None,
) -> Optional[Dict[str, Any]]:
    path_steps = None
    actions_int = None

    try:
        heuristic_params = (
            heuristic_model.prepare_heuristic_parameters(solve_config)
            if heuristic_model is not None
            else None
        )
        qfunction_params = (
            qfunction_model.prepare_q_parameters(solve_config)
            if qfunction_model is not None
            else None
        )

        def _evaluate_model(state):
            if heuristic_model is not None:
                params = heuristic_params if heuristic_params is not None else solve_config
                return float(heuristic_model.distance(params, state))
            if qfunction_model is not None:
                params = qfunction_params if qfunction_params is not None else solve_config
                q_vals = qfunction_model.q_value(params, state)
                q_vals_arr = jnp.asarray(q_vals)
                return float(jnp.min(q_vals_arr))
            return None

        def _batch_evaluate_model(states):
            if heuristic_model is not None:
                params = heuristic_params if heuristic_params is not None else solve_config
                return heuristic_model.batched_distance(params, states)
            if qfunction_model is not None:
                params = qfunction_params if qfunction_params is not None else solve_config
                return jnp.min(qfunction_model.batched_q_value(params, states), axis=-1)
            return None

        if action_sequence is not None:
            actions_int = []
            action_lookup = None

            needs_lookup = False
            for a in action_sequence:
                if isinstance(a, str) and not a.isdigit():
                    needs_lookup = True
                    break

            if needs_lookup:
                try:
                    action_size = getattr(puzzle, "action_size", None)
                    if action_size is not None:
                        action_lookup = {
                            str(puzzle.action_to_string(a)): a for a in range(action_size)
                        }
                except (AttributeError, ValueError, TypeError):
                    logger.debug("Could not build action lookup table for puzzle %r", puzzle)

            for a in action_sequence:
                if isinstance(a, str) and not a.isdigit():
                    if action_lookup and a in action_lookup:
                        actions_int.append(action_lookup[a])
                    else:
                        return None
                else:
                    actions_int.append(int(a))

            path_steps = build_path_steps_from_actions(
                puzzle=puzzle,
                solve_config=solve_config,
                initial_state=initial_state,
                actions=actions_int,
                heuristic=None,
                q_fn=None,
            )

            # Evaluate dist directly on each state to match search-time dist semantics (e.g., search_base_eager.py).
            if path_steps:
                states = [step.state for step in path_steps]
                states_batch = xnp.concatenate(states)
                dists = _batch_evaluate_model(states_batch)
                if dists is not None:
                    dists_arr = np.asarray(dists)
                    for step, dist_val in zip(path_steps, dists_arr):
                        step.dist = float(dist_val)
                else:
                    for step in path_steps:
                        step.dist = _evaluate_model(step.state)

        elif path_states is not None:
            # Recover per-step transition costs and actions via get_neighbours when only states are given.
            steps = []
            current_cost = 0.0
            valid_path = True

            state = path_states[0]
            steps.append(PathStep(state=state, cost=current_cost, dist=0.0, action=None))

            for i in range(len(path_states) - 1):
                curr_s = path_states[i]
                next_s = path_states[i + 1]

                step_cost = np.nan
                action_idx = None
                try:
                    neighbours, costs = puzzle.get_neighbours(solve_config, curr_s, True)

                    found = False
                    for idx in range(len(costs)):
                        n_s = neighbours[idx]

                        is_equal = False
                        try:
                            is_equal = np.array_equal(n_s, next_s)
                        except (ValueError, TypeError):
                            is_equal = n_s == next_s

                        if is_equal:
                            step_cost = float(costs[idx])
                            action_idx = int(idx)
                            found = True
                            break

                    if not found:
                        valid_path = False
                        break
                except (KeyError, ValueError, AttributeError, IndexError, RuntimeError):
                    valid_path = False
                    break

                steps[-1].action = action_idx

                steps[-1].dist = _evaluate_model(
                    curr_s,
                )

                current_cost += step_cost

                steps.append(PathStep(state=next_s, cost=current_cost, dist=0.0, action=None))

            if valid_path and steps:
                states = [s.state for s in steps]
                states_batch = xnp.concatenate(states)
                dists = _batch_evaluate_model(states_batch)
                if dists is not None:
                    dists_arr = np.asarray(dists)
                    for step, dist_val in zip(steps, dists_arr):
                        step.dist = float(dist_val)
                else:
                    steps[-1].dist = _evaluate_model(
                        steps[-1].state,
                    )
                path_steps = steps

                actions_int = [s.action for s in steps if s.action is not None]

        if not path_steps:
            return None

        path_cost = path_steps[-1].cost

        actual_dists = []
        estimated_dists = []
        states = []

        # Force dist of the last state (goal) to be 0.0 if it is solved,
        # to avoid outlier at Actual Cost 0 due to approximation error.
        if puzzle.is_solved(solve_config, path_steps[-1].state):
            path_steps[-1].dist = 0.0

        for step in path_steps:
            actual_dist = float(path_cost - step.cost)
            estimated_dist = float(step.dist) if step.dist is not None else np.inf

            if np.isfinite(estimated_dist):
                actual_dists.append(actual_dist)
                estimated_dists.append(estimated_dist)
            states.append(step.state)

        return {
            "actual": actual_dists,
            "estimated": estimated_dists,
            "states": xnp.concatenate(states),
            "actions": actions_int,
        }

    except (ValueError, RuntimeError, KeyError, AttributeError, TypeError):
        return None
