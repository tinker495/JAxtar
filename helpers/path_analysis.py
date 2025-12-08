from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import xtructure.numpy as xnp

from helpers.visualization import PathStep, build_path_steps_from_actions
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction


def extract_heuristic_accuracy_data(
    puzzle,
    solve_config,
    initial_state,
    action_sequence: Optional[List[Union[int, str]]] = None,
    path_states: Optional[List[Any]] = None,
    heuristic_model: Optional[Heuristic] = None,
    qfunction_model: Optional[QFunction] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extracts heuristic accuracy data (actual vs estimated distances) from an action sequence or state sequence.

    Args:
        puzzle: The puzzle instance.
        solve_config: The solve configuration for the puzzle.
        initial_state: The initial state of the puzzle.
        action_sequence: A list of actions (integers or strings that can be converted to ints).
        path_states: A list of states representing the path (optional, used if action_sequence is None).
        heuristic_model: Optional heuristic model to evaluate.
        qfunction_model: Optional Q-function model to evaluate.

    Returns:
        A dictionary containing 'actual' distances, 'estimated' distances, 'states', and 'actions',
        or None if extraction fails or the sequence is invalid.
    """
    path_steps = None
    actions_int = None

    try:
        if action_sequence is not None:
            actions_int = []
            action_lookup = None

            # Check if we need string-to-int conversion
            needs_lookup = False
            for a in action_sequence:
                if isinstance(a, str) and not a.isdigit():
                    needs_lookup = True
                    break

            if needs_lookup:
                try:
                    # Attempt to build lookup table from puzzle
                    action_size = getattr(puzzle, "action_size", None)
                    if action_size is not None:
                        action_lookup = {
                            str(puzzle.action_to_string(a)): a for a in range(action_size)
                        }
                except Exception:
                    pass

            for a in action_sequence:
                if isinstance(a, str) and not a.isdigit():
                    if action_lookup and a in action_lookup:
                        actions_int.append(action_lookup[a])
                    else:
                        # Cannot convert this string action, failing
                        return None
                else:
                    actions_int.append(int(a))

            path_steps = build_path_steps_from_actions(
                puzzle=puzzle,
                solve_config=solve_config,
                initial_state=initial_state,
                actions=actions_int,
                heuristic=heuristic_model,
                q_fn=qfunction_model,
            )

            # Re-evaluate dist for each step using the model directly on the state.
            # This matches how search algorithms (like search_base.py) update 'dist'
            # by evaluating the state when it is generated.
            for step in path_steps:
                step.dist = _evaluate_model(
                    step.state,
                    solve_config,
                    heuristic_model,
                    qfunction_model,
                )

        elif path_states is not None:
            # Construct path steps from states
            # Note: This requires calculating transition costs between states to get actual costs.
            steps = []
            current_cost = 0.0
            valid_path = True

            # 1. Add initial state placeholder
            state = path_states[0]
            steps.append(PathStep(state=state, cost=current_cost, dist=0.0, action=None))

            # 2. Process transitions
            for i in range(len(path_states) - 1):
                curr_s = path_states[i]
                next_s = path_states[i + 1]

                # Find transition cost and action from curr_s to next_s
                step_cost = np.nan
                action_idx = None
                try:
                    neighbours, costs = puzzle.get_neighbours(solve_config, curr_s, True)

                    # Find matching neighbor
                    found = False
                    for idx in range(len(costs)):
                        # Flexible equality check for JAX/NumPy arrays or objects
                        n_s = neighbours[idx]

                        is_equal = False
                        try:
                            is_equal = np.array_equal(n_s, next_s)
                        except Exception:
                            is_equal = n_s == next_s

                        if is_equal:
                            step_cost = float(costs[idx])
                            action_idx = int(idx)
                            found = True
                            break

                    if not found:
                        valid_path = False
                        break
                except Exception:
                    valid_path = False
                    break

                # Update previous step with action
                steps[-1].action = action_idx

                # Evaluate current state's dist (heuristic/Q-value)
                steps[-1].dist = _evaluate_model(
                    curr_s, solve_config, heuristic_model, qfunction_model
                )

                current_cost += step_cost

                # Add next state placeholder
                steps.append(PathStep(state=next_s, cost=current_cost, dist=0.0, action=None))

            # Update the final state's distance
            if valid_path and steps:
                steps[-1].dist = _evaluate_model(
                    steps[-1].state, solve_config, heuristic_model, qfunction_model
                )
                path_steps = steps

                # Extract actions for return dict
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

    except Exception:
        return None


def _evaluate_model(
    state,
    solve_config,
    heuristic,
    q_fn,
) -> Optional[float]:
    """Evaluates the model (Heuristic or Q-Function) on a given state."""
    if heuristic is not None:
        return float(heuristic.distance(solve_config, state))
    if q_fn is not None:
        # For Q-function, the 'heuristic' value of a state is typically min Q(s, a)
        # representing the estimated cost-to-go from state s.
        q_vals = q_fn.q_value(solve_config, state)
        q_vals_arr = jnp.asarray(q_vals)
        return float(jnp.min(q_vals_arr))
    return None
