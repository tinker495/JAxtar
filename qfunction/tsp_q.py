from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from puxle import TSP

from qfunction.q_base import QFunction


class TSPQ(QFunction):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def q_value(
        self, solve_config: TSP.SolveConfig, current: TSP.State, params: Optional[Any] = None
    ) -> chex.Array:
        """
        Get q values for all possible actions from the current state.
        For EmptyQFunction, this returns an array of zeros for each available move.
        """
        neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(None, 0))(solve_config, neighbors)
        return dists

    def _distance(self, solve_config: TSP.SolveConfig, current: TSP.State) -> float:
        """Computes Q-value using the same MST + 2-edge heuristic calculation.

        For each neighbor state, estimates the minimum remaining tour cost. In A* search,
        adding this value to the edge cost provides the same effect as f = g + h.
        """
        # Adapted MST + two nearest edges heuristic (same as in TSPHeuristic).
        current = current.unpacked
        visited = current.mask.astype(jnp.bool_)
        current_idx = current.point

        distance_matrix = solve_config.distance_matrix
        start_idx = solve_config.start

        # If tour is complete, cost is to return home.
        all_visited = jnp.all(visited)
        return_to_start = distance_matrix[current_idx, start_idx]
        if_distance = jnp.where(all_visited, return_to_start, 0.0)

        def _mst_weight(dist_matrix: jnp.ndarray, remaining_mask: jnp.ndarray) -> float:
            n = dist_matrix.shape[0]
            num_remaining = jnp.sum(remaining_mask)

            def _return_zero(_):
                return jnp.float32(0.0)

            def _compute_mst(_):
                first_idx = jnp.argmin(jnp.where(remaining_mask, 0, n + 1))
                visited_mst = jnp.zeros(n, dtype=jnp.bool_).at[first_idx].set(True)
                weight0 = jnp.float32(0.0)

                def body(carry, _):
                    visited_inner, w_inner = carry
                    mask_cols = remaining_mask & (~visited_inner)

                    def _add_edge(carry_in):
                        v_in, w_in = carry_in
                        mask_rows = v_in
                        edge_mask = mask_rows[:, None] & mask_cols[None, :]
                        candidate = jnp.where(edge_mask, dist_matrix, jnp.inf)
                        min_costs = jnp.min(candidate, axis=0)
                        idx = jnp.argmin(min_costs)

                        w_in = w_in + min_costs[idx]
                        v_in = v_in.at[idx].set(True)
                        return v_in, w_in

                    visited_inner, w_inner = jax.lax.cond(
                        jnp.any(mask_cols), _add_edge, lambda c: c, (visited_inner, w_inner)
                    )

                    return (visited_inner, w_inner), None

                (v_final, total_weight), _ = jax.lax.scan(
                    body, (visited_mst, weight0), xs=jnp.arange(n)
                )
                return total_weight

            return jax.lax.cond(num_remaining <= 1, _return_zero, _compute_mst, None)

        def not_solved():
            remaining_mask = ~visited

            d_current_to_remaining = jnp.where(
                remaining_mask, distance_matrix[current_idx], jnp.inf
            )
            h1 = jnp.min(d_current_to_remaining)

            d_start_to_remaining = jnp.where(remaining_mask, distance_matrix[start_idx], jnp.inf)
            h2 = jnp.min(d_start_to_remaining)

            h_mst = _mst_weight(distance_matrix, remaining_mask)

            return h1 + h2 + h_mst

        return jax.lax.cond(all_visited, lambda: if_distance, not_solved)
