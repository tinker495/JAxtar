import jax.numpy as jnp
import jax

from heuristic.heuristic_base import Heuristic
from puzzle import TSP


class TSPHeuristic(Heuristic):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def distance(self, solve_config: TSP.SolveConfig, current: TSP.State) -> float:
        """여행자 문제(TSP)용 MST + 2-edge 휴리스틱을 계산한다.

        단계별 설명 (한국어):
        1. `visited` 마스크를 풀어서 아직 방문하지 않은 도시 집합 **R** 을 구한다.
        2. R 가 비어 있으면 남은 비용은 *현 위치 → 시작점* 거리뿐이다.
        3. R 가 존재하면 다음 세 항을 더한다.
           a. 현 위치에서 R 로 가는 최단 거리 (출발 간선)
           b. R 로 유도된 최소 신장 트리(MST) 가중치 (집합 연결 비용)
           c. 시작점에서 R 로 가는 최단 거리 (귀환 전 마지막 간선)

        이 값은 실제 잔여 투어 비용의 하한이며, A* 탐색에서 허용적이다.
        """
        # Unpack visited mask and point index.
        unpacked = current.unpacking()
        visited = unpacked.mask.astype(jnp.bool_)
        current_idx = unpacked.point

        distance_matrix = solve_config.distance_matrix
        start_idx = solve_config.start

        # If every city has been visited, heuristic is simply the cost of
        # returning to the start city (or 0 if we are already at the start).
        all_visited = jnp.all(visited)
        return_to_start = distance_matrix[current_idx, start_idx]
        if_distance = jnp.where(all_visited, return_to_start, 0.0)

        # Helper to compute MST weight of the remaining (unvisited) nodes.
        def _mst_weight(dist_matrix: jnp.ndarray, remaining_mask: jnp.ndarray) -> float:
            """Compute the MST weight of the sub-graph induced by `remaining_mask`.

            The implementation uses a fixed-size Prim's algorithm so that it is
            compatible with JAX JIT/grad.  Complexity is O(N²).
            """

            n = dist_matrix.shape[0]
            num_remaining = jnp.sum(remaining_mask)

            # Quick exit when there is 0 or 1 vertex left.
            def _return_zero(_):
                return jnp.float32(0.0)

            def _compute_mst(_):
                first_idx = jnp.argmin(jnp.where(remaining_mask, 0, n + 1))
                visited = jnp.zeros(n, dtype=jnp.bool_).at[first_idx].set(True)
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

                (visited_final, total_weight), _ = jax.lax.scan(
                    body, (visited, weight0), xs=jnp.arange(n)
                )
                return total_weight

            return jax.lax.cond(num_remaining <= 1, _return_zero, _compute_mst, None)

        # Compute heuristic only if not all visited.
        def not_solved():
            remaining_mask = ~visited

            # 1) cheapest edge from current city to any remaining city
            d_current_to_remaining = jnp.where(
                remaining_mask, distance_matrix[current_idx], jnp.inf
            )
            h1 = jnp.min(d_current_to_remaining)

            # 2) cheapest edge from start city to any remaining city
            d_start_to_remaining = jnp.where(remaining_mask, distance_matrix[start_idx], jnp.inf)
            h2 = jnp.min(d_start_to_remaining)

            # 3) MST of remaining cities
            h_mst = _mst_weight(distance_matrix, remaining_mask)

            return h1 + h2 + h_mst

        heuristic = jax.lax.cond(all_visited, lambda _: if_distance, lambda _: not_solved(), None)
        return heuristic
