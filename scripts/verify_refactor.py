import os
import sys

# Ensure current directory and dependencies are in path
sys.path.append(os.getcwd())
sys.path.append("/home/tinker/PuXle")
sys.path.append("/home/tinker/Xtructure")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from puxle.puzzles.slidepuzzle import SlidePuzzle  # noqa: E402

# Import heuristic from top-level package
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic  # noqa: E402
from JAxtar.bi_stars.bi_astar import bi_astar_builder  # noqa: E402
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder  # noqa: E402
from JAxtar.stars.astar import astar_builder  # noqa: E402
from JAxtar.stars.astar_d import astar_d_builder  # noqa: E402
from JAxtar.stars.qstar import qstar_builder  # noqa: E402
from qfunction.q_base import QFunction  # noqa: E402


class HeuristicQFunction(QFunction):
    def __init__(self, heuristic):
        super().__init__(heuristic.puzzle)
        self.heuristic = heuristic

    def q_value(self, q_parameters, current):
        # Q(s, a) = cost(s, a) + h(next_s)
        # Using batched logic?
        # QFunction.batched_q_value calls this via vmap (params, current -> [action])
        # So current is single state here.
        pass  # We implement batched_q_value directly for efficiency

    def batched_q_value(self, q_parameters, current):
        # current: [batch] of states
        # Returns: [batch, action]
        solve_config = q_parameters

        # Get neighbors
        # filled assumed true for call?
        batch_size = jax.tree_util.tree_leaves(current)[0].shape[0]
        filled = jnp.ones((batch_size,), dtype=bool)

        neighbours, ncost = self.puzzle.batched_get_neighbours(solve_config, current, filled)
        # neighbours: [action, batch]
        # ncost: [action, batch]

        # Calculate heuristic for neighbors
        # Heuristic.batched_distance takes (config, states, filled)
        # We need to flatten neighbors
        action_size = self.puzzle.action_size
        flat_neighbours = jax.tree_util.tree_map(
            lambda x: x.flatten(), neighbours
        )  # [action*batch] (if scalar) or just flatten leading axes
        # Wait, flatten usually flattens to 1D. State fields might be multi-dim?
        # heuristic expects [batch] structure.
        # We reshape to [action*batch]
        flat_neighbours = jax.tree_util.tree_map(
            lambda x: x.reshape((action_size * batch_size,) + x.shape[2:]), neighbours
        )

        heuristic_params = self.heuristic.prepare_heuristic_parameters(solve_config)
        h_vals = self.heuristic.batched_distance(heuristic_params, flat_neighbours)

        h_vals = h_vals.reshape(action_size, batch_size)

        # Q = cost + h
        q_vals = ncost + h_vals  # [action, batch]

        return q_vals.transpose()  # [batch, action] - standard return for batched_q_value


def main():
    print("Verifying JAxtar Refactoring...")

    # 1. Setup Puzzle (Sliding Tile 3x3 / 8-puzzle)
    N = 3
    puzzle = SlidePuzzle(N)

    # 2. Setup Heuristic (Manhattan)
    heuristic = SlidePuzzleHeuristic(puzzle)

    # 3. Setup Q-Function (Dummy Q = Heuristic)
    q_fn = HeuristicQFunction(heuristic)

    # 4. Generate Problem
    solve_config = puzzle.get_solve_config()
    goal = solve_config.TargetState

    # Scramble from goal
    rng = jax.random.PRNGKey(42)
    scramble_key, solve_key = jax.random.split(rng)

    # Generate random start
    def _step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        actions = jax.random.randint(subkey, (1,), 0, puzzle.action_size)
        # Assuming batched step
        states_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
        filleds = jnp.array([True])
        next_states_b, _ = puzzle.batched_get_actions(solve_config, states_b, actions, filleds)
        next_state = jax.tree_util.tree_map(lambda x: x[0], next_states_b)
        return (next_state, key), None

    start, _ = jax.lax.scan(_step, (goal, scramble_key), None, length=20)[0]

    print("Puzzle initialized. Running solvers...")

    # 5. Build Solvers
    # Use small batch size for verification script speed/memory
    B = 256
    astar = astar_builder(puzzle, heuristic, batch_size=B, show_compile_time=True)
    astar_d = astar_d_builder(puzzle, heuristic, batch_size=B, show_compile_time=True)
    bi_astar = bi_astar_builder(
        puzzle,
        heuristic,
        batch_size=B,
        show_compile_time=True,
        inverse_action_map=jnp.array(puzzle.inverse_action_map),
    )
    bi_astar_d = bi_astar_d_builder(
        puzzle,
        heuristic,
        batch_size=B,
        show_compile_time=True,
        inverse_action_map=jnp.array(puzzle.inverse_action_map),
    )

    # Q* often needs special care, setup basic one
    qstar = qstar_builder(
        puzzle, q_fn, batch_size=B, look_ahead_pruning=True, show_compile_time=True
    )

    # 6. Run Solvers
    print("\n--- Running A* ---")
    res_astar = astar(solve_config, start)
    print(f"Solved: {res_astar.solved}")
    cost = -1
    if res_astar.solved:
        cost = res_astar.get_cost(res_astar.solved_idx)
        print(f"Cost: {cost}")
    else:
        print("Failed to solve.")

    print("\n--- Running A* Deferred ---")
    res_astar_d = astar_d(solve_config, start)
    print(f"Solved: {res_astar_d.solved}")
    if res_astar_d.solved:
        cost_d = res_astar_d.get_cost(res_astar_d.solved_idx)
        print(f"Cost: {cost_d}")
        if res_astar.solved:
            assert cost == cost_d, f"Cost mismatch: A*={cost}, A*d={cost_d}"

    print("\n--- Running Bi-A* ---")
    res_bi = bi_astar(solve_config, start)
    print(f"Meeting Found: {res_bi.meeting.found}")
    print(f"Meeting Cost: {res_bi.meeting.total_cost}")
    if res_bi.meeting.found:
        if res_astar.solved:
            assert (
                cost == res_bi.meeting.total_cost
            ), f"Cost mismatch: A*={cost}, Bi-A*={res_bi.meeting.total_cost}"

    print("\n--- Running Bi-A* Deferred ---")
    res_bi_d = bi_astar_d(solve_config, start)
    print(f"Meeting Found: {res_bi_d.meeting.found}")
    print(f"Meeting Cost: {res_bi_d.meeting.total_cost}")
    if res_bi_d.meeting.found:
        if res_astar.solved:
            assert (
                cost == res_bi_d.meeting.total_cost
            ), f"Cost mismatch: A*={cost}, Bi-A*d={res_bi_d.meeting.total_cost}"

    print("\n--- Running Q* ---")
    res_q = qstar(solve_config, start)
    print(f"Solved: {res_q.solved}")
    if res_q.solved:
        cost_q = res_q.get_cost(res_q.solved_idx)
        print(f"Cost: {cost_q}")
        if res_astar.solved:
            assert cost == cost_q, f"Cost mismatch: A*={cost}, Q*={cost_q}"

    print("\nVerification Successful!")


if __name__ == "__main__":
    main()
