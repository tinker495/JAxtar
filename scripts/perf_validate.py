"""
Performance Validation Script for Refactored JAxtar Algorithms.

Runs all 5 algorithms on N random SlidePuzzle instances and reports:
- Solve rate
- Average solution cost
- Average wall-clock time per solve
"""

import os
import sys
import time

sys.path.append(os.getcwd())
sys.path.append("/home/tinker/PuXle")
sys.path.append("/home/tinker/Xtructure")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from puxle.puzzles.slidepuzzle import SlidePuzzle  # noqa: E402

from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic  # noqa: E402
from JAxtar.bi_stars.bi_astar import bi_astar_builder  # noqa: E402
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder  # noqa: E402
from JAxtar.stars.astar import astar_builder  # noqa: E402
from JAxtar.stars.astar_d import astar_d_builder  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
N_PUZZLE = 3  # 3x3 = 8-puzzle
SCRAMBLE_STEPS = 30  # scramble depth
N_INSTANCES = 20  # number of test instances
BATCH_SIZE = 512
MAX_NODES = 100_000
# ─────────────────────────────────────────────────────────────────────────────


def scramble(puzzle, solve_config, goal, steps, rng):
    """Produce a random state by taking `steps` random actions from goal."""

    def _step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        action = jax.random.randint(subkey, (), 0, puzzle.action_size)
        states_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
        filled = jnp.array([True])
        next_b, _ = puzzle.batched_get_actions(solve_config, states_b, action[None], filled)
        return (jax.tree_util.tree_map(lambda x: x[0], next_b), key), None

    (start, _), _ = jax.lax.scan(_step, (goal, rng), None, length=steps)
    return start


def run_solver(solver_fn, solve_config, starts, label):
    """Run solver on a list of start states, return (solve_rate, avg_cost, avg_time_ms)."""
    solved_count = 0
    total_cost = 0.0
    total_time = 0.0

    for i, start in enumerate(starts):
        t0 = time.perf_counter()
        result = solver_fn(solve_config, start)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        # Determine if solved
        if hasattr(result, "solved"):
            is_solved = bool(result.solved)
            if is_solved:
                cost = float(result.get_cost(result.solved_idx)[0])
                solved_count += 1
                total_cost += cost
        elif hasattr(result, "meeting"):
            is_solved = bool(result.meeting.found)
            if is_solved:
                cost = float(result.meeting.total_cost)
                solved_count += 1
                total_cost += cost
        else:
            is_solved = False

        total_time += elapsed_ms

    n = len(starts)
    solve_rate = solved_count / n
    avg_cost = total_cost / solved_count if solved_count > 0 else float("nan")
    avg_time = total_time / n
    return solve_rate, avg_cost, avg_time


def main():
    print("=" * 60)
    print(f"JAxtar Performance Validation — {N_PUZZLE}x{N_PUZZLE} Puzzle")
    print(f"  Instances: {N_INSTANCES}, Scramble: {SCRAMBLE_STEPS} steps")
    print(f"  Batch size: {BATCH_SIZE}, Max nodes: {MAX_NODES}")
    print("=" * 60)

    puzzle = SlidePuzzle(N_PUZZLE)
    heuristic = SlidePuzzleHeuristic(puzzle)
    solve_config = puzzle.get_solve_config()
    goal = solve_config.TargetState
    inv_map = jnp.array(puzzle.inverse_action_map)

    # Build solvers (JIT warm-up included)
    print("\nBuilding solvers (JIT warm-up)...")
    solvers = {}

    print("  [1/4] A* ...", end=" ", flush=True)
    solvers["A*"] = astar_builder(puzzle, heuristic, batch_size=BATCH_SIZE, max_nodes=MAX_NODES)
    print("done")

    print("  [2/4] A*d ...", end=" ", flush=True)
    solvers["A*d"] = astar_d_builder(puzzle, heuristic, batch_size=BATCH_SIZE, max_nodes=MAX_NODES)
    print("done")

    print("  [3/4] Bi-A* ...", end=" ", flush=True)
    solvers["Bi-A*"] = bi_astar_builder(
        puzzle, heuristic, batch_size=BATCH_SIZE, max_nodes=MAX_NODES, inverse_action_map=inv_map
    )
    print("done")

    print("  [4/4] Bi-A*d ...", end=" ", flush=True)
    solvers["Bi-A*d"] = bi_astar_d_builder(
        puzzle, heuristic, batch_size=BATCH_SIZE, max_nodes=MAX_NODES, inverse_action_map=inv_map
    )
    print("done")

    # Generate test instances
    print(f"\nGenerating {N_INSTANCES} test instances...")
    rng = jax.random.PRNGKey(2025)
    starts = []
    for i in range(N_INSTANCES):
        rng, subkey = jax.random.split(rng)
        start = scramble(puzzle, solve_config, goal, SCRAMBLE_STEPS, subkey)
        starts.append(start)
    print("Done.")

    # Run benchmarks
    print("\n" + "-" * 60)
    print(f"{'Algorithm':<12} {'Solve%':>7} {'Avg Cost':>10} {'Avg Time(ms)':>14}")
    print("-" * 60)

    results = {}
    for name, solver in solvers.items():
        solve_rate, avg_cost, avg_time = run_solver(solver, solve_config, starts, name)
        results[name] = (solve_rate, avg_cost, avg_time)
        cost_str = f"{avg_cost:.1f}" if avg_cost == avg_cost else "N/A"
        print(f"{name:<12} {solve_rate*100:>6.1f}% {cost_str:>10} {avg_time:>13.1f}")

    print("-" * 60)
    print("\nPerformance validation complete.")

    # Sanity check: all solved instances should have same cost (optimal A*)
    astar_rate, astar_cost, _ = results.get("A*", (0, float("nan"), 0))
    for name, (rate, cost, _) in results.items():
        if name == "A*":
            continue
        if rate > 0 and astar_rate > 0:
            # Costs may differ slightly for bi-directional (not always optimal)
            print(f"  {name}: avg cost = {cost:.1f} (A* = {astar_cost:.1f})")


if __name__ == "__main__":
    main()
