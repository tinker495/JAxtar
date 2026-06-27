import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from qfunction import QFromHeuristic


class _ToyPuzzle:
    def get_neighbours(self, solve_config, current):
        costs = jnp.array([solve_config["base_cost"], solve_config["base_cost"] + 1.0])
        return jnp.array([current + 1.0, current + 2.0]), costs


class _ToyHeuristic(Heuristic):
    def prepare_heuristic_parameters(self, solve_config, **kwargs):
        return {"scale": solve_config["scale"] + kwargs.get("scale_delta", 0.0)}

    def distance(self, heuristic_parameters, current):
        return current * heuristic_parameters["scale"]


def test_q_from_heuristic_adds_neighbor_distance_to_move_cost():
    q_fn = QFromHeuristic(_ToyHeuristic(_ToyPuzzle()))
    solve_config = {"scale": jnp.array(3.0), "base_cost": jnp.array(0.5)}

    q_parameters = q_fn.prepare_q_parameters(solve_config, scale_delta=1.0)
    q_values = q_fn.q_value(q_parameters, jnp.array(2.0))

    assert jnp.allclose(q_values, jnp.array([12.5, 17.5]))


def test_q_from_heuristic_still_accepts_raw_solve_config():
    q_fn = QFromHeuristic(_ToyHeuristic(_ToyPuzzle()))
    solve_config = {"scale": jnp.array(3.0), "base_cost": jnp.array(0.5)}

    q_values = q_fn.q_value(solve_config, jnp.array(2.0))

    assert jnp.allclose(q_values, jnp.array([9.5, 13.5]))


def test_q_from_heuristic_batches_over_current_states():
    q_fn = QFromHeuristic(_ToyHeuristic(_ToyPuzzle()))
    solve_config = {"scale": jnp.array(3.0), "base_cost": jnp.array(0.5)}

    q_parameters = q_fn.prepare_q_parameters(solve_config, scale_delta=1.0)
    q_values = q_fn.batched_q_value(q_parameters, jnp.array([2.0, 5.0]))

    assert jnp.allclose(q_values, jnp.array([[12.5, 17.5], [24.5, 29.5]]))
