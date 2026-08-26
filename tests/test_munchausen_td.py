import jax
import jax.numpy as jnp
import pytest
from click.testing import CliRunner

import heuristic.neuralheuristic.target_dataset_builder as heuristic_builder
import qfunction.neuralq.target_dataset_builder as qfunction_builder
from cli.train_commands.dist_train_command import (
    heuristic_train_command,
    qfunction_train_command,
)
from config.pydantic_models import DistTrainOptions
from heuristic.neuralheuristic.target_dataset_builder import get_heuristic_dataset_builder
from qfunction.neuralq.target_dataset_builder import (
    _munchausen_cost_target,
    get_qfunction_dataset_builder,
)
from train_util.util import (
    MUNCHAUSEN_ALPHA,
    MUNCHAUSEN_CLIP_MIN,
    MUNCHAUSEN_LOG_CLIP_MIN,
    MUNCHAUSEN_LOG_DISCOUNT,
    MUNCHAUSEN_LOG_DISTANCE_SCALE,
    MUNCHAUSEN_LOG_TAU,
    MUNCHAUSEN_TAU,
    scaled_log_softmin_policy,
)


class _TwoActionPuzzle:
    def batched_is_solved(self, solve_configs, states, multi_solve_config=True):
        return jnp.zeros(solve_configs.shape[0], dtype=bool)

    def batched_get_neighbours(self, solve_configs, states, filleds, multi_solve_config=True):
        batch_size = solve_configs.shape[0]
        neighbors = jnp.zeros((2, batch_size, 1))
        costs = jnp.broadcast_to(jnp.array([[1.0], [2.0]]), (2, batch_size))
        return neighbors, costs


class _DeadEndSuccessorPuzzle(_TwoActionPuzzle):
    def batched_get_neighbours(self, solve_configs, states, filleds, multi_solve_config=True):
        batch_size = solve_configs.shape[0]
        is_root = states.reshape((batch_size, -1))[:, 0] == 0
        neighbors = jnp.ones((2, batch_size, 1))
        costs = jnp.stack(
            (
                jnp.where(is_root, 1.0, jnp.inf),
                jnp.full((batch_size,), jnp.inf),
            )
        )
        return neighbors, costs


class _ConstantQModel:
    def apply(self, params, preproc, training=False):
        return jnp.broadcast_to(jnp.asarray(params["values"]), (preproc.shape[0], 2))


class _ConstantHeuristicModel:
    def apply(self, params, preproc, training=False):
        return jnp.full((preproc.shape[0], 1), params["value"])


def _q_dataset(
    monkeypatch,
    *,
    target_values,
    online_values,
    munchausen=False,
    diffusion_value=jnp.inf,
    trajectory_action=1,
    puzzle=None,
    initial_state=0.0,
):
    monkeypatch.setattr(
        qfunction_builder,
        "boltzmann_action_selection",
        lambda q_values, temperature: jnp.broadcast_to(jnp.array([[1.0, 0.0]]), q_values.shape),
    )
    monkeypatch.setattr(
        qfunction_builder,
        "_compute_diffusion_q",
        lambda solve_configs, *args, **kwargs: jnp.full(
            (solve_configs.shape[0], 1), diffusion_value
        ),
    )
    return qfunction_builder._get_datasets_with_policy(
        puzzle=puzzle or _TwoActionPuzzle(),
        preproc_fn=lambda solve_config, state: jnp.atleast_1d(state),
        SolveConfigsAndStatesAndActions=None,
        SolveConfigsAndStates=None,
        q_model=_ConstantQModel(),
        minibatch_size=1,
        target_q_params={"values": jnp.asarray(target_values)},
        q_params={"values": jnp.asarray(online_values)},
        shuffled_path={
            "solve_configs": jnp.array([0]),
            "states": jnp.array([initial_state]),
            "actions": jnp.array([trajectory_action]),
            "move_costs": jnp.array([10.0]),
            "action_costs": jnp.array([2.0]),
            "parent_indices": jnp.array([-1]),
        },
        key=jax.random.PRNGKey(0),
        k_max=1,
        munchausen=munchausen,
    )


def _heuristic_dataset(
    monkeypatch,
    *,
    target_value,
    munchausen=False,
    puzzle=None,
    diffusion_value=jnp.inf,
):
    monkeypatch.setattr(
        heuristic_builder,
        "_compute_diffusion_distance",
        lambda solve_configs, *args, **kwargs: jnp.full(solve_configs.shape, diffusion_value),
    )
    return heuristic_builder._get_datasets(
        puzzle=puzzle or _TwoActionPuzzle(),
        preproc_fn=lambda solve_config, state: jnp.atleast_1d(state),
        SolveConfigsAndStates=None,
        heuristic_model=_ConstantHeuristicModel(),
        minibatch_size=2,
        target_heuristic_params={"value": target_value},
        heuristic_params={"value": -100.0},
        shuffled_path={
            "solve_configs": jnp.array([0, 0]),
            "states": jnp.array([0.0, 0.0]),
            "move_costs": jnp.array([10.0, 10.0]),
            "action_costs": jnp.array([1.0, 1.0]),
            "parent_indices": jnp.array([-1, -1]),
        },
        key=jax.random.PRNGKey(0),
        k_max=1,
        munchausen=munchausen,
    )


def test_munchausen_cost_target_matches_softmin_backup():
    current_q = jnp.array([[0.0, 1.0]])
    next_q = jnp.array([[1.0, 3.0]])
    alpha = 0.9
    tau = 0.5

    target = _munchausen_cost_target(
        current_q=current_q,
        next_q=next_q,
        actions=jnp.array([1]),
        selected_cost=jnp.array([2.0]),
        current_valid_mask=jnp.array([[True, True]]),
        next_valid_mask=jnp.array([[True, True]]),
        next_solved=jnp.array([False]),
        alpha=alpha,
        tau=tau,
        clip_min=-1.0,
    )

    scaled_log_policy = tau * jax.nn.log_softmax(-current_q / tau, axis=1)
    penalty = -alpha * jnp.clip(scaled_log_policy[0, 1], -1.0, 0.0)
    soft_next = -tau * jax.scipy.special.logsumexp(-next_q[0] / tau)
    assert jnp.allclose(target, 2.0 + penalty + soft_next)


def test_munchausen_log_discount_and_distance_coordinates_are_equivalent():
    rho = 0.99
    log_distance_scale = -jnp.log(rho)
    tau_log = 0.03
    clip_log = -1.0
    tau_distance = tau_log / log_distance_scale
    clip_distance = clip_log / log_distance_scale
    current_distance = jnp.array([[1.0, 3.0]])
    next_distance = jnp.array([[2.0, 4.0]])
    transition_cost = jnp.array([1.5])
    action = jnp.array([1])
    valid = jnp.array([[True, True]])

    distance_target = _munchausen_cost_target(
        current_q=current_distance,
        next_q=next_distance,
        actions=action,
        selected_cost=transition_cost,
        current_valid_mask=valid,
        next_valid_mask=valid,
        next_solved=jnp.array([False]),
        alpha=MUNCHAUSEN_ALPHA,
        tau=tau_distance,
        clip_min=clip_distance,
    )

    current_log_value = -log_distance_scale * current_distance
    next_log_value = -log_distance_scale * next_distance
    current_scaled_log_policy = tau_log * jax.nn.log_softmax(current_log_value / tau_log, axis=1)
    next_scaled_log_policy = tau_log * jax.nn.log_softmax(next_log_value / tau_log, axis=1)
    next_policy = jnp.exp(next_scaled_log_policy / tau_log)
    log_target = (
        jnp.log(rho) * transition_cost
        + MUNCHAUSEN_ALPHA * jnp.clip(current_scaled_log_policy[0, action[0]], clip_log, 0.0)
        + jnp.sum(next_policy * (next_log_value - next_scaled_log_policy), axis=1)
    )

    assert jnp.allclose(
        jax.nn.softmax(current_log_value / tau_log, axis=1),
        jax.nn.softmax(-current_distance / tau_distance, axis=1),
    )
    assert jnp.allclose(distance_target, -log_target / log_distance_scale)
    assert jnp.isclose(MUNCHAUSEN_LOG_DISCOUNT, jnp.exp(-MUNCHAUSEN_LOG_DISTANCE_SCALE))
    assert jnp.isclose(MUNCHAUSEN_TAU, MUNCHAUSEN_LOG_TAU / MUNCHAUSEN_LOG_DISTANCE_SCALE)
    assert jnp.isclose(
        MUNCHAUSEN_CLIP_MIN,
        MUNCHAUSEN_LOG_CLIP_MIN / MUNCHAUSEN_LOG_DISTANCE_SCALE,
    )


def test_default_q_td_target_is_unchanged(monkeypatch):
    dataset = _q_dataset(
        monkeypatch,
        target_values=[3.0, 5.0],
        online_values=[100.0, -100.0],
    )

    assert jnp.allclose(dataset["distance"], jnp.array([[4.0]]))


def test_default_heuristic_td_target_is_unchanged(monkeypatch):
    dataset = _heuristic_dataset(monkeypatch, target_value=3.0)

    assert jnp.allclose(dataset["distance"], jnp.array([4.0, 4.0]))


def test_default_td_targets_keep_physical_diffusion_caps(monkeypatch):
    q_dataset = _q_dataset(
        monkeypatch,
        target_values=[3.0, 5.0],
        online_values=[0.0, 0.0],
        diffusion_value=0.25,
        trajectory_action=0,
    )
    heuristic_dataset = _heuristic_dataset(
        monkeypatch,
        target_value=3.0,
        diffusion_value=0.25,
    )

    assert jnp.allclose(q_dataset["distance"], 0.25)
    assert jnp.allclose(heuristic_dataset["distance"], 0.25)


def test_munchausen_q_target_uses_frozen_target_values(monkeypatch):
    first = _q_dataset(
        monkeypatch,
        target_values=[0.0, 0.05],
        online_values=[0.0, 100.0],
        munchausen=True,
    )
    second = _q_dataset(
        monkeypatch,
        target_values=[0.0, 0.05],
        online_values=[100.0, 0.0],
        munchausen=True,
    )

    assert jnp.allclose(first["distance"], second["distance"])


def test_munchausen_targets_keep_physical_diffusion_caps(monkeypatch):
    q_dataset = _q_dataset(
        monkeypatch,
        target_values=[0.0, 0.05],
        online_values=[0.0, 0.0],
        munchausen=True,
        diffusion_value=0.0,
        trajectory_action=0,
    )
    heuristic_dataset = _heuristic_dataset(
        monkeypatch,
        target_value=-0.5,
        munchausen=True,
        diffusion_value=0.0,
    )

    assert jnp.allclose(q_dataset["distance"], 0.0)
    assert jnp.allclose(heuristic_dataset["distance"], 0.0)


def test_munchausen_q_preserves_nonterminal_dead_ends_as_infinite(monkeypatch):
    next_dead_end = _q_dataset(
        monkeypatch,
        target_values=[0.0, 0.0],
        online_values=[0.0, 0.0],
        munchausen=True,
        puzzle=_DeadEndSuccessorPuzzle(),
    )
    current_dead_end = _q_dataset(
        monkeypatch,
        target_values=[0.0, 0.0],
        online_values=[0.0, 0.0],
        munchausen=True,
        puzzle=_DeadEndSuccessorPuzzle(),
        initial_state=1.0,
    )

    assert jnp.all(jnp.isinf(next_dead_end["distance"]))
    assert jnp.all(jnp.isinf(current_dead_end["distance"]))


def test_munchausen_cost_target_masks_invalid_and_terminal_actions():
    kwargs = {
        "current_q": jnp.array([[0.0, 1.0]]),
        "next_q": jnp.array([[3.0, -100.0]]),
        "actions": jnp.array([1]),
        "selected_cost": jnp.array([2.0]),
        "current_valid_mask": jnp.array([[True, True]]),
        "next_valid_mask": jnp.array([[True, False]]),
        "alpha": 0.9,
        "tau": 0.03,
        "clip_min": -1.0,
    }
    selected_log_policy = 0.03 * jax.nn.log_softmax(-kwargs["current_q"] / 0.03)[0, 1]
    penalty = -0.9 * jnp.clip(selected_log_policy, -1.0, 0.0)

    assert jnp.allclose(
        _munchausen_cost_target(next_solved=jnp.array([False]), **kwargs),
        5.0 + penalty,
    )
    assert jnp.allclose(
        _munchausen_cost_target(next_solved=jnp.array([True]), **kwargs),
        2.0 + penalty,
    )


def test_munchausen_heuristic_uses_two_ply_softmin_then_greedy_projection(monkeypatch):
    dataset = _heuristic_dataset(
        monkeypatch,
        target_value=-0.5,
        munchausen=True,
    )
    transition_costs = jnp.array([[1.0, 2.0]])
    induced_action_distances = transition_costs - 0.5
    valid_mask = jnp.ones_like(transition_costs, dtype=bool)
    current_log_policy = scaled_log_softmin_policy(induced_action_distances, valid_mask)
    assert current_log_policy[0, 0] > current_log_policy[0, 1]
    soft_continuation = -MUNCHAUSEN_TAU * jax.scipy.special.logsumexp(
        -induced_action_distances[0] / MUNCHAUSEN_TAU
    )
    action_targets = (
        transition_costs[0]
        - MUNCHAUSEN_ALPHA * jnp.clip(current_log_policy[0], MUNCHAUSEN_CLIP_MIN, 0.0)
        + soft_continuation
    )
    assert jnp.allclose(dataset["distance"], jnp.min(action_targets))


def test_munchausen_heuristic_preserves_nonterminal_dead_end_as_infinite(monkeypatch):
    dataset = _heuristic_dataset(
        monkeypatch,
        target_value=0.0,
        munchausen=True,
        puzzle=_DeadEndSuccessorPuzzle(),
    )

    assert jnp.all(jnp.isinf(dataset["distance"]))


def test_munchausen_training_option_is_opt_in():
    options = DistTrainOptions()
    assert options.munchausen is False


def test_munchausen_cli_option_is_available_for_both_distance_models():
    runner = CliRunner()
    q_help = runner.invoke(qfunction_train_command, ["--help"])
    heuristic_help = runner.invoke(heuristic_train_command, ["--help"])

    assert q_help.exit_code == 0, q_help.output
    assert heuristic_help.exit_code == 0, heuristic_help.output
    assert "--munchausen" in q_help.output
    assert "--munchausen" in heuristic_help.output


def test_munchausen_rejects_incompatible_target_modes():
    for incompatible in (
        {"use_double_dqn": True},
        {"label": "diffusion"},
    ):
        with pytest.raises(ValueError, match="Munchausen"):
            get_qfunction_dataset_builder(
                None,
                None,
                None,
                1,
                1,
                1,
                munchausen=True,
                **incompatible,
            )

    with pytest.raises(ValueError, match="Munchausen"):
        get_heuristic_dataset_builder(
            None,
            None,
            None,
            1,
            1,
            1,
            munchausen=True,
            label="diffusion",
        )
