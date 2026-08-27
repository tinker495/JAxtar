import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax

import train_util.distance_train_builder as train_builder_module
from train_util.sampling import wrap_dataset_runner
from train_util.train_logs import TrainLogInfo
from train_util.train_state import TrainStateExtended, hard_update_target, soft_update_target


def test_distance_train_builder_does_not_update_target(monkeypatch):
    def train_loss(params, batch_stats, solve_configs, states, target, key):
        del solve_configs, states, key
        loss = jnp.mean((params["w"] - target) ** 2)
        new_batch_stats = jax.tree_util.tree_map(lambda x: x + 1, batch_stats)
        return loss, (new_batch_stats, jnp.array(0.0))

    monkeypatch.setattr(
        train_builder_module,
        "build_distance_train_loss",
        lambda *args, **kwargs: train_loss,
    )
    monkeypatch.setattr("train_util.optimizer.get_eval_params", lambda _opt_state, params: params)
    optimizer = optax.sgd(0.1)
    state = TrainStateExtended.create(
        apply_fn=None,
        params={"w": jnp.array(1.0)},
        target_params={"w": jnp.array(0.0)},
        batch_stats={"mean": jnp.array(0.0)},
        target_batch_stats={"mean": jnp.array(10.0)},
        tx=optimizer,
    )
    train = train_builder_module.distance_train_builder(
        minibatch_size=1,
        model=object(),
        optimizer=optimizer,
        preproc_fn=lambda x: x,
        target_keys=("distance",),
    )
    dataset = {
        "solve_config": jnp.arange(4),
        "state": jnp.arange(4),
        "distance": jnp.zeros(4),
    }

    new_state, _, _, _ = train(jax.random.PRNGKey(0), dataset, state)

    assert int(new_state.step) == 4
    assert float(new_state.target_params["w"]) == 0.0
    assert float(new_state.batch_stats["mean"]) == 4.0
    assert float(new_state.target_batch_stats["mean"]) == 10.0

    host_updated_state = hard_update_target(new_state)
    assert float(host_updated_state.target_params["w"]) == float(new_state.params["w"])
    assert float(host_updated_state.target_batch_stats["mean"]) == 4.0


def test_distance_train_builder_reports_model_health_metrics(monkeypatch):
    def train_loss(params, batch_stats, solve_configs, states, target, key):
        del batch_stats, solve_configs, states, key
        loss = jnp.mean((params["w"] - target) ** 2)
        return loss, (None, jnp.array(0.0))

    monkeypatch.setattr(
        train_builder_module,
        "build_distance_train_loss",
        lambda *args, **kwargs: train_loss,
    )
    optimizer = optax.sgd(0.1)
    state = TrainStateExtended.create(
        apply_fn=None,
        params={"w": jnp.array(2.0)},
        target_params={"w": jnp.array(0.0)},
        tx=optimizer,
    )
    train = train_builder_module.distance_train_builder(
        minibatch_size=1,
        model=object(),
        optimizer=optimizer,
        preproc_fn=lambda x: x,
        target_keys=("distance",),
    )
    dataset = {
        "solve_config": jnp.array([0]),
        "state": jnp.array([0]),
        "distance": jnp.array([0.0]),
    }

    new_state, _, _, diagnostics = train(jax.random.PRNGKey(0), dataset, state)
    scalars = diagnostics["scalars"]

    assert math.isclose(float(new_state.params["w"]), 1.6, rel_tol=1e-6)
    assert math.isclose(float(scalars["Model/Parameter RMS"]), 1.6, rel_tol=1e-6)
    assert math.isclose(float(scalars["Optimizer/Gradient Global Norm Mean"]), 4.0, rel_tol=1e-6)
    assert math.isclose(float(scalars["Optimizer/Gradient Global Norm Max"]), 4.0, rel_tol=1e-6)
    assert math.isclose(float(scalars["Optimizer/Update RMS"]), 0.4, rel_tol=1e-6)
    assert math.isclose(float(scalars["Optimizer/Update to Parameter Ratio"]), 0.2, rel_tol=1e-6)
    assert math.isclose(float(scalars["Metrics/TD Target Online Gap Before Update"]), 1.0)
    assert float(scalars["Health/Nonfinite Count"]) == 0.0
    assert diagnostics["histograms"]["Model/Layer Parameter RMS"].shape == (1,)


def test_distance_diagnostics_logs_td_prediction_and_histograms():
    from cli.train_commands.dist_train_command import _log_distance_diagnostics

    class Logger:
        def __init__(self):
            self.scalars = {}
            self.histograms = {}

        def log_scalar(self, name, value, step):
            self.scalars[(name, step)] = float(value)

        def log_histogram(self, name, values, step):
            self.histograms[(name, step)] = values

    logger = Logger()
    diagnostics = {
        "scalars": {"Model/Parameter RMS": jnp.array(2.0)},
        "histograms": {"Model/Layer Parameter RMS": jnp.array([1.0, 3.0])},
    }
    log_infos = (
        TrainLogInfo("Losses/diff", jnp.array([-1.0, 2.0]), log_mean=False),
        TrainLogInfo("Metrics/pred", jnp.array([2.0, 4.0])),
    )

    _log_distance_diagnostics(logger, diagnostics, log_infos, step=100)

    assert math.isclose(logger.scalars[("Metrics/TD Error RMS", 100)], math.sqrt(2.5), rel_tol=1e-6)
    assert math.isclose(logger.scalars[("Metrics/Prediction Std", 100)], 1.0)
    assert ("Model/Layer Parameter RMS", 100) in logger.histograms


def test_dataset_runner_uses_frozen_target_batch_stats(monkeypatch):
    monkeypatch.setattr("train_util.optimizer.get_eval_params", lambda _opt_state, params: params)

    def create_path(_key):
        return {"row": jnp.array([0])}

    def extract(target_params, params, _paths, _key):
        return {
            "target_mean": target_params["batch_stats"]["mean"],
            "online_mean": params["batch_stats"]["mean"],
        }

    state = TrainStateExtended.create(
        apply_fn=None,
        params={"w": jnp.array(1.0)},
        tx=optax.sgd(0.1),
        batch_stats={"mean": jnp.array(2.0)},
        target_batch_stats={"mean": jnp.array(9.0)},
    )
    get_dataset = wrap_dataset_runner(
        dataset_size=1,
        steps=1,
        jited_create_shuffled_path=create_path,
        base_get_datasets=extract,
        diffusion_get_datasets=extract,
        should_use_diffusion_fn=lambda _step: False,
    )

    dataset = get_dataset(state, jax.random.PRNGKey(0), 0)

    assert float(dataset["target_mean"]) == 9.0
    assert float(dataset["online_mean"]) == 2.0


def test_soft_update_updates_full_target_state(monkeypatch):
    monkeypatch.setattr("train_util.optimizer.get_eval_params", lambda _opt_state, params: params)

    state = TrainStateExtended.create(
        apply_fn=None,
        params={"w": jnp.array(1.0)},
        target_params={"w": jnp.array(0.0)},
        batch_stats={"mean": jnp.array(2.0)},
        target_batch_stats={"mean": jnp.array(9.0)},
        tx=optax.sgd(0.1),
    )

    updated = soft_update_target(state, 0.25)

    assert float(updated.target_params["w"]) == 0.25
    assert float(updated.target_batch_stats["mean"]) == 2.0


def test_hard_target_update_schedule_uses_outer_iteration():
    from cli.train_commands.dist_train_command import _should_update_hard_target

    kwargs = {
        "last_update_iteration": 0,
        "update_interval": 3,
        "force_update_interval": 10,
        "loss": 0.5,
        "loss_threshold": 1.0,
    }

    assert not _should_update_hard_target(iteration=1, **kwargs)
    assert not _should_update_hard_target(iteration=2, **kwargs)
    assert _should_update_hard_target(iteration=3, **kwargs)
    assert not _should_update_hard_target(
        iteration=9,
        **{**kwargs, "loss": 2.0},
    )
    assert _should_update_hard_target(
        iteration=10,
        **{**kwargs, "loss": 2.0},
    )


def test_update_interval_scales_with_multi_device():
    from cli.train_commands.dist_train_command import (
        _effective_update_interval,
        _soft_update_tau,
    )

    assert _effective_update_interval(32, n_devices=4) == 8
    assert _effective_update_interval(4, n_devices=8) == 1
    assert _effective_update_interval(32, n_devices=1) == 32
    assert math.isclose(_soft_update_tau(32, n_devices=1), 1.0 / 32)
    assert math.isclose(_soft_update_tau(32, n_devices=4), 1.0 - (31.0 / 32) ** 4)


def test_training_component_options_preserve_startup_initialization_choice():
    from cli.options import dist_heuristic_options, dist_qfunction_options
    from config.pydantic_models import DistTrainOptions

    constructed = []

    class FakeComponent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.metadata = {}

    component_config = SimpleNamespace(callable=FakeComponent, param_path="unused.pkl")
    bundle = SimpleNamespace(
        heuristic_nn_configs={"default": component_config},
        q_function_nn_configs={"default": component_config},
    )
    common_kwargs = {
        "puzzle_bundle": bundle,
        "puzzle": SimpleNamespace(size=2),
        "puzzle_name": "test",
        "train_options": DistTrainOptions(reset=False),
        "param_path": "unused.pkl",
        "neural_config": None,
        "model_type": None,
        "use_quantize": False,
        "quant_type": "int8",
    }

    for decorator in (dist_heuristic_options, dist_qfunction_options):
        decorator(lambda **kwargs: kwargs)(**common_kwargs)

    assert [kwargs["init_params"] for kwargs in constructed] == [False, False]
