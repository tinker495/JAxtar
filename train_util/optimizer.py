from typing import Any, Mapping, Optional

import jax
import optax

PyTree = Any


def adoptw(
    learning_rate: float,
    weight_decay: float = None,
    weight_decay_mask: Optional[PyTree] = None,
    cautious_weight_decay: bool = True,
    **adopt_kwargs: Any,
) -> optax.GradientTransformation:
    if weight_decay is not None:
        return optax.chain(
            optax.contrib.scale_by_adopt(
                **adopt_kwargs,
            ),
            optax.contrib.add_cautious_weight_decay(weight_decay, weight_decay_mask)
            if cautious_weight_decay
            else optax.add_decayed_weights(weight_decay, weight_decay_mask),
            optax.scale_by_learning_rate(learning_rate),
        )
    else:
        return optax.chain(
            optax.contrib.scale_by_adopt(
                **adopt_kwargs,
            ),
            optax.scale_by_learning_rate(learning_rate),
        )


OPTIMIZERS = {
    "adam": lambda **kwargs: optax.adamw(
        cautious_weight_decay=True, mask=kwargs.get("weight_decay_mask", None), **kwargs
    ),
    "schedule_free_adamw": optax.contrib.schedule_free_adamw,
    "nadam": lambda **kwargs: optax.adamw(
        nesterov=True,
        cautious_weight_decay=True,
        mask=kwargs.get("weight_decay_mask", None),
        **kwargs,
    ),
    "adopt": adoptw,
    "nadopt": lambda **kwargs: adoptw(nesterov=True, cautious_weight_decay=True, **kwargs),
    "muon": lambda **kwargs: optax.contrib.muon(cautious_weight_decay=True, **kwargs),
    "normuon": lambda **kwargs: optax.contrib.normuon(cautious_weight_decay=True, **kwargs),
    "adago": lambda **kwargs: optax.contrib.adago(cautious_weight_decay=True, **kwargs),
    "noradago": lambda **kwargs: optax.contrib.adago(
        cautious_weight_decay=False, use_normuon=True, **kwargs
    ),
    "rmsprop": optax.rmsprop,
    "prodigy": optax.contrib.prodigy,
    "lamb_adam": optax.lamb,
}


def setup_optimizer(
    params: PyTree,
    num_devices: int,
    steps: int,
    one_iter_size: int,
    optimizer_name: str,
    lr_init: float = 1e-3,
    weight_decay_size: float = 0.001,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    # Create the main decay schedule, making it conditional
    is_no_wd = weight_decay_size == 0.0
    in_prodigy = "prodigy" in optimizer_name
    in_schedule_free = "schedule_free" in optimizer_name

    # Add warmup to the learning rate schedule
    lr = lr_init * num_devices if not in_prodigy else 1.0

    warmup_steps = one_iter_size

    # Create a warmup schedule that linearly increases from 0 to init_value
    warmup_schedule = optax.linear_schedule(
        init_value=1e-6, end_value=lr, transition_steps=warmup_steps
    )

    if not in_schedule_free:
        decay_schedule = optax.schedules.exponential_decay(
            lr,
            5000,
            0.995,
        )
        # Combine the schedules
        lr_schedule = optax.join_schedules(
            schedules=[warmup_schedule, decay_schedule], boundaries=[warmup_steps]
        )
    else:
        constant_schedule = optax.constant_schedule(lr)
        lr_schedule = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule], boundaries=[warmup_steps]
        )

    def mask_batch_stat_or_bias(params):
        def mask_fn(path, value):
            # Check if 'batch_stats' is part of any dictionary key in the path
            is_batch_stat = any(
                isinstance(entry, jax.tree_util.DictKey) and "batch_stats" in entry.key
                for entry in path
            )
            is_no_wd_param = (
                path
                and isinstance(path[-1], jax.tree_util.DictKey)
                and path[-1].key in ("bias", "scale", "beta")
            )
            return not (is_batch_stat or is_no_wd_param)

        return jax.tree_util.tree_map_with_path(mask_fn, params)

    mask_tree = None if is_no_wd else mask_batch_stat_or_bias(params)

    def optimizer_fn(learning_rate):
        if optimizer_name not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        if is_no_wd:
            scaler = OPTIMIZERS[optimizer_name](learning_rate=learning_rate)
        else:
            scaler = OPTIMIZERS[optimizer_name](
                learning_rate=learning_rate,
                weight_decay=weight_decay_size,
                weight_decay_mask=mask_tree,
            )
        return optax.chain(optax.clip_by_global_norm(1.0), scaler)

    optimizer = optax.inject_hyperparams(optimizer_fn)(lr_schedule)
    opt_state = optimizer.init(params)

    # test get eval params
    eval_params = get_eval_params(opt_state, params)
    assert eval_params is not None, "eval_params is None"
    return optimizer, opt_state


def _coerce_estim_lr(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_estim_lr(opt_state: Any) -> float | None:
    seen: set[int] = set()

    def _search(node: Any) -> float | None:
        if node is None:
            return None
        node_id = id(node)
        if node_id in seen:
            return None
        seen.add(node_id)

        if hasattr(node, "estim_lr"):
            estim_lr = _coerce_estim_lr(getattr(node, "estim_lr"))
            if estim_lr is not None:
                return estim_lr

        if isinstance(node, Mapping):
            for value in node.values():
                result = _search(value)
                if result is not None:
                    return result

        if isinstance(node, tuple):
            if hasattr(node, "_fields"):
                for field in node._fields:
                    result = _search(getattr(node, field))
                    if result is not None:
                        return result
            for value in node:
                result = _search(value)
                if result is not None:
                    return result

        if isinstance(node, list):
            for value in node:
                result = _search(value)
                if result is not None:
                    return result

        if hasattr(node, "__dict__"):
            for value in vars(node).values():
                result = _search(value)
                if result is not None:
                    return result

        if hasattr(node, "_asdict"):
            for value in node._asdict().values():
                result = _search(value)
                if result is not None:
                    return result

        return None

    return _search(opt_state)


def get_learning_rate(optimizer_state: optax.OptState):
    estim_lr = _extract_estim_lr(optimizer_state)
    if estim_lr is not None:
        return estim_lr
    return optimizer_state.hyperparams["learning_rate"]


def get_eval_params(optimizer_state: optax.OptState, params: PyTree):
    b1 = getattr(optimizer_state.inner_state, "b1", None)
    z = getattr(optimizer_state.inner_state, "z", None)
    if b1 is None or z is None:
        return params
    params_for_eval = optax.contrib.schedule_free_eval_params(optimizer_state.inner_state, params)
    return params_for_eval
