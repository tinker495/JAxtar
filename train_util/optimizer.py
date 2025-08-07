from typing import Any

import jax
import optax

PyTree = Any

OPTIMIZERS = {
    "adam": optax.scale_by_adam,
    "nadam": lambda **kwargs: optax.scale_by_adam(nesterov=True, **kwargs),
    "adopt": optax.contrib.scale_by_adopt,
    "nadopt": lambda **kwargs: optax.contrib.scale_by_adopt(nesterov=True, **kwargs),
    "rmsprop": optax.scale_by_rms,
    "lamb_adam": None,  # This is a placeholder for the lamb optimizer
    "lamb_adopt": None,  # This is a placeholder for the lamb optimizer
}


def setup_optimizer(
    params: PyTree,
    num_devices: int,
    steps: int,
    one_iter_size: int,
    optimizer_name: str,
    lr_init: float = 1e-3,
    weight_decay_size: float = 0.001,
) -> optax.OptState:
    # Create the main decay schedule, making it conditional
    is_lamb = optimizer_name.startswith("lamb")
    optimizer_name = optimizer_name.replace("lamb_", "")
    is_no_wd = weight_decay_size == 0.0

    # Add warmup to the learning rate schedule
    lr = lr_init * num_devices

    warmup_steps = one_iter_size

    # Create a warmup schedule that linearly increases from 0 to init_value
    warmup_schedule = optax.linear_schedule(
        init_value=0.0, end_value=lr, transition_steps=warmup_steps
    )

    decay_schedule = optax.schedules.exponential_decay(
        lr,
        5000,
        0.995,
    )
    # Combine the schedules
    lr_schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule], boundaries=[warmup_steps]
    )

    def mask_batch_stat_or_bias(params):
        def mask_fn(path, value):
            # Check if 'batch_stats' is part of any dictionary key in the path
            is_batch_stat = any(
                isinstance(entry, jax.tree_util.DictKey) and "batch_stats" in entry.key
                for entry in path
            )
            is_bias = (
                path and isinstance(path[-1], jax.tree_util.DictKey) and path[-1].key == "bias"
            )
            return not (is_batch_stat or is_bias)

        return jax.tree_util.tree_map_with_path(mask_fn, params)

    def optimizer_fn(learning_rate):
        if optimizer_name not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scaler = OPTIMIZERS[optimizer_name]()

        if is_lamb:
            chain = optax.chain(
                scaler,
                optax.add_decayed_weights(weight_decay_size, mask=mask_batch_stat_or_bias)
                if not is_no_wd
                else optax.identity(),
                optax.scale_by_trust_ratio(),
                optax.scale_by_learning_rate(learning_rate),
            )
            return chain
        else:
            chain = optax.chain(
                scaler,
                optax.add_decayed_weights(weight_decay_size, mask=mask_batch_stat_or_bias)
                if not is_no_wd
                else optax.identity(),
                optax.scale_by_learning_rate(learning_rate),
            )
            return chain

    optimizer = optax.inject_hyperparams(optimizer_fn)(lr_schedule)
    return optimizer, optimizer.init(params)
