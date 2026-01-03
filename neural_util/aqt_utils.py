import functools
import inspect

import jax
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.numerics import no_numerics
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze


def get_aqt_cfg(aqt_cfg: str = "int8"):
    """
    Returns an AQT configuration based on the provided string.
    Supported: 'int8', 'int4', 'int4_w8a', 'int8_w_only'
    """
    if aqt_cfg == "int8":
        return get_int8_config()
    elif aqt_cfg == "int4":
        return get_int4_config()
    elif aqt_cfg == "int4_w8a":
        return get_int4_weight_int8_act_config()
    elif aqt_cfg == "int8_w_only":
        return get_int8_weight_only_config()
    else:
        raise ValueError(f"Invalid AQT configuration: {aqt_cfg}")


def get_int8_config():
    """Returns a fully quantized 8-bit configuration (weights and activations)."""
    return aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)


def get_int4_config():
    """Returns a fully quantized 4-bit configuration (weights and activations)."""
    return aqt_config.fully_quantized(fwd_bits=4, bwd_bits=4)


def get_int4_weight_int8_act_config():
    """
    Returns a configuration with 4-bit weights and 8-bit activations.
    """
    cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
    # RHS is typically weights in nn.Dense
    # Update bits in the quantizers
    cfg.fwd.dg_quantizer.rhs.numerics = cfg.fwd.dg_quantizer.rhs.numerics.replace(bits=4)
    cfg.dlhs.dg_quantizer.rhs.numerics = cfg.dlhs.dg_quantizer.rhs.numerics.replace(bits=4)
    cfg.drhs.dg_quantizer.rhs.numerics = cfg.drhs.dg_quantizer.rhs.numerics.replace(bits=4)
    return cfg


def get_int8_weight_only_config():
    """Returns a configuration where only weights are quantized to 8-bit."""
    cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
    # LHS is activations, RHS is weights
    cfg.fwd.dg_quantizer.lhs.numerics = no_numerics.NoNumerics()
    cfg.dlhs.dg_quantizer.lhs.numerics = no_numerics.NoNumerics()
    cfg.drhs.dg_quantizer.lhs.numerics = no_numerics.NoNumerics()
    return cfg


def _resolve_freezer_mode(quant_mode):
    freezer_mode = getattr(aqt_flax, "FreezerMode", None)
    if freezer_mode is None:
        return None
    if quant_mode == aqt_flax.QuantMode.CONVERT:
        for name in ("CALIBRATION", "CALIBRATE", "WRITE", "CONVERT"):
            if hasattr(freezer_mode, name):
                return getattr(freezer_mode, name)
    if quant_mode == aqt_flax.QuantMode.SERVE:
        for name in ("CALIBRATION_AND_VALUE", "FREEZE", "FROZEN", "READ", "SERVE", "INFERENCE"):
            if hasattr(freezer_mode, name):
                return getattr(freezer_mode, name)
    for name in ("NONE",):
        if hasattr(freezer_mode, name):
            return getattr(freezer_mode, name)
    return None


def build_aqt_dot_general(aqt_cfg, quant_mode):
    """
    Build an AqtDotGeneral partial with quant/freezer modes wired for the active AQT version.
    """
    kwargs = {
        "rhs_quant_mode": quant_mode,
    }
    freezer_mode = _resolve_freezer_mode(quant_mode)
    if freezer_mode is not None:
        sig = inspect.signature(aqt_flax.AqtDotGeneral)
        if "rhs_freeze_mode" in sig.parameters:
            kwargs["rhs_freeze_mode"] = freezer_mode
        if "freezer_mode" in sig.parameters and "rhs_freeze_mode" not in kwargs:
            kwargs["freezer_mode"] = freezer_mode
    return functools.partial(aqt_flax.AqtDotGeneral, aqt_cfg, **kwargs)


def convert_to_serving(model_cls, params, sample_input, **model_kwargs):
    """
    Converts a trained model with AQT config to a serving model (freezing quantization stats).

    Args:
        model_cls: The Flax model class (e.g. ResMLPModel).
        params: The trained parameters (FrozenDict).
        sample_input: A sample input array with correct shape and dtype.
        **model_kwargs: Additional arguments for model initialization, including 'aqt_cfg'.

    Returns:
        serving_model: The model instance configured for serving.
        serving_variables: The variables (params + frozen quant stats) for serving.
    """
    # Extract aqt_cfg from model_kwargs
    aqt_cfg = model_kwargs.pop("aqt_cfg", None)

    if aqt_cfg is None:
        raise ValueError("aqt_cfg must be provided in model_kwargs")

    # If it's a string, resolve it to a DotGeneral config object
    if isinstance(aqt_cfg, str):
        aqt_cfg = get_aqt_cfg(aqt_cfg)

    # 1. Initialize model in CONVERT mode
    convert_kwargs = model_kwargs.copy()
    convert_kwargs["aqt_cfg"] = aqt_cfg
    convert_kwargs["quant_mode"] = aqt_flax.QuantMode.CONVERT

    convert_model = model_cls(**convert_kwargs)

    # 2. Run forward pass with mutable=True to update quantization statistics
    # We use training=True/False depending on how we want stats to be collected.
    # Usually CONVERT mode expects to see data distribution.
    # We assume 'params' contains the trained weights.
    # We pass them as the initial variables.

    # Note: If params structure matches what apply expects, we use it.
    # If params is {'params': ...}, we might need to handle batch_stats if they exist.

    variables = params

    # We run apply. The 'mutable' output will contain the updated variables including quant states.
    try:
        _, updated_vars = convert_model.apply(
            variables,
            sample_input,
            training=True,  # Often needed to activate stats collection if any
            mutable=True,
            rngs={"params": jax.random.PRNGKey(0)},
        )
    except Exception as e:
        if "INTERNAL: the requested functionality is not supported" in str(e):
            raise RuntimeError(
                "AQT Quantization failed. This is likely due to model dimensions not being aligned "
                "with XLA requirements (e.g. using prime numbers for hidden dimensions). "
                "Please ensure hidden_dim and other dimensions are multiples of 32 or 128."
            ) from e
        raise e

    # Merge the updated collections back into the original variables to preserve trained weights.
    # Flax apply with mutable=True only returns collections that were actually modified.
    if isinstance(variables, FrozenDict):
        new_vars = unfreeze(variables)
    else:
        new_vars = dict(variables)

    new_vars.update(updated_vars)
    converting_variables = freeze(new_vars)

    # 3. Create Serving Model
    serve_kwargs = model_kwargs.copy()
    serve_kwargs["aqt_cfg"] = aqt_cfg
    serve_kwargs["quant_mode"] = aqt_flax.QuantMode.SERVE

    serving_model = model_cls(**serve_kwargs)

    return serving_model, converting_variables


def create_serving_fn(model_cls, serving_variables, **model_kwargs):
    """
    Creates a jit-compiled serving function.
    """
    aqt_cfg = model_kwargs.pop("aqt_cfg", None)

    if aqt_cfg is None:
        raise ValueError("aqt_cfg must be provided in model_kwargs")

    if isinstance(aqt_cfg, str):
        aqt_cfg = get_aqt_cfg(aqt_cfg)

    serve_kwargs = model_kwargs.copy()
    serve_kwargs["aqt_cfg"] = aqt_cfg
    serve_kwargs["quant_mode"] = aqt_flax.QuantMode.SERVE

    serving_model = model_cls(**serve_kwargs)

    @jax.jit
    def serve_fn(x):
        return serving_model.apply(
            serving_variables, x, training=False, rngs={"params": jax.random.PRNGKey(0)}
        )

    return serve_fn
