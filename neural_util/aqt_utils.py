import dataclasses
import functools
from typing import Any

import jax
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2.aqt_dot_general import DefaultDotGeneralQuantizer
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.numerics import no_numerics
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze


def get_aqt_cfg(aqt_cfg: Any = "int8"):
    if not isinstance(aqt_cfg, str):
        return aqt_cfg
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


def set_stochastic_rounding(cfg):
    def _enable(numerics):
        if hasattr(numerics, "replace"):
            # Guard for non-IntNumerics (e.g., NoNumerics), which lack replace(noise_fn=...).
            return numerics.replace(noise_fn=stochastic_rounding.RandomCenteredUniform())
        return numerics

    cfg.fwd.dg_quantizer.lhs.numerics = _enable(cfg.fwd.dg_quantizer.lhs.numerics)
    cfg.fwd.dg_quantizer.rhs.numerics = _enable(cfg.fwd.dg_quantizer.rhs.numerics)

    # Stochastic rounding on backward gradient quantizers prevents quantized training stagnation.
    cfg.dlhs.dg_quantizer.rhs.numerics = _enable(cfg.dlhs.dg_quantizer.rhs.numerics)
    cfg.drhs.dg_quantizer.rhs.numerics = _enable(cfg.drhs.dg_quantizer.rhs.numerics)

    return cfg


def get_int8_config():
    cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
    return cfg


def get_int4_config():
    return aqt_config.fully_quantized(fwd_bits=4, bwd_bits=4)


def get_int4_weight_int8_act_config():
    cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
    # RHS is typically weights in nn.Dense
    for dot_general in (cfg.fwd, cfg.dlhs, cfg.drhs):
        quantizer = dot_general.dg_quantizer
        assert isinstance(quantizer, DefaultDotGeneralQuantizer)
        numerics = quantizer.rhs.numerics
        assert dataclasses.is_dataclass(numerics) and not isinstance(numerics, type)
        quantizer.rhs.numerics = dataclasses.replace(numerics, bits=4)
    return cfg


def get_int8_weight_only_config():
    cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
    # LHS is activations, RHS is weights
    for dot_general in (cfg.fwd, cfg.dlhs, cfg.drhs):
        quantizer = dot_general.dg_quantizer
        assert isinstance(quantizer, DefaultDotGeneralQuantizer)
        quantizer.lhs.numerics = no_numerics.NoNumerics()
    # int32 accumulation is only valid when BOTH inputs are int8; with an
    # unquantized lhs the dot is bf16 x int8, so let lax pick the accumulator.
    aqt_config.set_accumulator_dtype(cfg, fwd_dtype=None, dlhs_dtype=None, drhs_dtype=None)
    return cfg


def build_aqt_dot_general(aqt_cfg, quant_mode):
    """
    Build an AqtDotGeneral partial with quant/freezer modes wired.

    CONVERT must freeze with CALIBRATION_AND_VALUE so the int8 weights (qvalue)
    are stored alongside the scales; CALIBRATION strips the qvalue at store time,
    which forces SERVE to re-quantize the f32 master weights on every forward.
    """
    if isinstance(aqt_cfg, str):
        aqt_cfg = get_aqt_cfg(aqt_cfg)

    if quant_mode == aqt_flax.QuantMode.TRAIN:
        aqt_cfg = set_stochastic_rounding(aqt_cfg)

    if quant_mode in (aqt_flax.QuantMode.CONVERT, aqt_flax.QuantMode.SERVE):
        rhs_freeze_mode = aqt_flax.FreezerMode.CALIBRATION_AND_VALUE
    else:
        rhs_freeze_mode = aqt_flax.FreezerMode.NONE

    return functools.partial(
        aqt_flax.AqtDotGeneral,
        aqt_cfg,
        rhs_quant_mode=quant_mode,
        rhs_freeze_mode=rhs_freeze_mode,
    )


def convert_to_serving(model_cls, params, sample_input, **model_kwargs):
    """Freeze AQT quantization state via a CONVERT pass and return (serving_model, serving_variables)."""
    aqt_cfg = model_kwargs.pop("aqt_cfg", None)

    if aqt_cfg is None:
        raise ValueError("aqt_cfg must be provided in model_kwargs")

    if isinstance(aqt_cfg, str):
        aqt_cfg = get_aqt_cfg(aqt_cfg)

    convert_kwargs = model_kwargs.copy()
    convert_kwargs["aqt_cfg"] = aqt_cfg
    convert_kwargs["quant_mode"] = aqt_flax.QuantMode.CONVERT

    convert_model = model_cls(**convert_kwargs)

    # Run CONVERT with training=False so the dummy sample_input does not corrupt batch_stats, while mutable=True populates AQT serving variables.

    variables = params

    try:
        _, updated_vars = convert_model.apply(
            variables,
            sample_input,
            training=False,
            mutable=True,
            rngs={"params": jax.random.PRNGKey(0)},
        )
    except (RuntimeError, ValueError) as e:
        if "requested functionality is not supported" in str(e):
            raise RuntimeError(
                "AQT Quantization failed: cuBLASLt has no int8 kernel for a quantized GEMM. "
                "Ensure every quantized dimension (hidden_dim etc.) is a multiple of 4 "
                "(32 recommended for tensor-core efficiency)."
            ) from e
        raise

    # Merge the updated collections back into the original variables to preserve trained weights.
    # Flax apply with mutable=True only returns collections that were actually modified.
    if isinstance(variables, FrozenDict):
        new_vars = unfreeze(variables)
    else:
        new_vars = dict(variables)

    new_vars.update(updated_vars)
    converting_variables = freeze(new_vars)

    serve_kwargs = model_kwargs.copy()
    serve_kwargs["aqt_cfg"] = aqt_cfg
    serve_kwargs["quant_mode"] = aqt_flax.QuantMode.SERVE

    serving_model = model_cls(**serve_kwargs)

    return serving_model, converting_variables
