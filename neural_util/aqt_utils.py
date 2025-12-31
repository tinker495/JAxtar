import jax
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax


def get_aqt_cfg(aqt_cfg: str = "int8"):
    if aqt_cfg == "int8":
        return get_int8_config()
    else:
        raise ValueError(f"Invalid AQT configuration: {aqt_cfg}")


def get_int8_config():
    """Returns a fully quantized int8 configuration for AQT."""
    return aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)


def convert_to_serving(model_cls, params, sample_input, aqt_cfg, **model_kwargs):
    """
    Converts a trained model with AQT config to a serving model (freezing quantization stats).

    Args:
        model_cls: The Flax model class (e.g. ResMLPModel).
        params: The trained parameters (FrozenDict).
        sample_input: A sample input array with correct shape and dtype.
        aqt_cfg: The AQT configuration used during training.
        **model_kwargs: Additional arguments for model initialization.

    Returns:
        serving_model: The model instance configured for serving.
        serving_variables: The variables (params + frozen quant stats) for serving.
    """
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
    _, converting_variables = convert_model.apply(
        variables,
        sample_input,
        training=True,  # Often needed to activate stats collection if any
        mutable=True,
    )

    # 3. Create Serving Model
    serve_kwargs = model_kwargs.copy()
    serve_kwargs["aqt_cfg"] = aqt_cfg
    serve_kwargs["quant_mode"] = aqt_flax.QuantMode.SERVE

    serving_model = model_cls(**serve_kwargs)

    return serving_model, converting_variables


def create_serving_fn(model_cls, serving_variables, aqt_cfg, **model_kwargs):
    """
    Creates a jit-compiled serving function.
    """
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
