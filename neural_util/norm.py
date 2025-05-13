from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax.lax as lax
import jax.numpy as jnp
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Axes = Union[int, Sequence[int]]


class BatchReNorm(nn.Module):
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Any], Array] = nn.initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Any], Array] = nn.initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = nn.merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            "batch_stats",
            "r_max",
            lambda val: jnp.array(val, dtype=jnp.float32),
            3,
        )
        d_max = self.variable(
            "batch_stats",
            "d_max",
            lambda val: jnp.array(val, dtype=jnp.float32),
            5,
        )
        steps = self.variable(
            "batch_stats",
            "steps",
            lambda val: jnp.array(val, dtype=jnp.float32),  # Changed from int32 to float32
            0,
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                r = 1
                d = 0
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
                warmed_up = jnp.greater_equal(steps.value, 100).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1.0 - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1.0 - warmed_up) * custom_mean

                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
