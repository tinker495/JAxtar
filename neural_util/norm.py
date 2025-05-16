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
    r_max: float = 3.0
    d_max: float = 5.0
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
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jnp.clip(std / ra_std, 1 / self.r_max, self.r_max)
                r = lax.stop_gradient(r)
                d = jnp.clip((mean - ra_mean.value) / ra_std, -self.d_max, self.d_max)
                d = lax.stop_gradient(d)
                custom_mean = mean - (std * d / r)
                custom_var = (var + self.epsilon) / (r * r) - self.epsilon

                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

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
