import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class TrainLogInfo:
    def __init__(
        self,
        name: str,
        data,
        log_mean: bool = True,
        log_histogram: bool = True,
        mean_name: str | None = None,
        histogram_name: str | None = None,
    ):
        self.name = name
        self.data = data
        self.log_mean = log_mean
        self.log_histogram = log_histogram
        self.mean_name_override = mean_name
        self.histogram_name_override = histogram_name

    @property
    def short_name(self):
        return self.name.split("/")[-1]

    @property
    def mean_name(self):
        if self.mean_name_override:
            return self.mean_name_override

        # Fallback: Metric/log -> Metric/mean_log
        parts = self.name.split("/")
        parts[-1] = "mean_" + parts[-1]
        return "/".join(parts)

    @property
    def histogram_name(self):
        return self.histogram_name_override if self.histogram_name_override else self.name

    def tree_flatten(self):
        return (self.data,), (
            self.name,
            self.log_mean,
            self.log_histogram,
            self.mean_name_override,
            self.histogram_name_override,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        name, log_mean, log_histogram, mean_name_override, histogram_name_override = aux_data
        return cls(
            name,
            children[0],
            log_mean,
            log_histogram,
            mean_name_override,
            histogram_name_override,
        )

    @property
    def mean(self):
        return jnp.mean(self.data)
