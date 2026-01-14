import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class TrainLogInfo:
    def __init__(self, data, log_mean: bool = True, log_histogram: bool = True):
        self.data = data
        self.log_mean = log_mean
        self.log_histogram = log_histogram

    def tree_flatten(self):
        # We only treat 'data' as a child (PyTree node) because it contains JAX arrays.
        # log_mean and log_histogram are metadata (aux_data).
        return (self.data,), (self.log_mean, self.log_histogram)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)

    @property
    def mean(self):
        return jnp.mean(self.data)
