from typing import Any

import jax
import optax
from flax.training import train_state


class TrainStateExtended(train_state.TrainState):
    batch_stats: Any = None
    target_params: Any = None
    target_batch_stats: Any = None

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx: optax.GradientTransformation,
        batch_stats=None,
        target_params=None,
        target_batch_stats=None,
        **kwargs,
    ):
        opt_state = tx.init(params)
        if target_params is None:
            target_params = jax.tree_util.tree_map(lambda x: x, params)
        if target_batch_stats is None and batch_stats is not None:
            target_batch_stats = jax.tree_util.tree_map(lambda x: x, batch_stats)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            batch_stats=batch_stats,
            target_params=target_params,
            target_batch_stats=target_batch_stats,
            **kwargs,
        )

    def get_full_params(self) -> dict:
        full_params = {"params": self.params}
        if self.batch_stats is not None:
            full_params["batch_stats"] = self.batch_stats
        return full_params

    def get_full_eval_params(self) -> dict:
        """Like get_full_params, but resolves schedule-free evaluation weights from opt_state when applicable."""
        from train_util.optimizer import get_eval_params

        eval_params = get_eval_params(self.opt_state, self.params)
        full_params = {"params": eval_params}
        if self.batch_stats is not None:
            full_params["batch_stats"] = self.batch_stats
        return full_params

    def apply_gradients(self, *, grads, **kwargs):
        new_state = super().apply_gradients(grads=grads, **kwargs)
        return new_state.replace(
            batch_stats=self.batch_stats,
            target_params=self.target_params,
            target_batch_stats=self.target_batch_stats,
        )

    def update_batch_stats(self, batch_stats):
        return self.replace(batch_stats=batch_stats)

    def update_target_params(self, target_params):
        return self.replace(target_params=target_params)


def soft_update_target(state: TrainStateExtended, tau: float) -> TrainStateExtended:
    from train_util.optimizer import get_eval_params

    eval_params = get_eval_params(state.opt_state, state.params)
    new_target_params = optax.incremental_update(eval_params, state.target_params, tau)
    return state.replace(
        target_params=new_target_params,
        target_batch_stats=state.batch_stats,
    )


def hard_update_target(state: TrainStateExtended) -> TrainStateExtended:
    from train_util.optimizer import get_eval_params

    eval_params = get_eval_params(state.opt_state, state.params)
    return state.replace(
        target_params=eval_params,
        target_batch_stats=state.batch_stats,
    )
