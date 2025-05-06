import flashbax as fbx
import jax.numpy as jnp

BUFFER_TYPE = fbx.flat_buffer.TrajectoryBuffer
BUFFER_STATE_TYPE = fbx.flat_buffer.TrajectoryBufferState


def init_experience_replay(
    dummy_preprocessed_state: jnp.ndarray,
    max_length: int = int(1e6),
    min_length: int = int(1e5),
    sample_batch_size: int = int(1e4),
    add_batch_size: int = int(1e4),
    use_action: bool = False,
) -> tuple[BUFFER_TYPE, BUFFER_STATE_TYPE]:
    buffer = fbx.make_flat_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
    )
    # Initialise the buffer's state.
    if use_action:
        fake_timestep = {
            "obs": dummy_preprocessed_state,
            "action": jnp.array(0, dtype=jnp.int32),
            "distance": jnp.array(0.0, dtype=jnp.bfloat16),
        }
    else:
        fake_timestep = {
            "obs": dummy_preprocessed_state,
            "distance": jnp.array(0.0, dtype=jnp.bfloat16),
        }
    state = buffer.init(fake_timestep)
    return buffer, state
