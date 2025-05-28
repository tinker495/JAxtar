import flashbax as fbx
import jax.numpy as jnp

from puzzle import Puzzle

BUFFER_TYPE = fbx.flat_buffer.TrajectoryBuffer
BUFFER_STATE_TYPE = fbx.flat_buffer.TrajectoryBufferState


def init_trajectory_experience_replay(
    solve_config: Puzzle.SolveConfig,
    state: Puzzle.State,
    sample_batch_size: int = 100,
    add_batch_size: int = 100,
    sample_sequence_length: int = 30,
) -> tuple[BUFFER_TYPE, BUFFER_STATE_TYPE]:
    buffer = fbx.make_trajectory_buffer(
        add_batch_size=add_batch_size,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        min_length_time_axis=sample_sequence_length,
    )
    # Initialise the buffer's state.
    fake_timestep = {
        "solve_config": solve_config.default(),
        "state": state.default(),
        "cost": jnp.array(0.0, dtype=jnp.float16),
        "action": jnp.array(0, dtype=jnp.uint8),
    }
    state = buffer.init(fake_timestep)
    return buffer, state
