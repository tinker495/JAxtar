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
    replay_size: int = 1000000,
) -> tuple[BUFFER_TYPE, BUFFER_STATE_TYPE]:
    buffer = fbx.make_flat_buffer(
        max_length=replay_size // sample_batch_size + 1,
        min_length=sample_batch_size,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
    )
    # Initialise the buffer's state.
    fake_timestep = {
        "solve_config": solve_config.default(),
        "state": state.default((sample_sequence_length + 1,)),
        "cost": jnp.zeros((sample_sequence_length + 1,), dtype=jnp.float32),
        "action": jnp.zeros((sample_sequence_length + 1,), dtype=jnp.uint8),
    }
    state = buffer.init(fake_timestep)
    return buffer, state
