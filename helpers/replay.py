import flashbax as fbx
import jax.numpy as jnp

from puzzle import Puzzle

BUFFER_TYPE = fbx.flat_buffer.TrajectoryBuffer
BUFFER_STATE_TYPE = fbx.flat_buffer.TrajectoryBufferState


def init_transition_experience_replay(
    solve_config: Puzzle.SolveConfig,
    state: Puzzle.State,
    max_length: int = int(1e6),
    min_length: int = int(1e5),
    sample_batch_size: int = int(1e4),
    add_batch_size: int = int(1e4),
    use_action: bool = False,
) -> tuple[BUFFER_TYPE, BUFFER_STATE_TYPE]:
    buffer = fbx.make_trajectory_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
    )
    # Initialise the buffer's state.
    if use_action:
        fake_timestep = {
            "solve_config": solve_config.default(),
            "next_state": state.default(),
            "state": state.default(),
            "cost": jnp.array(0.0, dtype=jnp.bfloat16),
            "action": jnp.array(0, dtype=jnp.uint8),
        }
    state = buffer.init(fake_timestep)
    return buffer, state
