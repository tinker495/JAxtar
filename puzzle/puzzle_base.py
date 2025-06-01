from abc import ABC, abstractmethod
from typing import Any, TypeVar

import chex
import jax
import jax.numpy as jnp

from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puzzle.util import add_img_parser

T = TypeVar("T")


class Puzzle(ABC):

    action_size: int = None

    class State(PuzzleState):
        pass

    class SolveConfig(PuzzleState):
        pass

    def define_solve_config_class(self) -> PuzzleState:
        @state_dataclass
        class SolveConfig:
            TargetState: FieldDescriptor[self.State]

            def __str__(self, **kwargs):
                return self.TargetState.str(**kwargs)

        return SolveConfig

    @abstractmethod
    def define_state_class(self) -> PuzzleState:
        pass

    @property
    def has_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the environment has a target state or not.
        """
        return "TargetState" in self.SolveConfig.__annotations__.keys()

    @property
    def only_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the environment has only a target state or not.
        """
        return self.has_target and len(self.SolveConfig.__annotations__.keys()) == 1

    @property
    def fixed_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the target state is fixed and doesn't change.
        default is only_target, but if the target state is not fixed, you should redefine this function.
        """
        return self.only_target

    def __init__(self, **kwargs):
        """
        This function should be called in the __init__ of the subclass.
        """
        super().__init__()
        self.data_init()

        self.State = self.define_state_class()
        self.SolveConfig = self.define_solve_config_class()
        self.State = add_img_parser(self.State, self.get_img_parser())
        self.SolveConfig = add_img_parser(self.SolveConfig, self.get_solve_config_img_parser())

        self.get_initial_state = jax.jit(self.get_initial_state)
        self.get_solve_config = jax.jit(self.get_solve_config)
        self.get_inits = jax.jit(self.get_inits)
        self.get_neighbours = jax.jit(self.get_neighbours)
        self.batched_get_neighbours = jax.jit(self.batched_get_neighbours, static_argnums=(3,))
        self.is_solved = jax.jit(self.is_solved)
        self.batched_is_solved = jax.jit(self.batched_is_solved, static_argnums=(2,))

        if self.action_size is None:
            self.action_size = self._get_action_size()

    def data_init(self):
        """
        This function should be called in the __init__ of the subclass.
        If the puzzle need to load dataset, this function should be filled.
        """
        pass

    def _get_action_size(self) -> int:
        """
        This function should return the size of the action.
        """
        dummy_solve_config = self.SolveConfig.default()
        dummy_state = self.State.default()
        _, costs = self.get_neighbours(dummy_solve_config, dummy_state)
        return costs.shape[0]

    def get_solve_config_string_parser(self) -> callable:
        """
        This function should return a callable that takes a solve config and returns a string representation of it.
        function signature: (solve_config: SolveConfig) -> str
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        stringparser_state = self.get_string_parser()

        def stringparser(solve_config: "Puzzle.SolveConfig") -> str:
            return stringparser_state(solve_config.TargetState)

        return stringparser

    @abstractmethod
    def get_string_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a string representation of it.
        function signature: (state: State) -> str
        """
        pass

    def get_solve_config_img_parser(self) -> callable:
        """
        This function should return a callable that takes a solve config and returns a image representation of it.
        function signature: (solve_config: SolveConfig) -> jnp.ndarray
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        imgparser_state = self.get_img_parser()

        def imgparser(solve_config: "Puzzle.SolveConfig") -> jnp.ndarray:
            return imgparser_state(solve_config.TargetState)

        return imgparser

    @abstractmethod
    def get_img_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a image representation of it.
        function signature: (state: State) -> jnp.ndarray
        """
        pass

    def get_data(self, key=None) -> Any:
        """
        This function should be called in the __init__ of the subclass.
        If the puzzle need to load dataset, this function should be filled.
        """
        return None

    @abstractmethod
    def get_solve_config(self, key=None, data=None) -> SolveConfig:
        """
        This function should return a solve config.
        """
        pass

    @abstractmethod
    def get_initial_state(self, solve_config: SolveConfig, key=None, data=None) -> State:
        """
        This function should return a initial state.
        """
        pass

    def get_inits(self, key=None) -> tuple[State, SolveConfig]:
        """
        This function should return a initial state and solve config.
        """
        datakey, solveconfigkey, initkey = jax.random.split(key, 3)
        data = self.get_data(datakey)
        solve_config = self.get_solve_config(solveconfigkey, data)
        return solve_config, self.get_initial_state(solve_config, initkey, data)

    def batched_get_neighbours(
        self,
        solve_configs: SolveConfig,
        states: State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        """
        if multi_solve_config:
            return jax.vmap(self.get_neighbours, in_axes=(0, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )
        else:
            return jax.vmap(self.get_neighbours, in_axes=(None, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )

    @abstractmethod
    def get_neighbours(
        self, solve_config: SolveConfig, state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        pass

    def batched_is_solved(
        self, solve_configs: SolveConfig, states: State, multi_solve_config: bool = False
    ) -> bool:
        """
        This function should return a boolean array that indicates whether the state is the target state.
        """
        if multi_solve_config:
            return jax.vmap(self.is_solved, in_axes=(0, 0))(solve_configs, states)
        else:
            return jax.vmap(self.is_solved, in_axes=(None, 0))(solve_configs, states)

    @abstractmethod
    def is_solved(self, solve_config: SolveConfig, state: State) -> bool:
        """
        This function should return True if the state is the target state.
        if the puzzle has multiple target states, this function should return
        True if the state is one of the target conditions.
        e.g sokoban puzzle has multiple target states. box's position should
        be the same as the target position but the player's position can be different.
        """
        pass

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return f"action {action}"

    def batched_hindsight_transform(self, solve_configs: SolveConfig, states: State) -> SolveConfig:
        """
        This function shoulde transformt the state to the solve config.
        """
        return jax.vmap(self.hindsight_transform)(solve_configs, states)

    def solve_config_to_state_transform(
        self, solve_config: SolveConfig, key: jax.random.PRNGKey = None
    ) -> State:
        """
        This function shoulde transformt the solve config to the state.
        """
        assert self.has_target, "This puzzle does not have target state"
        assert self.only_target, (
            "Default solve config to state transform is for only target state,"
            "you should redefine this function"
        )
        return solve_config.TargetState

    def hindsight_transform(self, solve_config: SolveConfig, states: State) -> SolveConfig:
        """
        This function shoulde transformt the state to the solve config.
        """
        assert self.has_target, "This puzzle does not have target state"
        assert self.only_target, (
            "Default hindsight transform is for only target state,"
            "you should redefine this function"
        )
        solve_config.TargetState = states
        return solve_config

    def get_inverse_neighbours(
        self, solve_config: SolveConfig, state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return inverse neighbours and the cost of the move.
        For puzzles that are reversible, this function can simply return the same neighbours as `get_neighbours`.
        However, for puzzles like Sokoban, which are not reversible (pushing a box is not easily reversed),
        this function needs to be implemented specifically to return the actual inverse neighbours.
        By default, it can just use the `get_neighbours` function if inverse neighbours are not explicitly defined.
        """
        return self.get_neighbours(solve_config, state, filled)

    def batched_get_inverse_neighbours(
        self,
        solve_configs: SolveConfig,
        states: State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[State, chex.Array]:
        """
        This function should return inverse neighbours and the cost of the move.
        """
        if multi_solve_config:
            return jax.vmap(self.get_inverse_neighbours, in_axes=(0, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )
        else:
            return jax.vmap(self.get_inverse_neighbours, in_axes=(None, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )

    def _get_suffled_state(
        self, solve_config: "Puzzle.SolveConfig", init_state: "Puzzle.State", key, num_shuffle
    ):
        def cond_fun(loop_state):
            iteration_count, _, _, _ = loop_state
            return iteration_count < num_shuffle

        def body_fun(loop_state):
            iteration_count, current_state, previous_state, key = loop_state
            neighbor_states, costs = self.get_neighbours(solve_config, current_state, filled=True)
            old_eq = jax.vmap(lambda x, y: x == y, in_axes=(None, 0))(
                previous_state, neighbor_states
            )
            valid_mask = jnp.where(old_eq, 0.0, 1.0)

            valid_mask_sum = jnp.sum(valid_mask)
            probabilities = jax.lax.cond(
                valid_mask_sum > 0,
                lambda: valid_mask / valid_mask_sum,
                lambda: jnp.ones_like(costs) / costs.shape[0],
            )

            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(costs.shape[0]), p=probabilities)
            next_state = neighbor_states[idx]
            return (iteration_count + 1, next_state, current_state, key)

        _, final_state, _, _ = jax.lax.while_loop(
            cond_fun, body_fun, (0, init_state, init_state, key)
        )
        return final_state
