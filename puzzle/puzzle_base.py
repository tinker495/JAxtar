from abc import ABC, abstractmethod
from typing import Any, TypeVar

import chex
import jax
import jax.numpy as jnp

from puzzle.util import add_default, add_img_parser, add_string_parser, state_dataclass

T = TypeVar("T")


class Puzzle(ABC):
    @state_dataclass
    class State:
        """
        This class should be a dataclass that represents the state of the puzzle.
        """

        @abstractmethod
        def dtype(self):
            pass

        @abstractmethod
        def shape(self):
            pass

        @abstractmethod
        def default(_=None) -> T:
            pass

        @abstractmethod
        def img(self) -> jnp.ndarray:
            pass

    @state_dataclass
    class SolveConfig:
        """
        This class should be a dataclass that represents the configuration for solving the puzzle.

        Generally, most puzzles terminate when the current state is equal to the target state.
        However, there are many cases where this is not true.
        Some problems have independent, fixed variables separate from the state.
        For example, in TSP, the positions of the points are independent and fixed,
        but the state consists of the path traversed and the current index.
        In TSP, we don't know the target state (optimal path), but we do know the locations of the points.
        We define these fixed parameters as the SolveConfig.

        To manage these cases, we use SolveConfig, a broader concept that encompasses the idea of a TargetState.

        # noqa: E501
        Ex) n_puzzle, rubikscube: We know the solvable configuration (TargetState) and the entire logic of the environment.
                In this case, SolveConfig only needs TargetState.
                Variables: TargetState.
        Ex) maze: The map can vary for each maze problem, so the map must be included in SolveConfig.
                A TargetState can also be defined, so it should be included.
                Variables: Map, TargetState.
        Ex) TSP: We don't know the TargetState (optimal path), but the positions of the points, which are specified for each problem, must vary.
                Therefore, SolveConfig should contain the points.
                Variables: Points.
        Ex) Dotknot: We don't know the TargetState, and there are no specific values assigned for each problem, so SolveConfig should be empty.
                Variables: None.
        """

        TargetState: "Puzzle.State"

        def dtype(self):
            pass

        def shape(self):
            pass

        def default(_=None) -> T:
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

    def __init__(self, **kwargs):
        """
        This function should be called in the __init__ of the subclass.
        """
        super().__init__()
        self.data_init()

        self.State = add_string_parser(self.State, self.get_string_parser())
        self.State = add_default(self.State, self.get_default_gen())
        self.State = add_img_parser(self.State, self.get_img_parser())
        self.SolveConfig = add_string_parser(
            self.SolveConfig, self.get_solve_config_string_parser()
        )
        self.SolveConfig = add_default(self.SolveConfig, self.get_solve_config_default_gen())
        self.SolveConfig = add_img_parser(self.SolveConfig, self.get_solve_config_img_parser())

        self.get_initial_state = jax.jit(self.get_initial_state)
        self.get_solve_config = jax.jit(self.get_solve_config)
        self.get_inits = jax.jit(self.get_inits)
        self.get_neighbours = jax.jit(self.get_neighbours)
        self.batched_get_neighbours = jax.jit(self.batched_get_neighbours, static_argnums=(3,))
        self.is_solved = jax.jit(self.is_solved)
        self.batched_is_solved = jax.jit(self.batched_is_solved, static_argnums=(2,))
        self.is_equal = jax.jit(self.is_equal)

    def data_init(self):
        """
        This function should be called in the __init__ of the subclass.
        If the puzzle need to load dataset, this function should be filled.
        """
        pass

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

    def get_solve_config_default_gen(self) -> SolveConfig:
        """
        This function should return a default solve config.
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        default_state = self.State.default()

        def default_gen():
            return self.SolveConfig(TargetState=default_state)

        return default_gen

    @abstractmethod
    def get_default_gen(self) -> callable:
        """
        This function should return a callable that takes a state and returns a shape of it.
        function signature: (state: State) -> Dict[str, Any]
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
        data = self.get_data(key)
        solve_config = self.get_solve_config(key, data)
        return solve_config, self.get_initial_state(solve_config, key, data)

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

    def is_equal(self, state1: State, state2: State) -> bool:
        """
        This function should return True if the two states are equal.
        this functions must be all puzzle's state(dataclass) compatible, so this is not a abstract method.
        """
        tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), state1, state2)
        return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

    def batched_hindsight_transform(self, states: State) -> SolveConfig:
        """
        This function shoulde transformt the state to the solve config.
        """
        return jax.vmap(self.hindsight_transform)(states)

    def hindsight_transform(self, states: State) -> SolveConfig:
        """
        This function shoulde transformt the state to the solve config.
        """
        assert self.has_target, "This puzzle does not have target state"
        assert self.only_target, (
            "Default hindsight transform is for only target state,"
            "you should redefine this function"
        )
        solve_config = self.SolveConfig(TargetState=states)
        return solve_config
