from ..world_model_puzzle_base import WorldModelPuzzleBase


class _RubiksCubeWorldModelBase(WorldModelPuzzleBase):
    data_path = "world_model_puzzle/data/rubikscube"
    latent_shape = (400,)

    def __init__(self, **kwargs):
        super().__init__(
            data_path=self.data_path,
            data_shape=(32, 64, 3),
            latent_shape=self.latent_shape,
            action_size=12,
            **kwargs,
        )


class _ReversedDataMixin:
    def data_init(self):
        super().data_init()
        self.inits, self.targets = self.targets, self.inits


class RubiksCubeWorldModel_test(_RubiksCubeWorldModelBase):
    data_path = "world_model_puzzle/data/rubikscube_test"


class RubiksCubeWorldModel(_RubiksCubeWorldModelBase):
    pass


class RubiksCubeWorldModel_reversed(_ReversedDataMixin, RubiksCubeWorldModel):
    pass


class RubiksCubeWorldModelOptimized(_RubiksCubeWorldModelBase):
    """
    This is the optimized version of the rubiks cube world model.
    rubiks cube has 6 faces, 9 stickers per face, 3 colors per sticker.
    6 colors -> 3 bits x (9 - 1) stickers per face x (6 faces) = 144 bits
    but we can reduce it with the fact that the stickers are arranged in a specific way.
    so we can reduce it to 144 bits. but it is hard to train, so we use 200 bit as 8 x 25
    """

    latent_shape = (240,)  # almost optimal is 144 bits


class RubiksCubeWorldModelOptimized_test(RubiksCubeWorldModelOptimized):
    data_path = "world_model_puzzle/data/rubikscube_test"


class RubiksCubeWorldModelOptimized_reversed(_ReversedDataMixin, RubiksCubeWorldModelOptimized):
    pass
