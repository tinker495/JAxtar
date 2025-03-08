from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


class RubiksCubeWorldModel_test(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube_test",
            data_shape=(32, 64, 3),
            latent_shape=(400,),
            action_size=12,
            **kwargs
        )


class RubiksCubeWorldModel(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube",
            data_shape=(32, 64, 3),
            latent_shape=(400,),
            action_size=12,
            **kwargs
        )


class RubiksCubeWorldModelOptimized_test(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube_test",
            data_shape=(32, 64, 3),
            latent_shape=(144,),  # almost optimal is 144 bits
            action_size=12,
            **kwargs
        )


class RubiksCubeWorldModelOptimized(WorldModelPuzzleBase):
    """
    This is the optimized version of the rubiks cube world model.
    rubiks cube has 6 faces, 9 stickers per face, 3 colors per sticker.
    6 colors -> 3 bits x (9 - 1) stickers per face x (6 faces) = 144 bits
    but we can reduce it with the fact that the stickers are arranged in a specific way.
    so we can reduce it to 144 bits. but it is hard to train, so we use 256 bits as 32x8
    """

    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube",
            data_shape=(32, 64, 3),
            latent_shape=(144,),
            action_size=12,
            **kwargs
        )
