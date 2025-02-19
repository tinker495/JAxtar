from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


class RubiksCubeWorldModel_test(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube_test",
            data_shape=(32, 64, 3),
            latent_shape=(160,),
            action_size=12,
            **kwargs
        )


class RubiksCubeWorldModel(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube",
            data_shape=(32, 64, 3),
            latent_shape=(160,),
            action_size=12,
            **kwargs
        )
