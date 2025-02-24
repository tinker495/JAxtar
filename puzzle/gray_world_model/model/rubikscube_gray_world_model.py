from puzzle.gray_world_model.gray_world_model_puzzle_base import (
    GrayWorldModelPuzzleBase,
)


class RubiksCubeGrayWorldModel_test(GrayWorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube_test",
            data_shape=(32, 64, 3),
            latent_shape=(144,),
            action_size=12,
            **kwargs
        )


class RubiksCubeGrayWorldModel(GrayWorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube",
            data_shape=(32, 64, 3),
            latent_shape=(144,),
            action_size=12,
            **kwargs
        )
