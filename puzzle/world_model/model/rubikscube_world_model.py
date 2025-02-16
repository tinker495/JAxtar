from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


class RubiksCubeWorldModel(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/rubikscube",
            data_shape=(64, 32, 3),
            latent_shape=(400,),
            action_size=12,
            **kwargs
        )
