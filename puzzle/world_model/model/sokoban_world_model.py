from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


class SokobanWorldModel(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/sokoban",
            data_shape=(40, 40, 3),
            latent_shape=(200,),
            action_size=4,
            **kwargs
        )
