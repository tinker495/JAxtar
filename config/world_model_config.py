from puzzle import Puzzle, RubiksCubeDS, SokobanDS
from puzzle.world_model import RubiksCubeWorldModel, SokobanWorldModel

puzzle_dict_ds: dict[str, Puzzle] = {
    "rubikscube": RubiksCubeDS,
    "sokoban": SokobanDS,
}

world_model_dict: dict[str, callable] = {
    "rubikscube": lambda reset: RubiksCubeWorldModel()
    if reset
    else RubiksCubeWorldModel.load_model("puzzle/world_model/model/params/rubikscube.pkl"),
    "sokoban": lambda reset: SokobanWorldModel()
    if reset
    else SokobanWorldModel.load_model("puzzle/world_model/model/params/sokoban.pkl"),
}


world_model_ds_dict: dict[str, str] = {
    "rubikscube": "puzzle/world_model/data/rubikscube/transition",
    "sokoban": "puzzle/world_model/data/sokoban/transition",
}
