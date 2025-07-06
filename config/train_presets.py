from .pydantic_models import DistTrainOptions

train_presets = {
    "default": DistTrainOptions(),
    "quality": DistTrainOptions(
        steps=int(5e4),
        update_interval=1024,
    ),
    "hindsight": DistTrainOptions(
        using_hindsight_target=True,
    ),
    "hindsight_quality": DistTrainOptions(
        using_hindsight_target=True,
        update_interval=1024,
    ),
}
