from .pydantic_models import DistTrainOptions

train_presets = {
    "default": DistTrainOptions(),
    "quality": DistTrainOptions(
        steps=int(1e4),
        update_interval=128,
    ),
    "largebatch": DistTrainOptions(
        dataset_batch_size=32768 * 256,
        dataset_minibatch_size=32768,
        train_minibatch_size=32768,
    ),
}
