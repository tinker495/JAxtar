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
    "diffusion_distance": DistTrainOptions(
        dataset_batch_size=8124 * 8 * 256,
        dataset_minibatch_size=8192 * 32,
        steps=1250,
        update_interval=16,
        use_diffusion_distance=True,
    ),
    "diffusion_distance_mixture": DistTrainOptions(
        dataset_batch_size=8124 * 8 * 256,
        dataset_minibatch_size=8192 * 32,
        steps=1250,
        update_interval=16,
        use_diffusion_distance_mixture=True,
    ),
    "diffusion_distance_warmup": DistTrainOptions(
        dataset_batch_size=8124 * 8 * 256,
        dataset_minibatch_size=8192 * 32,
        steps=1250,
        update_interval=16,
        use_diffusion_distance=True,
        use_diffusion_distance_warmup=True,
        diffusion_distance_warmup_steps=500,
    ),
}
