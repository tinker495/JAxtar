from .pydantic_models import DistTrainOptions, WBSDistTrainOptions

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
        steps=int(1e4),
        update_interval=16,
        use_soft_update=True,
        use_diffusion_distance=True,
    ),
    "diffusion_distance_mixture": DistTrainOptions(
        steps=int(1e4),
        update_interval=128,
        use_diffusion_distance_mixture=True,
    ),
    "diffusion_distance_warmup": DistTrainOptions(
        steps=int(1e4),
        update_interval=128,
        use_diffusion_distance=True,
        use_diffusion_distance_warmup=True,
        diffusion_distance_warmup_steps=500,
    ),
}

wbs_train_presets = {
    "default": WBSDistTrainOptions(),
    "high_capacity": WBSDistTrainOptions(
        replay_size=int(5e8),
        max_nodes=int(5e7),
        steps=int(5e3),
    ),
    "fast": WBSDistTrainOptions(
        steps=int(1e3),
        replay_size=int(5e7),
        max_nodes=int(1e7),
        add_batch_size=262144,
        search_batch_size=4096,
        train_minibatch_size=4096,
    ),
    "optimal_branch": WBSDistTrainOptions(
        use_optimal_branch=True,
        sample_ratio=0.5,
    ),
}
