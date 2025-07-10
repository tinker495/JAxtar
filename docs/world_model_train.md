# World Model Training Command (`world_model_train train`)

The `world_model_train train` command trains a discrete world model for a specific puzzle environment. This model learns the transition dynamics of the puzzle—that is, given a state and an action, it predicts the next state. It typically consists of an encoder, a decoder, and a transition model. This allows for planning and search to be performed in a learned latent space, which can be much more efficient than searching in the original state space.

This command is intended for users interested in model-based reinforcement learning and heuristic search.

## Usage

The basic syntax for the `world_model_train train` command is:

```bash
python main.py world_model_train train [OPTIONS]
```

Example:

```bash
python main.py world_model_train train --dataset rubikscube --world_model rubikscube --train_epochs 2000
```

## Options

The `world_model_train train` command uses several option groups to configure the dataset, the model architecture, and the training process.

### Dataset Options (`@wm_get_ds_options`)

These options specify the dataset of state transitions used for training.

-   `--dataset`: The name of the dataset to use. This typically corresponds to a puzzle environment.
    -   Type: `Choice`
    -   Default: `rubikscube`
-   `--dataset_size`: The total number of state transitions to generate for the dataset.
    -   Type: `Integer`
    -   Default: `300000`
-   `--dataset_minibatch_size`: The size of mini-batches for processing the dataset generation.
    -   Type: `Integer`
    -   Default: `30000`
-   `--shuffle_length`: The number of random moves to apply to the solved state to generate initial states for the dataset.
    -   Type: `Integer`
    -   Default: `30`
-   `--img_size`: The height and width of the images to be generated for the dataset.
    -   Type: `Tuple (int, int)`
    -   Default: `(32, 32)`
-   `--key`: The random seed for dataset generation.
    -   Type: `Integer`
    -   Default: `0`

### World Model Options (`@wm_get_world_model_options`)

These options define the world model to be trained.

-   `--world_model`: The name of the world model architecture to use.
    -   Type: `Choice`
    -   Default: `rubikscube`

### Training Options (`@wm_train_options`)

These options control the training loop and hyperparameters for the world model.

-   `--train_epochs`: The total number of training epochs.
    -   Type: `Integer`
    -   Default: `2000`
-   `--mini_batch_size`: The size of mini-batches used in a single training step.
    -   Type: `Integer`
    -   Default: `1000`
-   `--optimizer`: The optimization algorithm to use.
    -   Type: `Choice`
    -   Choices: `adam`, `sgd`, etc.
    -   Default: `adam`
