# Q-Learning Training Command (`qlearning`)

The `qlearning` command is used to train a neural network to serve as a Q-function. The goal of the model is to estimate the expected cost (Q-value) of taking a specific action from a given state. This is a fundamental concept in reinforcement learning and can be used to guide search algorithms like Q\*.

This command is intended for users interested in reinforcement learning, Q-functions, or training custom models for puzzles.

## Usage

The basic syntax for the `qlearning` command is:

```bash
python main.py qlearning [OPTIONS]
```

Example:

```bash
python main.py qlearning -p rubikscube -pre default -s 10000
```

## Options

The `qlearning` command uses several option groups to configure the training process.

### Puzzle Options (`@dist_puzzle_options`)

These options define the puzzle environment for which the Q-function is being trained.

-   `-p, --puzzle`: Specifies the puzzle to use for training.
    -   Type: `Choice`
    -   Default: `rubikscube`
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
-   `-ps, --puzzle_size`: A simpler way to set the size for puzzles that support it.
    -   Type: `String`
    -   Default: `default`
-   `-h, --hard`: If available, use a "hard" version of the puzzle for generating training data.
    -   Type: `Flag`
-   `-sl, --shuffle_length`: Overrides the default shuffle length when generating puzzle states for the dataset.
    -   Type: `Integer`

### Training Options (`@dist_train_options`)

These are the core options that control the training loop and hyperparameters.

-   `-pre, --preset`: Selects a training configuration preset (e.g., learning rate, batch sizes).
    -   Type: `Choice`
    -   Default: `default`
-   `-s, --steps`: Total number of training steps.
    -   Type: `Integer`
-   `-db, --dataset_batch_size`: The total number of states in the training dataset.
    -   Type: `Integer`
-   `-dmb, --dataset_minibatch_size`: The size of mini-batches to sample from the main dataset.
    -   Type: `Integer`
-   `-tmb, --train_minibatch_size`: The size of mini-batches used in a single training step.
    -   Type: `Integer`
-   `-k, --key`: Seed for the random number generator used in training.
    -   Type: `Integer`
-   `-r, --reset`: If flagged, resets the model weights before starting training.
    -   Type: `Flag`
-   `-lt, --loss_threshold`: A loss value threshold that can trigger updates or other events during training.
    -   Type: `Float`
-   `-ri, --reset_interval`: The interval (in steps) at which to reset the model's weights to a previous state, a technique to escape local minima.
    -   Type: `Integer`
-   `-ui, --update_interval`: The interval (in steps) for updating the target network.
    -   Type: `Integer`
-   `-su, --use_soft_update`: Use soft updates (Polyak averaging) for the target network instead of hard updates.
    -   Type: `Flag`
-   `-her, --using_hindsight_target`: Use Hindsight Experience Replay (HER) for generating target values.
    -   Type: `Flag`
-   `-is, --using_importance_sampling`: Use importance sampling to weigh losses during training.
    -   Type: `Flag`
-   `--optimizer`: The optimization algorithm to use.
    -   Type: `Choice`
    -   Choices: `adam`, `sgd`, etc.
    -   Default: `adam`
-   `-d, --debug`: Disables JIT compilation for easier debugging.
    -   Type: `Flag`
-   `-md, --multi_device`: Enables training across multiple JAX devices (e.g., multiple GPUs).
    -   Type: `Flag`

### Q-Function Model Options (`@dist_qfunction_options`)

This group contains options specific to the Q-function model being trained.

-   `-nwp, --not_with_policy`: If flagged, does not use the policy for training. The specifics of this depend on the `NeuralQFunctionBase` implementation.
    -   Type: `Flag`
