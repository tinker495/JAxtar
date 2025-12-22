# Q-Learning Training Command (`distance_train qfunction`)

The `distance_train qfunction` command is used to train a neural network to serve as a Q-function. The goal of the model is to estimate the expected cost (Q-value) of taking a specific action from a given state. This is a fundamental concept in reinforcement learning and can be used to guide search algorithms like Q\*.

This command is intended for users interested in reinforcement learning, Q-functions, or training custom models for puzzles.

## Usage

The basic syntax for the `distance_train qfunction` command is:

```bash
python main.py distance_train qfunction [OPTIONS]
```

Example:

```bash
python main.py distance_train qfunction -p rubikscube -pre default -s 10000
```

## Options

The `distance_train qfunction` command uses several option groups to configure the training process.

### Puzzle Options (`@dist_puzzle_options`)

These options define the puzzle environment for which the Q-function is being trained.

-   `-p, --puzzle`: Specifies the puzzle to use for training.
    -   Type: `Choice`
    -   Default: `rubikscube`
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
-   `-h, --hard`: If available, use a "hard" version of the puzzle for generating training data. (Default: True for distance training)
    -   Type: `Flag`
-   `-s, --seeds`: Seed for the random puzzle (if supported).
    -   Type: `String`
    -   Default: `0`

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
    -   Type: `Boolean`
-   `-lt, --loss_threshold`: A loss value threshold that can trigger updates or other events during training.
    -   Type: `Float`
-   `-ui, --update_interval`: The interval (in steps) for updating the target network.
    -   Type: `Integer`
-   `-fui, --force_update_interval`: The interval (in steps) to force update the target network.
    -   Type: `Integer`
-   `-su, --use_soft_update`: Use soft updates (Polyak averaging) for the target network instead of hard updates.
    -   Type: `Flag`
-   `-ddn, --use_double_dqn`: Enable Double DQN target computation.
    -   Type: `Flag`
-   `-her, --using_hindsight_target`: Use Hindsight Experience Replay (HER) for generating target values.
    -   Type: `Flag`
-   `-ts, --using_triangular_sampling`: Use triangular sampling for generating states.
    -   Type: `Flag`
-   `-dd, --use_diffusion_distance`: Enable diffusion distance features in dataset creation.
    -   Type: `Flag`
-   `-ddm, --use_diffusion_distance_mixture`: Enable diffusion distance mixture features in dataset creation.
    -   Type: `Flag`
-   `--use_diffusion_distance_warmup`: Enable warmup schedule when using diffusion distance features.
    -   Type: `Flag`
-   `--diffusion_distance_warmup_steps`: Number of iterations to run before enabling diffusion distance features.
    -   Type: `Integer`
-   `--sampling-non-backtracking-steps`: Number of previous states to avoid revisiting during dataset sampling.
    -   Type: `Integer`
-   `-tp, --temperature`: Boltzmann temperature for action selection.
    -   Type: `Float`
-   `-d, --debug`: Disables JIT compilation for easier debugging.
    -   Type: `Flag`
-   `-md, --multi_device`: Enables training across multiple JAX devices (e.g., multiple GPUs).
    -   Type: `Boolean`
-   `-ri, --reset_interval`: The interval (in steps) at which to reset the model's weights to a previous state.
    -   Type: `Integer`
-   `-osr, --opt_state_reset`: Reset optimizer state when target network is updated.
    -   Type: `Boolean`
-   `--tau`: Tau parameter for soft updates or scaled reset.
    -   Type: `Float`
-   `--optimizer`: The optimization algorithm to use.
    -   Type: `Choice`
    -   Default: `adam`
-   `-lr, --learning_rate`: Learning rate.
    -   Type: `Float`
-   `-wd, --weight_decay_size`: Weight decay size for regularization.
    -   Type: `Float`
-   `--loss`: Select training loss function.
    -   Type: `Choice`
    -   Choices: `mse`, `huber`, `logcosh`, `asymmetric_huber`, `asymmetric_logcosh`
-   `--loss-args`: JSON object of additional keyword arguments for the selected loss.
    -   Type: `String`
-   `--td-error-clip`: Absolute clip value for TD-error; set <= 0 to disable.
    -   Type: `Float`
-   `-km, --k_max`: Override puzzle's default k_max (formerly shuffle_length).
    -   Type: `Integer`
-   `--logger`: Logger to use.
    -   Type: `Choice`
    -   Choices: `aim`, `tensorboard`, `wandb`, `none`

### Q-Function Model Options (`@dist_qfunction_options`)

This group contains options specific to the Q-function model being trained.

-   `--param-path`: Path to the Q-function parameter file.
    -   Type: `String`
-   `-nc, --neural_config`: Neural configuration JSON string. Overrides the default configuration.
    -   Type: `String`

### Evaluation Options (`@eval_options`)

These options control the evaluation performed during/after training.

-   `-b, --batch-size`: Batch size for search during evaluation.
    -   Type: `Integer`
-   `-m, --max-node-size`: Maximum number of nodes to search during evaluation.
    -   Type: `String`
-   `-w, --cost-weight`: Weight for cost in search.
    -   Type: `Float`
-   `-pr, --pop_ratio`: Ratio(s) for popping nodes from the priority queue.
    -   Type: `String`
-   `-ne, --num-eval`: Number of puzzles to evaluate.
    -   Type: `Integer`
-   `-rn, --run-name`: Name of the evaluation run.
    -   Type: `String`
-   `--use-early-stopping`: Enable early stopping based on success rate threshold.
    -   Type: `Boolean`
-   `--early-stop-patience`: Number of samples to check before considering early stopping.
    -   Type: `Integer`
-   `--early-stop-threshold`: Minimum success rate threshold for early stopping (0.0 to 1.0).
    -   Type: `Float`
