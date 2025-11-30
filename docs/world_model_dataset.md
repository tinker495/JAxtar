# World Model Dataset Commands

This document describes the set of commands used to generate various datasets required for training and evaluating a world model. These commands take a standard puzzle environment and produce datasets of state transitions, initial/target pairs, and evaluation trajectories.

## Usage

These commands are typically run before `world_model_train train` to prepare the necessary `.npy` files.

```bash
# Example: Create a transition dataset for the Rubik's Cube
python main.py world_model_train make_transition_dataset -p rubikscube

# Example: Create an evaluation trajectory
python main.py world_model_train make_eval_trajectory -p rubikscube
```

## Common Options

All commands in this group share the following option sets, which are used to configure the puzzle environment and the dataset generation parameters.

### Puzzle Options (`@wm_puzzle_ds_options`)

-   `-p, --puzzle`: Specifies the puzzle to use for generating the dataset.
    -   Type: `Choice`
    -   Default: `rubikscube`
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
-   `-h, --hard`: If available, use a "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: Seed for the random puzzle (if supported).
    -   Type: `String`
    -   Default: `0`

### Dataset Generation Options (`@wm_dataset_options`)

-   `--dataset_size`: The total number of samples (e.g., transitions, states) to generate.
    -   Type: `Integer`
    -   Default: `300000`
-   `--dataset_minibatch_size`: The batch size for processing during dataset generation.
    -   Type: `Integer`
    -   Default: `30000`
-   `--shuffle_length`: The number of random moves applied to a solved state to generate varied initial states.
    -   Type: `Integer`
    -   Default: `30`
-   `--img_size`: The `(height, width)` to which the state images will be resized.
    -   Type: `Tuple (int, int)`
    -   Default: `(32, 32)`
-   `--key`: The random seed for dataset generation, ensuring reproducibility.
    -   Type: `Integer`
    -   Default: `0`

---

## Commands

### `world_model_train make_transition_dataset`

This command generates a dataset of state transitions. For each sample, it records a starting state, the action taken, and the resulting next state. This is the primary dataset used for training the world model's transition dynamics.

-   **Outputs:**
    -   `tmp/<puzzle_name>/actions.npy`
    -   `tmp/<puzzle_name>/images.npy` (the starting states)
    -   `tmp/<puzzle_name>/next_images.npy` (the resulting states)

### `world_model_train make_sample_data`

This command generates a dataset of initial and target state pairs. This can be useful for tasks that require a set of problems to solve, such as evaluating a planner.

-   **Outputs:**
    -   `tmp/<puzzle_name>/inits.npy` (initial states)
    -   `tmp/<puzzle_name>/targets.npy` (target states)

### `world_model_train make_eval_trajectory`

This command generates a long trajectory of states and actions by starting from a solved state and applying a large number of random moves. This trajectory is used as a consistent benchmark to evaluate the world model's prediction accuracy over multiple steps during training.

-   **Outputs:**
    -   `tmp/<puzzle_name>/eval_actions.npy`
    -   `tmp/<puzzle_name>/eval_traj_images.npy`
