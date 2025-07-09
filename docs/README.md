# JAxtar Documentation

Welcome to the official documentation for `JAxtar`. This documentation provides detailed information about the available commands and their options.

## Commands

`JAxtar` is operated through a command-line interface. Below is a list of the main commands available.

### Search Commands

These commands are used to solve puzzles using different search algorithms.

-   [**`astar`**](./astar.md): Solves a puzzle using the A\* search algorithm.
-   [**`qstar`**](./qstar.md): Solves a puzzle using the Q\* search algorithm, guided by a Q-function.

### Interactive Commands

-   [**`human_play`**](./human_play.md): Allows you to play a puzzle interactively in the terminal.

### Training Commands

These commands are used to train neural network models for heuristic search.

-   [**`davi`**](./davi_train.md): Trains a neural network to act as a heuristic function by predicting the distance to the goal.
-   [**`qlearning`**](./qlearning_train.md): Trains a neural network to serve as a Q-function for estimating action costs.
-   [**`world_model_train`**](./world_model_train.md): Trains a discrete world model that learns the puzzle's transition dynamics.
-   [**`world_model_dataset`**](./world_model_dataset.md): A set of commands to generate datasets for training world models.

### Evaluation Commands

These commands are used to evaluate trained models and search algorithms.

-   [**`eval heuristic`**](./eval_heuristic.md): Evaluates a heuristic on a set of puzzles.
-   [**`eval qlearning`**](./eval_qlearning.md): Evaluates a Q-function on a set of puzzles.
