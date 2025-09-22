<div align="center">
  <img src="images/JAxtar.svg" alt="JAxtar logo" width="400"></img>
</div>

# JA<sup>xtar</sup>: GPU-accelerated Batched parallel A\* & Q\* solver in pure JAX!

JA<sup>xtar</sup> is a project with a JAX-native implementation of parallelizable a A\* & Q\* solver for neural heuristic search research.
This project is inspired by [mctx](https://github.com/google-deepmind/mctx) from Google DeepMind. If MCTS can be implemented entirely in pure JAX, why not A\*?

MCTS, or tree search, is used in many RL algorithmic techniques, starting with AlphaGo, but graph search (not tree search) doesn't seem to have received much attention. Nevertheless, there are puzzle‐solving algorithms that use neural heuristics like [DeepcubeA](https://github.com/forestagostinelli/DeepCubeA) with A\* or [Q\*](https://arxiv.org/abs/2102.04518)(graph search).

However, the most frustrating aspect of [my brief research (MSc)](https://drive.google.com/file/d/1clo8OmuXvIHhJzOUhH__0ZWzAamgVK84/view?usp=drive_link) in this area is the time it takes to pass information back and forth between the GPU and CPU.
When using a neural heuristic to evaluate a single node, the communication between the CPU and GPU, rather than the computation itself, can consume between 50% and 80% of the total processing time. Because of this communication overhead, DeepcubeA batches multiple nodes concurrently, which appears to work quite well.

However, these issues indicate that a more fundamental solution is needed. This led me to search for ways to perform A\* directly on the GPU, but I discovered that most implementations suffer from the following problems.

- Many are written in pure C and CUDA, which is not well-suited for machine learning research.
- Some are written in JAX or PyTorch, but these are often limited to 2D grid environments or connectivity matrices, and cannot scale to an infinite number of different states that cannot all be held in memory.
- The implementation itself is often dependent on the specific definition of the state or problem.

To address these challenges, I decided to develop code based on the following principles:

- Pure JAX implementation
  - Specifically for machine learning research.
- JAX-native priority queue
  - The A\* algorithm necessitates a priority queue to process nodes based on the lowest combined cost and heuristic estimate.
  - However, standard Python heaps use lists, which are not JIT-compilable in JAX. Thus, a JAX-iterable heap is necessary.
- Hashable state representation and a hashtable for JAX operations.
  - This is crucial for tracking node status (open/closed) in A\* and efficiently retrieving parent state information.
  - Hashing is optional for simple, indexable states. But for complex or infinite state spaces, hashing becomes essential for efficient indexing and retrieval of unique states.
- Fully batched and parallelized operations
  - GPUs provide massive parallelism but have slower cores than CPUs. Therefore, algorithms for GPUs must be highly parallelized to leverage their architecture.
- Puzzle-agnostic implementation
  - The implementation should be general enough to handle any puzzle with a defined state and action space.
  - This generality enables wider research and allows for formalizing 'strict' behaviors in future implementations.

This project features specially written components, including:

- [`Xtructure`](https://github.com/tinker495/Xtructure): A pip package providing JAX-compatible hash and priority queue implementations, originally developed as part of this project and later separated. This package includes:
  - a hashtable for parallel lookup and insertion operations
  - a priority queue that supports batching, push, and pop operations
- [`PuXle`](https://github.com/tinker495/PuXle): All puzzle implementations have been moved to this separate high-performance library for parallelized puzzle environments built on JAX
- World model implementations based on PuXle for discrete world model learning and heuristic search
- Network heuristics and Q-functions designed for JIT-compilable integration with A\* & Q\* algorithm
- a fully JIT-compiled A\* & Q\* algorithm for puzzles

This project was quite challenging to develop, and it felt like performing acrobatics with JAX. However, I managed to create a fully functional version, and hopefully it will inspire you to discover something amazing as you delve into JAX.

## Usage and Documentation

For detailed information on all available commands and their options, please refer to the official documentation.

[**Go to Documentation**](./docs/README.md)

Hydra users can check the dedicated guide for the new configuration system introduced during the migration (covers both DAVI and Q-learning training flows):

- [Hydra Workflow](./docs/hydra.md)

## Result

We can find the optimal path using a jittable, batched A\* search as shown below. This is not a super blazingly fast result, but it can be well integrated with heuristics using neural networks.

The following speed benchmarks were measured on an Nvidia RTX 5090 hosted at vast.ai.

You can easily test it yourself with the colab link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TJUazlkm9miP4sIKCXShExaRcT6oGD4K?usp=sharing)

### Test Run

```bash
$ python main.py astar
╭─────────────── Seed 0 ────────────────╮
│    Start State        Target State    │
│ ┏━━━┳━━━┳━━━┳━━━┓   ┏━━━┳━━━┳━━━┳━━━┓ │
│ ┃ 9 ┃ E ┃ 6 ┃ 8 ┃   ┃ 1 ┃ 2 ┃ 3 ┃ 4 ┃ │
│ ┣━━━╋━━━╋━━━╋━━━┫   ┣━━━╋━━━╋━━━╋━━━┫ │
│ ┃ D ┃ 4 ┃ 7 ┃   ┃   ┃ 5 ┃ 6 ┃ 7 ┃ 8 ┃ │
│ ┣━━━╋━━━╋━━━╋━━━┫ → ┣━━━╋━━━╋━━━╋━━━┫ │
│ ┃ B ┃ 1 ┃ A ┃ C ┃   ┃ 9 ┃ A ┃ B ┃ C ┃ │
│ ┣━━━╋━━━╋━━━╋━━━┫   ┣━━━╋━━━╋━━━╋━━━┫ │
│ ┃ 5 ┃ 3 ┃ F ┃ 2 ┃   ┃ D ┃ E ┃ F ┃   ┃ │
│ ┗━━━┻━━━┻━━━┻━━━┛   ┗━━━┻━━━┻━━━┻━━━┛ │
│              Dist: 34.00              │
╰───────────────────────────────────────╯
     Search Result for Seed 0
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric        ┃          Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Status        │ Solution Found │
│ Cost          │           50.0 │
│ Search Time   │         0.12 s │
│ Search States │           738K │
│ States/s      │          6.15M │
└───────────────┴────────────────┘
```

### Test vmapped run

```bash
$ python main.py astar --vmap_size 20
╭──────────────────────────────── Vmapped Search Setup ─────────────────────────────────╮
│ ╭────────────────────────────────── Batched State ──────────────────────────────────╮ │
│ │  ┏━━━┳━━━┳━━━┳━━━┓  ┏━━━┳━━━┳━━━┳━━━┓       ┏━━━┳━━━┳━━━┳━━━┓  ┏━━━┳━━━┳━━━┳━━━┓  │ │
│ │  ┃ 9 ┃ E ┃ 6 ┃ 8 ┃  ┃ 9 ┃ E ┃ 6 ┃ 8 ┃       ┃ 9 ┃ E ┃ 6 ┃ 8 ┃  ┃ 9 ┃ E ┃ 6 ┃ 8 ┃  │ │
│ │  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫       ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  │ │
│ │  ┃ D ┃ 4 ┃ 7 ┃   ┃  ┃ D ┃ 4 ┃ 7 ┃   ┃       ┃ D ┃ 4 ┃ 7 ┃   ┃  ┃ D ┃ 4 ┃ 7 ┃   ┃  │ │
│ │  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  ...  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  │ │
│ │  ┃ B ┃ 1 ┃ A ┃ C ┃  ┃ B ┃ 1 ┃ A ┃ C ┃       ┃ B ┃ 1 ┃ A ┃ C ┃  ┃ B ┃ 1 ┃ A ┃ C ┃  │ │
│ │  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫       ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  │ │
│ │  ┃ 5 ┃ 3 ┃ F ┃ 2 ┃  ┃ 5 ┃ 3 ┃ F ┃ 2 ┃       ┃ 5 ┃ 3 ┃ F ┃ 2 ┃  ┃ 5 ┃ 3 ┃ F ┃ 2 ┃  │ │
│ │  ┗━━━┻━━━┻━━━┻━━━┛  ┗━━━┻━━━┻━━━┻━━━┛       ┗━━━┻━━━┻━━━┻━━━┛  ┗━━━┻━━━┻━━━┻━━━┛  │ │
│ ╰────────────────────────────────── shape: (20,) ───────────────────────────────────╯ │
│                                                                                       │
│                                           ↓                                           │
│                                                                                       │
│ ╭─────────────────────────────── Batched SolveConfig ───────────────────────────────╮ │
│ │  ┏━━━┳━━━┳━━━┳━━━┓  ┏━━━┳━━━┳━━━┳━━━┓       ┏━━━┳━━━┳━━━┳━━━┓  ┏━━━┳━━━┳━━━┳━━━┓  │ │
│ │  ┃ 1 ┃ 2 ┃ 3 ┃ 4 ┃  ┃ 1 ┃ 2 ┃ 3 ┃ 4 ┃       ┃ 1 ┃ 2 ┃ 3 ┃ 4 ┃  ┃ 1 ┃ 2 ┃ 3 ┃ 4 ┃  │ │
│ │  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫       ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  │ │
│ │  ┃ 5 ┃ 6 ┃ 7 ┃ 8 ┃  ┃ 5 ┃ 6 ┃ 7 ┃ 8 ┃       ┃ 5 ┃ 6 ┃ 7 ┃ 8 ┃  ┃ 5 ┃ 6 ┃ 7 ┃ 8 ┃  │ │
│ │  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  ...  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  │ │
│ │  ┃ 9 ┃ A ┃ B ┃ C ┃  ┃ 9 ┃ A ┃ B ┃ C ┃       ┃ 9 ┃ A ┃ B ┃ C ┃  ┃ 9 ┃ A ┃ B ┃ C ┃  │ │
│ │  ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫       ┣━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━┫  │ │
│ │  ┃ D ┃ E ┃ F ┃   ┃  ┃ D ┃ E ┃ F ┃   ┃       ┃ D ┃ E ┃ F ┃   ┃  ┃ D ┃ E ┃ F ┃   ┃  │ │
│ │  ┗━━━┻━━━┻━━━┻━━━┛  ┗━━━┻━━━┻━━━┻━━━┛       ┗━━━┻━━━┻━━━┻━━━┛  ┗━━━┻━━━┻━━━┻━━━┛  │ │
│ ╰────────────────────────────────── shape: (20,) ───────────────────────────────────╯ │
│                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────╯
           Vmapped Search Result
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric          ┃                  Value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Search Time     │       3.18s (x26.5/20) │
│ Search States   │ 14.8M (4.64M states/s) │
│ Speedup         │                   x0.8 │
│ Solutions Found │                100.00% │
└─────────────────┴────────────────────────┘
```

### A\* with neural heuristic model

```bash
$ python main.py astar -nn -h -p rubikscube -w 0.2

...

     Search Result for Seed 0
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric        ┃          Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Status        │ Solution Found │
│ Cost          │           22.0 │
│ Search Time   │         0.49 s │
│ Search States │          1.85M │
│ States/s      │          3.78M │
└───────────────┴────────────────┘
```

### Q\* with neural Q model

```bash
$ python main.py qstar -nn -h -p rubikscube -w 0.2

...

     Search Result for Seed 0
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric        ┃          Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Status        │ Solution Found │
│ Cost          │           22.0 │
│ Search Time   │         0.16 s │
│ Search States │          1.79M │
│ States/s      │          10.9M │
└───────────────┴────────────────┘
```

### World Model Puzzle with A\* & Q\*

```bash
$ python main.py qstar -p rubikscube_world_model_optimized -nn -w 0.6

...

     Search Result for Seed 0
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric        ┃          Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Status        │ Solution Found │
│ Cost          │           22.0 │
│ Search Time   │         0.17 s │
│ Search States │          1.82M │
│ States/s      │          10.5M │
└───────────────┴────────────────┘
```

## Puzzles

### Target available puzzle

| Rubikscube                                              | Slidepuzzle                                               | Lightsout                                             | Sokoban                                          |
| ------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------ |
| ![Rubiks cube solve](images/rubikscubesolve.png)        | ![Slide puzzle solve](images/slidepuzzlesolve.png)        | ![Lights out solve](images/lightsoutsolve.png)        | ![Sokoban solve](images/sokobansolve.png)        |
| ![Rubiks cube animate](images/rubikscube_animation.gif) | ![Slide puzzle animate](images/slidepuzzle_animation.gif) | ![Lights out animate](images/lightsout_animation.gif) | ![Sokoban animate](images/sokoban_animation.gif) |


| Maze                                       | Pancake Sorting                                          | Hanoi                                        |  TopSpin |
| ------------------------------------------ | -------------------------------------------------------- | -------------------------------------------- | ------------------------------ |
| ![Maze solve](images/mazesolve.png)        | ![Pancake Sorting solve](images/pancakesolve.png)        | ![Hanoi solve](images/hanoisolve.png)        | ![TopSpin Solve](images/topspinsolve.png) |
| ![Maze animate](images/maze_animation.gif) | ![Pancake Sorting animate](images/pancake_animation.gif) | ![Hanoi animate](images/hanoi_animation.gif) | ![TopSpin Solve](images/topsplin_animation.gif) |

### Target not available puzzle

These types of puzzles are not strictly the kind that are typically solved with A\*, but after some simple testing, it turns out that, depending on how the problem is defined, they can be solved. Furthermore, this approach can be extended to TSP and countless other COP problems, provided that with a good heuristic. The training method will need to be investigated further.

| Dotknot                                          | TSP                                      |
| ------------------------------------------------ | ---------------------------------------- |
| ![dotknot solve](images/dotknotsolve.png)        | ![tsp solve](images/tspsolve.png)        |
| ![dotknot animate](images/dotknot_animation.gif) | ![tsp animate](images/tsp_animation.gif) |

### World Model Puzzle

This is an implementation of learning a world model, as introduced in the paper ["Learning Discrete World Models for Heuristic Search"](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_225.pdf), and performing A\* & Q\* search within that world model.

<!-- Currently, this implementation achieves node search speeds that are basically more than 10 times faster than those presented in the paper. -->

| Terminal View                                | Rubiks Cube                                                |   Rubiks Cube Reversed | Sokoban                                             |
| -------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------- |
| ![terminal view](images/worldmodelsolve.png) | ![rubiks cube](images/worldmodel_rubikscube_animation.gif) | ![rubiks cube rev](images/worldmodel_rubikscube_rev_animation.gif) |![sokoban](images/worldmodel_sokoban_animation.gif) |

## Citation

Please use this citation to reference this project.

```bibtex
@software{kyuseokjung2024jaxtar,
  title = {JA^{xtar}: GPU-accelerated Batched parallel A* & Q* solver in pure JAX!},
  author = {Kyuseok Jung},
  url = {https://github.com/tinker495/JAxtar},
  year = {2024},
}
```
