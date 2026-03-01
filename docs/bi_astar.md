# Bidirectional A\* Command (`bi_astar`)

The `bi_astar` command solves a puzzle using the Bidirectional A\* search algorithm. This algorithm runs two simultaneous A\* searches: one forward from the initial state and one backward from the goal state. It stops when the two searches meet, often significantly reducing the number of nodes explored compared to standard A\*.

## Usage

The basic syntax for the `bi_astar` command is:

```bash
python main.py bi-astar [OPTIONS]
```

Example:

```bash
python main.py bi-astar -p rubikscube -nn
```

## Options

The `bi_astar` command uses similar option groups to the standard `astar` command.

### Puzzle Options (`@puzzle_options`)

These options define the puzzle environment to be solved.

-   `-p, --puzzle`: Specifies the puzzle to solve.
    -   Type: `Choice`
    -   Default: `n-puzzle`
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
-   `-h, --hard`: If available, use a "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: A comma-separated list of seeds for generating initial puzzle states.
    -   Type: `String`
    -   Default: `"0"`

### Search Options (`@search_options`)

These options control the behavior of the search algorithm.

-   `-m, --max_node_size`: The maximum number of nodes to explore.
    -   Type: `String`
-   `-b, --batch_size`: The number of nodes to process in a single batch.
    -   Type: `Integer`
-   `-w, --cost_weight`: The weight `w` for the path cost.
    -   Type: `Float`
-   `-pr, --pop_ratio`: Ratio for popping nodes from the priority queue.
    -   Type: `Float`
-   `-vm, --vmap_size`: The number of different initial states to solve in parallel.
    -   Type: `Integer`
-   `--debug`: Disables JIT compilation.
    -   Type: `Flag`
-   `--profile`: Enables profiler.
    -   Type: `Flag`
-   `--show_compile_time`: Prints compilation time.
    -   Type: `Flag`
-   `--search-preset`: Apply puzzle-specific search defaults.
    -   Type: `String`

### Heuristic Options (`@heuristic_options`)

-   `-nn, --neural_heuristic`: Use a pre-trained neural network as the heuristic function.
    -   Type: `Flag`
-   `--param-path`: Path to the heuristic parameter file.
    -   Type: `String`
-   `--model-type`: Type of the heuristic model.
    -   Type: `String`
-   `-q, --use-quantize`: Enable quantized neural inference.
    -   Type: `Flag`
-   `--quant-type`: Quantization preset (`int8`, `int4`, `int4_w8a`, `int8_w_only`).
    -   Type: `Choice`

### Visualization Options (`@visualize_options`)

-   `-vt, --visualize_terminal`: Renders the solution path in the terminal.
    -   Type: `Flag`
-   `-vi, --visualize_imgs`: Generates images and GIF for the solution.
    -   Type: `Flag`
-   `-mt, --max_animation_time`: Max duration for GIF.
    -   Type: `Integer`

## Related Commands

```bash
python main.py eval bi-astar [OPTIONS]
python main.py benchmark bi-astar [OPTIONS]
```

Training-time evaluation can also use this algorithm via:

```bash
python main.py distance-train heuristic --eval-search-metric bi_astar
```

## Low-memory benchmark fallback profile

If default benchmark settings trigger OOM on your machine, start with:

```bash
python main.py benchmark bi-astar \
  --benchmark rubikscube-deepcubea \
  --num-eval 1 \
  --batch-size 16 \
  --max-node-size 4096 \
  --param-path heuristic/neuralheuristic/model/params/rubikscube_3_v2.pkl
```
