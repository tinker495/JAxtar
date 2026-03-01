# Bidirectional A\* Deferred Command (`bi_astar_d`)

The `bi_astar_d` command solves a puzzle using the Bidirectional A\* Deferred search algorithm. This combines bidirectional search (forward and backward) with deferred node expansion. It is useful for reducing the search space in complex problems where node expansion is costly.

## Search Correctness Notes

- Both directions use the same action-major batch insertion helper before deferred popping.
- Deferred pop selection is based on the merged final key batch, so `pop_ratio` and `min_pop` apply to the actual post-merge frontier.
- Path reconstruction diagnostics are standardized via `PATH_RECONSTRUCTION_DIAGNOSTIC` payloads.

## Usage

The basic syntax for the `bi_astar_d` command is:

```bash
python main.py bi-astar-d [OPTIONS]
```

Example:

```bash
python main.py bi-astar-d -p rubikscube -nn
```

## Options

The `bi_astar_d` command uses the same option groups as `bi_astar`.

### Puzzle Options (`@puzzle_options`)

-   `-p, --puzzle`: Specifies the puzzle to solve.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle arguments.
-   `-h, --hard`: Use a hard version of the puzzle.
-   `-s, --seeds`: Comma-separated list of seeds.

### Search Options (`@search_options`)

-   `-m, --max_node_size`: Max nodes to explore.
-   `-b, --batch_size`: Batch size for GPU processing.
-   `-w, --cost_weight`: Path cost weight.
-   `-pr, --pop_ratio`: Pop ratio.
-   `-vm, --vmap_size`: Parallel solve size via vmap.
-   `--debug`: Disable JIT.
-   `--profile`: Enable profiling.
-   `--show_compile_time`: Print compile time.
-   `--search-preset`: Apply puzzle-specific search defaults.

### Heuristic Options (`@heuristic_options`)

-   `-nn, --neural_heuristic`: Use neural network heuristic.
-   `--param-path`: Path to heuristic parameters.
-   `--model-type`: Heuristic model type.
-   `-q, --use-quantize`: Enable quantized neural inference.
-   `--quant-type`: Quantization preset (`int8`, `int4`, `int4_w8a`, `int8_w_only`).

### Visualization Options (`@visualize_options`)

-   `-vt, --visualize_terminal`: Render path in terminal.
-   `-vi, --visualize_imgs`: Generate images/GIF.
-   `-mt, --max_animation_time`: Max GIF duration.

## Related Commands

```bash
python main.py eval bi-astar-d [OPTIONS]
python main.py benchmark bi-astar-d [OPTIONS]
```

Training-time evaluation can also use this algorithm via:

```bash
python main.py distance-train heuristic --eval-search-metric bi_astar_d
```

## Low-memory benchmark fallback profile

If default benchmark settings trigger OOM on your machine, start with:

```bash
python main.py benchmark bi-astar-d \
  --benchmark rubikscube-deepcubea \
  --num-eval 1 \
  --batch-size 16 \
  --max-node-size 4096 \
  --param-path heuristic/neuralheuristic/model/params/rubikscube_3_v2.pkl
```
