# Bidirectional Q\* Command (`bi_qstar`)

The `bi_qstar` command solves a puzzle using the Bidirectional Q\* search algorithm. It performs bidirectional search using Q-values to guide the expansion in both forward and backward directions. This allows for more informed search steps compared to standard heuristic-based search.

## Usage

The basic syntax for the `bi_qstar` command is:

```bash
python main.py bi-qstar [OPTIONS]
```

Example:

```bash
python main.py bi-qstar -p rubikscube -nn
```

## Options

The `bi_qstar` command uses similar option groups to the `qstar` command.

### Puzzle Options (`@puzzle_options`)

-   `-p, --puzzle`: Specifies the puzzle to solve.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle arguments.
-   `-h, --hard`: Use a hard version of the puzzle.
-   `-s, --seeds`: Comma-separated list of seeds.

### Search Options (`@search_options`)

-   `-m, --max_node_size`: Max nodes to explore.
-   `-b, --batch_size`: Batch size.
-   `-w, --cost_weight`: Path cost weight.
-   `-pr, --pop_ratio`: Pop ratio.
-   `-vm, --vmap_size`: Parallel solve size via vmap.
-   `--debug`: Disable JIT.
-   `--profile`: Enable profiling.
-   `--show_compile_time`: Print compile time.
-   `--search-preset`: Apply puzzle-specific search defaults.

### Q-Function Options (`@qfunction_options`)

-   `-nn, --neural_qfunction`: Use neural network Q-function.
-   `--param-path`: Path to Q-function parameters.
-   `--model-type`: Q-function model type.
-   `-q, --use-quantize`: Enable quantized neural inference.
-   `--quant-type`: Quantization preset (`int8`, `int4`, `int4_w8a`, `int8_w_only`).

### Visualization Options (`@visualize_options`)

-   `-vt, --visualize_terminal`: Render path in terminal.
-   `-vi, --visualize_imgs`: Generate images/GIF.
-   `-mt, --max_animation_time`: Max GIF duration.

## Related Commands

```bash
python main.py eval bi-qstar [OPTIONS]
python main.py benchmark bi-qstar [OPTIONS]
```

Training-time evaluation can also use this algorithm via:

```bash
python main.py distance-train qfunction --eval-search-metric bi_qstar
```

## Low-memory benchmark fallback profile

If default benchmark settings trigger OOM on your machine, start with:

```bash
python main.py benchmark bi-qstar \
  --benchmark rubikscube-deepcubea \
  --num-eval 1 \
  --batch-size 16 \
  --max-node-size 4096 \
  --param-path qfunction/neuralq/model/params/rubikscube_3_v2.pkl \
  --use-quantize \
  --quant-type int8
```
