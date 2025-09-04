import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import jax
import numpy as np
import pandas as pd
import xtructure.numpy as xnp
from puxle import Puzzle
from rich.console import Console

from config.pydantic_models import EvalOptions, PuzzleOptions
from helpers import human_format
from helpers.artifact_manager import ArtifactManager
from helpers.config_printer import print_config
from helpers.logger import BaseLogger
from helpers.metrics import calculate_heuristic_metrics
from helpers.plots import (
    plot_expansion_distribution,
    plot_heuristic_accuracy,
    plot_nodes_generated_by_path_cost,
    plot_path_cost_distribution,
    plot_search_time_by_path_cost,
)
from helpers.rich_progress import trange
from helpers.summaries import create_summary_panel
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction

from .comparison_generator import ComparisonGenerator


class EvaluationRunner:
    def __init__(
        self,
        puzzle: Puzzle,
        puzzle_name: str,
        search_model: Union[Heuristic, QFunction],
        search_model_name: str,
        search_builder_fn: Callable,
        eval_options: EvalOptions,
        puzzle_opts: PuzzleOptions,
        output_dir: Optional[Path] = None,
        logger: Optional[BaseLogger] = None,
        step: int = 0,
        **kwargs,
    ):
        self.puzzle = puzzle
        self.puzzle_name = puzzle_name
        self.search_model = search_model
        self.search_model_name = search_model_name
        self.search_builder_fn = search_builder_fn
        self.eval_options = eval_options
        self.puzzle_opts = puzzle_opts
        self.output_dir = output_dir
        self.logger = logger
        self.step = step
        self.console = Console()
        self.kwargs = kwargs

    def run(self):
        model_metadata = getattr(self.search_model, "metadata", {})

        pop_ratios = (
            self.eval_options.pop_ratio
            if isinstance(self.eval_options.pop_ratio, list)
            else [self.eval_options.pop_ratio]
        )
        cost_weights = (
            self.eval_options.cost_weight
            if isinstance(self.eval_options.cost_weight, list)
            else [self.eval_options.cost_weight]
        )
        batch_sizes = (
            self.eval_options.batch_size
            if isinstance(self.eval_options.batch_size, list)
            else [self.eval_options.batch_size]
        )

        param_combinations = list(itertools.product(pop_ratios, cost_weights, batch_sizes))
        is_sweep = len(param_combinations) > 1

        base_run_name = (
            self.eval_options.run_name
            if self.eval_options.run_name
            else f"{self.puzzle_name}_{self.search_model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        main_run_dir = self.output_dir if self.output_dir else Path("runs") / base_run_name
        main_run_dir.mkdir(parents=True, exist_ok=True)

        if is_sweep:
            self.console.print(
                f"Starting parameter sweep with {len(param_combinations)} combinations."
                f"Results will be in [bold]{main_run_dir}[/bold]"
            )

        sub_run_dirs = []
        for i, (pr, cw, bs) in enumerate(param_combinations):
            run_dir = main_run_dir
            run_name = f"pr_{pr}_cw_{cw}_bs_{bs}".replace("inf", "Infinity")
            if is_sweep:
                run_dir = main_run_dir / run_name
                sub_run_dirs.append(str(run_dir))

            am = ArtifactManager(run_dir, self.logger, self.step)

            current_eval_opts = self.eval_options.copy(
                update={"pop_ratio": pr, "cost_weight": cw, "batch_size": bs}
            )

            config = {
                "puzzle_options": self.puzzle_opts.dict(),
                self.search_model_name: self.search_model.__class__.__name__,
                f"{self.search_model_name}_metadata": model_metadata,
                "eval_options": current_eval_opts.dict(),
            }

            if is_sweep:
                self.console.rule(
                    f"[bold cyan]Run {i+1}/{len(param_combinations)}: pr={pr}, cw={cw}, bs={bs}[/bold cyan]"
                )
            print_config(f"{self.search_model_name.capitalize()} Evaluation Configuration", config)

            search_fn = self.search_builder_fn(
                self.puzzle,
                self.search_model,
                bs,
                self.eval_options.get_max_node_size(bs),
                pop_ratio=pr,
                cost_weight=cw,
            )

            eval_seeds = list(range(self.eval_options.num_eval))
            results = self._run_evaluation(
                search_fn=search_fn,
                puzzle=self.puzzle,
                seeds=eval_seeds,
            )
            for r in results:
                r["pop_ratio"] = pr
                r["cost_weight"] = cw
                r["batch_size"] = bs

            # Calculate and store heuristic metrics
            heuristic_metrics = calculate_heuristic_metrics(results)
            if heuristic_metrics:
                config["heuristic_metrics"] = heuristic_metrics

            am.save_config(config)
            am.save_results(results)
            am.save_path_states(results)

            solved_df = pd.DataFrame([r for r in results if r["solved"]])
            # Log final success rate based on results
            total = len(results)
            final_success_rate = (len(solved_df) / total) * 100 if total > 0 else 0.0
            am.log_scalar("success_rate", final_success_rate)
            if not is_sweep:
                self.console.print(create_summary_panel(results, heuristic_metrics))

            if not solved_df.empty:
                fig = plot_path_cost_distribution(solved_df)
                am.save_and_log_plot("path_cost_distribution", fig)

                fig = plot_search_time_by_path_cost(solved_df)
                am.save_and_log_plot("search_time_by_path_cost", fig)

                fig = plot_nodes_generated_by_path_cost(solved_df)
                am.save_and_log_plot("nodes_generated_by_path_cost", fig)

                am.log_scalar("time_to_solve", solved_df["search_time_s"].mean())
                am.log_scalar("nodes_generated", solved_df["nodes_generated"].mean())
                am.log_scalar("path_cost", solved_df["path_cost"].mean())

            if heuristic_metrics:
                am.log_scalar("heuristic_r_squared", heuristic_metrics["r_squared"])
                am.log_scalar("heuristic_ccc", heuristic_metrics["ccc"])

            fig = plot_heuristic_accuracy(results, metrics=heuristic_metrics)
            am.save_and_log_plot("heuristic_accuracy", fig)

            plots_generated = 0
            for r in results:
                if plots_generated >= current_eval_opts.max_expansion_plots:
                    break
                if r.get("expansion_analysis"):
                    fig = plot_expansion_distribution(
                        [r], scatter_max_points=current_eval_opts.scatter_max_points
                    )
                    am.save_and_log_plot(
                        f"expansion_dist_seed_{r['seed']}", fig, sub_dir="expansion_plots"
                    )
                    plots_generated += 1

        if is_sweep:
            self.console.rule(
                "[bold green]Sweep Complete. Generating Comparison Report.[/bold green]"
            )
            comparison_generator = ComparisonGenerator(
                run_dirs=sub_run_dirs,
                output_dir=main_run_dir,
                scatter_max_points=self.eval_options.scatter_max_points,
                logger=self.logger,
                step=self.step,
            )
            comparison_generator.generate_report()
            self.console.print(f"Comparison report saved in [bold]{main_run_dir}[/bold]")

    def _run_evaluation(
        self,
        search_fn,
        puzzle: Puzzle,
        seeds: list[int],
    ) -> list[dict]:
        num_puzzles = len(seeds)
        results = []

        pbar = trange(
            num_puzzles,
            desc="Running Evaluations",
        )

        for i in pbar:
            seed = seeds[i]
            solve_config, state = puzzle.get_inits(jax.random.PRNGKey(seed))

            start_time = time.time()
            search_result = search_fn(solve_config, state)
            solved = bool(search_result.solved.block_until_ready())
            end_time = time.time()

            search_time = end_time - start_time
            generated_nodes = int(search_result.generated_size)

            result_item = {
                "seed": seed,
                "solved": solved,
                "search_time_s": search_time,
                "nodes_generated": generated_nodes,
                "path_cost": 0,
                "path_analysis": None,
                "expansion_analysis": None,
            }

            if solved:
                path = search_result.get_solved_path()
                path_cost = search_result.get_cost(path[-1])
                result_item["path_cost"] = float(path_cost)

                states = []
                actual_dists = []
                estimated_dists = []
                for state_in_path in path:
                    states.append(search_result.get_state(state_in_path))
                    actual_dist = float(path_cost - search_result.get_cost(state_in_path))
                    estimated_dist = float(search_result.get_dist(state_in_path))

                    if np.isfinite(estimated_dist):
                        actual_dists.append(actual_dist)
                        estimated_dists.append(estimated_dist)

                result_item["path_analysis"] = {
                    "actual": actual_dists,
                    "estimated": estimated_dists,
                    "states": xnp.concatenate(states),
                }

            # Extract expansion data for plotting node value distributions
            expanded_nodes_mask = search_result.pop_generation > -1
            # Use np.asarray to handle potential JAX arrays on different devices
            if np.any(np.asarray(expanded_nodes_mask)):
                pop_generations = np.asarray(search_result.pop_generation[expanded_nodes_mask])
                costs = np.asarray(search_result.cost[expanded_nodes_mask])
                dists = np.asarray(search_result.dist[expanded_nodes_mask])

                if pop_generations.size > 0:
                    result_item["expansion_analysis"] = {
                        "pop_generation": pop_generations,
                        "cost": costs,
                        "dist": dists,
                    }

            results.append(result_item)

            solved_results = [r for r in results if r["solved"]]
            num_solved = len(solved_results)
            success_rate = (num_solved / (i + 1)) * 100

            pbar_desc_dict = {"Success Rate": f"{success_rate:.2f}%"}
            if num_solved > 0:
                avg_time = sum(r["search_time_s"] for r in solved_results) / num_solved
                avg_nodes = sum(r["nodes_generated"] for r in solved_results) / num_solved
                pbar_desc_dict.update(
                    {
                        "Avg Time (Solved)": f"{avg_time:.2f}s",
                        "Avg Nodes (Solved)": f"{human_format(avg_nodes)}",
                    }
                )
            pbar.set_description("Evaluating", desc_dict=pbar_desc_dict)

            # Early stopping logic
            if (
                self.eval_options.use_early_stopping
                and (i + 1) >= self.eval_options.early_stop_patience
            ):
                current_success_rate = num_solved / (i + 1)
                if current_success_rate < self.eval_options.early_stop_threshold:
                    self.console.print(
                        f"[bold yellow]Early stopping triggered![/bold yellow] "
                        f"Success rate ({current_success_rate:.2%}) below threshold "
                        f"({self.eval_options.early_stop_threshold:.2%}) after {i + 1} samples."
                    )
                    break

        return results
