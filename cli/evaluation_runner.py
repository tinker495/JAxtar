import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import jax
import numpy as np
import pandas as pd
import xtructure.numpy as xnp
from puxle import Puzzle
from puxle.benchmark import Benchmark, BenchmarkSample
from rich.console import Console

from config.pydantic_models import EvalOptions, PuzzleOptions
from helpers import human_format
from helpers.artifact_manager import ArtifactManager
from helpers.config_printer import print_config
from helpers.logger import BaseLogger
from helpers.metrics import calculate_heuristic_metrics
from helpers.plots import (
    plot_benchmark_path_comparison,
    plot_expansion_distribution,
    plot_heuristic_accuracy,
    plot_nodes_generated_by_path_cost,
    plot_path_cost_distribution,
    plot_search_time_by_path_cost,
)
from helpers.rich_progress import trange
from helpers.summaries import create_summary_panel
from helpers.visualization import (
    build_path_steps_from_actions,
    build_path_steps_from_nodes,
)
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
        node_metric_label: Optional[str] = None,
        run_label: Optional[str] = None,
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
        self.node_metric_label = node_metric_label or "Nodes Generated"
        self.kwargs = kwargs
        self.run_label = run_label or search_model_name
        self.benchmark: Optional[Benchmark] = kwargs.get("benchmark")
        self.benchmark_name: Optional[str] = kwargs.get("benchmark_name")
        benchmark_cli_options = kwargs.get("benchmark_cli_options", {})
        self.benchmark_sample_ids: Optional[Iterable] = benchmark_cli_options.get("sample_ids")
        self.benchmark_sample_limit: Optional[int] = benchmark_cli_options.get("sample_limit")
        self._benchmark_total_samples: Optional[int] = None
        self._selected_benchmark_ids: Optional[list] = None

        self.base_run_name = (
            self.eval_options.run_name
            if self.eval_options.run_name
            else f"{self.puzzle_name}_{self.run_label}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        self.main_run_dir = (
            self.output_dir if self.output_dir else Path("runs") / self.base_run_name
        )
        self.main_run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.main_run_dir / "console.log"

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

        eval_inputs = self._prepare_eval_inputs()

        main_run_dir = self.main_run_dir
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

            am = ArtifactManager(run_dir, self.logger, self.step, log_namespace=run_name)

            current_eval_opts = self.eval_options.copy(
                update={"pop_ratio": pr, "cost_weight": cw, "batch_size": bs}
            )

            config = {
                "puzzle_options": self.puzzle_opts.dict(),
                self.search_model_name: self.search_model.__class__.__name__,
                f"{self.search_model_name}_metadata": model_metadata,
                "eval_options": current_eval_opts.dict(),
                "node_metric_label": self.node_metric_label,
            }

            if self.benchmark is not None:
                config["benchmark"] = {
                    "name": self.benchmark_name,
                    "total_available_samples": self._benchmark_total_samples,
                    "selected_sample_ids": list(self._selected_benchmark_ids or []),
                    "sample_limit": self.benchmark_sample_limit,
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

            results = self._run_evaluation(
                search_fn=search_fn,
                puzzle=self.puzzle,
                eval_inputs=eval_inputs,
            )
            for r in results:
                r["pop_ratio"] = pr
                r["cost_weight"] = cw
                r["batch_size"] = bs

            # Calculate and store heuristic metrics
            heuristic_metrics = calculate_heuristic_metrics(results)
            if heuristic_metrics:
                config["heuristic_metrics"] = heuristic_metrics

            if self.benchmark is not None:
                benchmark_metrics = {}

                solved_with_opt_cost = [
                    r
                    for r in results
                    if r.get("solved")
                    and r.get("benchmark_optimal_path_cost") is not None
                    and r.get("path_cost") is not None
                ]
                if solved_with_opt_cost:
                    avg_optimal = float(
                        sum(r["benchmark_optimal_path_cost"] for r in solved_with_opt_cost)
                        / len(solved_with_opt_cost)
                    )
                    path_costs = [r["path_cost"] for r in solved_with_opt_cost]
                    avg_path_cost = float(sum(path_costs) / len(path_costs)) if path_costs else None
                    cost_gap = avg_path_cost - avg_optimal if avg_path_cost is not None else None
                    benchmark_metrics.update(
                        {
                            "avg_optimal_cost": avg_optimal,
                            "avg_path_cost": avg_path_cost,
                            "avg_cost_gap": cost_gap,
                            "solved_with_optimal_cost": len(solved_with_opt_cost),
                        }
                    )
                    am.log_scalar("benchmark/avg_optimal_cost", avg_optimal)
                    if avg_path_cost is not None:
                        am.log_scalar("benchmark/avg_path_cost", avg_path_cost)
                    if cost_gap is not None:
                        am.log_scalar("benchmark/avg_cost_gap", cost_gap)

                solved_with_opt_length = [
                    r
                    for r in results
                    if r.get("solved")
                    and r.get("benchmark_optimal_action_count") is not None
                    and r.get("path_action_count") is not None
                ]
                if solved_with_opt_length:
                    avg_opt_actions = float(
                        sum(r["benchmark_optimal_action_count"] for r in solved_with_opt_length)
                        / len(solved_with_opt_length)
                    )
                    avg_path_actions = float(
                        sum(r["path_action_count"] for r in solved_with_opt_length)
                        / len(solved_with_opt_length)
                    )
                    action_gap = avg_path_actions - avg_opt_actions
                    benchmark_metrics.update(
                        {
                            "avg_optimal_actions": avg_opt_actions,
                            "avg_path_actions": avg_path_actions,
                            "avg_action_gap": action_gap,
                            "solved_with_optimal_length": len(solved_with_opt_length),
                        }
                    )
                    am.log_scalar("benchmark/avg_optimal_actions", avg_opt_actions)
                    am.log_scalar("benchmark/avg_path_actions", avg_path_actions)
                    am.log_scalar("benchmark/avg_action_gap", action_gap)

                    matches = [
                        r["matches_optimal_path"]
                        for r in solved_with_opt_length
                        if r.get("matches_optimal_path") is not None
                    ]
                    if matches:
                        exact_matches = sum(1 for m in matches if m)
                        match_rate = exact_matches / len(matches)
                        benchmark_metrics.update(
                            {
                                "exact_optimal_path_rate": match_rate,
                                "exact_optimal_path_count": exact_matches,
                            }
                        )
                        am.log_scalar("benchmark/exact_optimal_path_rate", match_rate)

                if benchmark_metrics:
                    config["benchmark_metrics"] = benchmark_metrics

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

                has_benchmark_cost = (
                    "benchmark_optimal_path_cost" in solved_df
                    and solved_df["benchmark_optimal_path_cost"].notna().any()
                )
                has_benchmark_length = (
                    "benchmark_optimal_action_count" in solved_df
                    and solved_df["benchmark_optimal_action_count"].notna().any()
                )
                if self.benchmark is not None and (has_benchmark_cost or has_benchmark_length):
                    fig = plot_benchmark_path_comparison(solved_df)
                    am.save_and_log_plot("benchmark_path_comparison", fig)

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

    def _prepare_eval_inputs(self) -> List[int]:
        if self.benchmark is None:
            limit = self.eval_options.num_eval
            if limit < 0:
                raise ValueError("num_eval must be non-negative when no benchmark is provided")
            return list(range(limit))

        available_ids = list(self.benchmark.sample_ids())
        if not available_ids:
            raise ValueError(f"Benchmark '{self.benchmark_name}' provided no sample ids.")

        self._benchmark_total_samples = len(available_ids)

        if self.benchmark_sample_ids is not None:
            selected = [
                sample_id for sample_id in self.benchmark_sample_ids if sample_id in available_ids
            ]
            if not selected:
                raise ValueError(
                    "None of the provided sample IDs were found in the benchmark dataset."
                )
        else:
            limit = self.benchmark_sample_limit
            if limit is None or limit < 0:
                limit = self.eval_options.num_eval

            if limit is None or limit < 0:
                selected = available_ids
            else:
                selected = available_ids[: min(limit, len(available_ids))]

        self._selected_benchmark_ids = selected
        return selected

    def _run_evaluation(
        self,
        search_fn,
        puzzle: Puzzle,
        eval_inputs: List[int],
    ) -> list[dict]:
        num_puzzles = len(eval_inputs)
        results = []

        pbar = trange(
            num_puzzles,
            desc="Running Evaluations",
        )

        heuristic_model = self.search_model if isinstance(self.search_model, Heuristic) else None
        qfunction_model = self.search_model if isinstance(self.search_model, QFunction) else None

        for i in pbar:
            identifier = eval_inputs[i]

            if self.benchmark is not None:
                sample_id = identifier
                benchmark_sample: BenchmarkSample = self.benchmark.get_sample(sample_id)
                solve_config = benchmark_sample.solve_config
                state = benchmark_sample.state
                run_identifier = sample_id
            else:
                seed = int(identifier)
                solve_config, state = puzzle.get_inits(jax.random.PRNGKey(seed))
                run_identifier = seed

            start_time = time.time()
            search_result = search_fn(solve_config, state)
            solved = bool(search_result.solved.block_until_ready())
            end_time = time.time()

            search_time = end_time - start_time
            generated_nodes = int(search_result.generated_size)

            result_item = {
                "seed": run_identifier,
                "solved": solved,
                "search_time_s": search_time,
                "nodes_generated": generated_nodes,
                "node_metric_label": self.node_metric_label,
                "path_cost": 0,
                "path_analysis": None,
                "expansion_analysis": None,
                "path_state_count": None,
                "path_action_count": None,
                "matches_optimal_path": None,
                "path_actions": None,
                "path_action_strings": None,
            }

            if self.benchmark is not None:
                result_item["benchmark_sample_id"] = run_identifier
                benchmark_sample = self.benchmark.get_sample(run_identifier)
                optimal_path_cost = getattr(benchmark_sample, "optimal_path_cost", None)
                if optimal_path_cost is None:
                    optimal_path_cost = getattr(benchmark_sample, "optimal_path_costs", None)
                if optimal_path_cost is not None:
                    result_item["benchmark_optimal_path_cost"] = float(optimal_path_cost)
                optimal_path = getattr(benchmark_sample, "optimal_path", None)
                if optimal_path is not None:
                    optimal_state_count = len(optimal_path)
                    result_item["benchmark_optimal_path_state_count"] = optimal_state_count
                    if optimal_state_count > 0:
                        result_item["benchmark_optimal_path_length"] = max(
                            0, optimal_state_count - 1
                        )
                optimal_action_sequence = getattr(benchmark_sample, "optimal_action_sequence", None)
                if optimal_action_sequence is not None:
                    if not isinstance(optimal_action_sequence, (list, tuple)):
                        optimal_action_sequence = list(optimal_action_sequence)
                    optimal_actions: list[int | str] = []
                    for action_val in optimal_action_sequence:
                        if isinstance(action_val, str):
                            optimal_actions.append(action_val)
                        else:
                            optimal_actions.append(int(action_val))
                    result_item["benchmark_optimal_action_sequence"] = optimal_actions
                    result_item["benchmark_optimal_action_count"] = len(optimal_actions)
                elif optimal_path is not None:
                    result_item["benchmark_optimal_action_count"] = max(0, len(optimal_path) - 1)

            if solved:
                if hasattr(search_result, "solution_trace"):
                    (
                        states_trace,
                        costs_trace,
                        dists_trace,
                        actions_trace,
                    ) = search_result.solution_trace()
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=state,
                        actions=actions_trace,
                        heuristic=heuristic_model,
                        q_fn=qfunction_model,
                        states=states_trace,
                        costs=costs_trace,
                        dists=dists_trace,
                    )
                elif hasattr(search_result, "solution_actions"):
                    actions = search_result.solution_actions()
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=state,
                        actions=actions,
                        heuristic=heuristic_model,
                        q_fn=qfunction_model,
                    )
                else:
                    path = search_result.get_solved_path()
                    path_steps = build_path_steps_from_nodes(
                        search_result=search_result,
                        path=path,
                        puzzle=puzzle,
                        solve_config=solve_config,
                    )

                path_cost = path_steps[-1].cost if path_steps else 0.0
                result_item["path_cost"] = float(path_cost)
                path_state_count = len(path_steps)
                result_item["path_state_count"] = path_state_count
                result_item["path_action_count"] = max(0, path_state_count - 1)
                actual_actions = [
                    int(step.action) for step in path_steps[:-1] if step.action is not None
                ]
                result_item["path_actions"] = actual_actions

                states = [step.state for step in path_steps]
                actual_dists = []
                estimated_dists = []
                for step in path_steps:
                    actual_dist = float(path_cost - step.cost)
                    estimated_dist = float(step.dist) if step.dist is not None else np.inf

                    if np.isfinite(estimated_dist):
                        actual_dists.append(actual_dist)
                        estimated_dists.append(estimated_dist)

                if states:
                    result_item["path_analysis"] = {
                        "actual": actual_dists,
                        "estimated": estimated_dists,
                        "states": xnp.concatenate(states),
                        "actions": actual_actions,
                    }

                if (
                    self.benchmark is not None
                    and result_item.get("benchmark_optimal_action_count") is not None
                ):
                    optimal_actions = getattr(benchmark_sample, "optimal_action_sequence", None)
                    optimal_sequence = result_item.get("benchmark_optimal_action_sequence")
                    if optimal_actions is not None and optimal_sequence is None:
                        if not isinstance(optimal_actions, (list, tuple)):
                            optimal_actions = list(optimal_actions)
                        optimal_sequence = []
                        for action_val in optimal_actions:
                            if isinstance(action_val, str):
                                optimal_sequence.append(action_val)
                            else:
                                optimal_sequence.append(int(action_val))
                        result_item["benchmark_optimal_action_sequence"] = optimal_sequence

                    if optimal_sequence:
                        first_opt = optimal_sequence[0]
                        if isinstance(first_opt, str):
                            actual_action_labels = []
                            for action_id in actual_actions:
                                try:
                                    label = puzzle.action_to_string(action_id)
                                except (AttributeError, ValueError, IndexError):
                                    label = str(action_id)
                                actual_action_labels.append(label)
                            result_item["path_action_strings"] = actual_action_labels
                            result_item["matches_optimal_path"] = actual_action_labels == list(
                                optimal_sequence
                            )
                        else:
                            optimal_ints = [int(a) for a in optimal_sequence]
                            result_item["matches_optimal_path"] = actual_actions == optimal_ints
                    elif result_item["path_action_count"] is not None:
                        expected_count = result_item["benchmark_optimal_action_count"]
                        result_item["matches_optimal_path"] = (
                            result_item["path_action_count"] == expected_count
                        )

            # Extract expansion data for plotting node value distributions
            if hasattr(search_result, "pop_generation"):
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
            total_completed = i + 1
            success_rate = (num_solved / total_completed) * 100

            solved_with_opt_reference = [
                r
                for r in solved_results
                if r.get("benchmark_optimal_path_cost") is not None
                and r.get("path_cost") is not None
            ]
            optimal_costs = [r["benchmark_optimal_path_cost"] for r in solved_with_opt_reference]
            optimal_hits = [
                r
                for r in solved_with_opt_reference
                if abs(r["path_cost"] - r["benchmark_optimal_path_cost"]) < 1e-6
            ]
            has_optimal_reference = bool(optimal_costs)
            optimal_rate = (
                (len(optimal_hits) / len(solved_with_opt_reference)) * 100
                if has_optimal_reference
                else None
            )

            success_key = "Success Rate/Optimal Rate" if has_optimal_reference else "Success Rate"
            if has_optimal_reference and optimal_rate is not None:
                rate_label = f"{success_rate:.2f}%/{optimal_rate:.2f}%"
            else:
                rate_label = f"{success_rate:.2f}%"

            pbar_desc_dict = {success_key: rate_label}
            if num_solved > 0:
                avg_time = sum(r["search_time_s"] for r in solved_results) / num_solved
                avg_nodes = sum(r["nodes_generated"] for r in solved_results) / num_solved
                path_costs = [
                    r["path_cost"] for r in solved_results if r.get("path_cost") is not None
                ]
                avg_path_cost = None
                if path_costs:
                    avg_path_cost = sum(path_costs) / len(path_costs)
                pbar_desc_dict.update(
                    {
                        "Avg Time (Solved)": f"{avg_time:.2f}s",
                        f"Avg {self.node_metric_label} (Solved)": f"{human_format(avg_nodes)}",
                    }
                )
                has_optimal_cost_info = bool(optimal_costs)
                cost_key = "Avg Cost/Optimal Cost" if has_optimal_cost_info else "Avg Cost"
                if avg_path_cost is not None:
                    cost_label = f"{avg_path_cost:.2f}"
                    if has_optimal_cost_info:
                        avg_optimal_cost = sum(optimal_costs) / len(optimal_costs)
                        delta = avg_path_cost - avg_optimal_cost
                        cost_label = f"{avg_path_cost:.2f}/{avg_optimal_cost:.2f} ({delta:+.2f})"
                    pbar_desc_dict[cost_key] = cost_label
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
