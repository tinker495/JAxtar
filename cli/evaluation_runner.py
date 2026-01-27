import concurrent.futures
import inspect
import itertools
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xtructure.numpy as xnp
from puxle import Puzzle
from puxle.benchmark import Benchmark
from rich.console import Console

from config.pydantic_models import EvalOptions, PuzzleOptions
from helpers import human_format
from helpers.artifact_manager import ArtifactManager
from helpers.capture import tee_console
from helpers.config_printer import print_config
from helpers.logger import BaseLogger
from helpers.metrics import calculate_benchmark_metrics, calculate_heuristic_metrics
from helpers.path_analysis import extract_heuristic_accuracy_data
from helpers.plots import (
    plot_benchmark_path_comparison,
    plot_expansion_distribution,
    plot_heuristic_accuracy,
    plot_nodes_generated_by_path_cost,
    plot_path_cost_distribution,
    plot_search_time_by_path_cost,
    plot_search_tree_semantic,
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
from .config_utils import enrich_config

_VERIFY_BENCHMARK = None


def _init_verify_worker(benchmark):
    global _VERIFY_BENCHMARK
    _VERIFY_BENCHMARK = benchmark


def _verify_solution_worker(args):
    benchmark_sample, states, action_sequence = args
    return _VERIFY_BENCHMARK.verify_solution(
        benchmark_sample,
        states=states,
        action_sequence=action_sequence,
    )


@partial(jax.jit)
def _bulk_actual_estimated(
    costs: jnp.ndarray,
    dists: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    path_cost = costs[-1]
    actual = path_cost - costs
    valid = jnp.isfinite(dists)
    return actual, dists, valid


@partial(jax.jit)
def _bulk_actual_estimated_batch(
    costs: jnp.ndarray,
    dists: jnp.ndarray,
    path_costs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    actual = path_costs[:, None] - costs
    valid = jnp.isfinite(dists)
    return actual, dists, valid


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
                    f"[bold cyan]Run {i + 1}/{len(param_combinations)}: pr={pr}, cw={cw}, bs={bs}[/bold cyan]"
                )

            config_title = f"{self.run_label.replace('_', ' ').title()} Evaluation Configuration"
            print_config(config_title, enrich_config(config))

            # Some builder fns accept optional kwargs (e.g. show_compile_time). We only pass
            # supported kwargs to keep compatibility across different search implementations.
            builder_kwargs = {
                "pop_ratio": pr,
                "cost_weight": cw,
                "show_compile_time": current_eval_opts.show_compile_time,
            }
            sig = inspect.signature(self.search_builder_fn)
            supported_kwargs = {k: v for k, v in builder_kwargs.items() if k in sig.parameters}
            search_fn = self.search_builder_fn(
                self.puzzle,
                self.search_model,
                bs,
                self.eval_options.get_max_node_size(bs),
                **supported_kwargs,
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
                benchmark_metrics = calculate_benchmark_metrics(results)
                if benchmark_metrics:
                    config["benchmark_metrics"] = benchmark_metrics

                    # Log metrics to artifact manager
                    if "avg_optimal_cost" in benchmark_metrics:
                        am.log_scalar(
                            "benchmark/avg_optimal_cost",
                            benchmark_metrics["avg_optimal_cost"],
                        )
                    if "avg_path_cost" in benchmark_metrics:
                        am.log_scalar(
                            "benchmark/avg_path_cost",
                            benchmark_metrics["avg_path_cost"],
                        )
                    if "avg_cost_gap" in benchmark_metrics:
                        am.log_scalar("benchmark/avg_cost_gap", benchmark_metrics["avg_cost_gap"])
                    if "avg_optimal_actions" in benchmark_metrics:
                        am.log_scalar(
                            "benchmark/avg_optimal_actions",
                            benchmark_metrics["avg_optimal_actions"],
                        )
                    if "avg_path_actions" in benchmark_metrics:
                        am.log_scalar(
                            "benchmark/avg_path_actions",
                            benchmark_metrics["avg_path_actions"],
                        )
                    if "avg_action_gap" in benchmark_metrics:
                        am.log_scalar(
                            "benchmark/avg_action_gap",
                            benchmark_metrics["avg_action_gap"],
                        )
                    if "exact_optimal_path_rate" in benchmark_metrics:
                        am.log_scalar(
                            "benchmark/exact_optimal_path_rate",
                            benchmark_metrics["exact_optimal_path_rate"],
                        )

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

            file_suffix = (
                "_optimal_path"
                if heuristic_metrics and heuristic_metrics.get("has_optimal_path_used")
                else ""
            )
            fig = plot_heuristic_accuracy(results, metrics=heuristic_metrics)
            am.save_and_log_plot(f"heuristic_accuracy{file_suffix}", fig)

            for r in results[: current_eval_opts.max_expansion_plots]:
                if r.get("expansion_analysis"):
                    # Original Expansion Plots
                    fig = plot_expansion_distribution(
                        [r], scatter_max_points=current_eval_opts.scatter_max_points
                    )
                    am.save_and_log_plot(
                        f"expansion_dist_seed_{r['seed']}",
                        fig,
                        sub_dir="expansion_plots",
                    )

                    # New Semantic Search Tree Plot (g vs h)
                    try:
                        fig_tree = plot_search_tree_semantic(
                            r,
                            max_points=current_eval_opts.max_node_size,  # Use a large limit
                        )
                        am.save_and_log_plot(
                            f"search_tree_semantic_seed_{r['seed']}",
                            fig_tree,
                            sub_dir="expansion_plots",
                        )
                    except (ValueError, RuntimeError, AttributeError, OSError) as e:
                        print(f"Warning: Failed to generate semantic search tree plot: {e}")

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

    def _evaluate_single_sample(
        self,
        identifier: int,
        search_fn,
        heuristic_model,
        qfunction_model,
    ) -> tuple[dict, Optional[dict]]:
        benchmark_sample = None
        if self.benchmark is not None:
            sample_id = identifier
            benchmark_sample = self.benchmark.get_sample(sample_id)
            solve_config = benchmark_sample.solve_config
            state = benchmark_sample.state
            run_identifier = sample_id
        else:
            seed = int(identifier)
            solve_config, state = self.puzzle.get_inits(jax.random.PRNGKey(seed))
            run_identifier = seed

        start_time = time.time()
        search_result = search_fn(solve_config, state)
        is_bidirectional = (
            hasattr(search_result, "meeting")
            and hasattr(search_result, "forward")
            and hasattr(search_result, "backward")
        )
        if is_bidirectional:
            solved = bool(search_result.meeting.found.block_until_ready())
        else:
            solved = bool(search_result.solved.block_until_ready())
        end_time = time.time()

        search_time = end_time - start_time
        if is_bidirectional:
            generated_nodes = int(jax.device_get(search_result.total_generated))
        else:
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
            "benchmark_verification_error": None,
            "benchmark_has_optimal_action_sequence": False,
        }

        if self.benchmark is not None:
            result_item["benchmark_sample_id"] = run_identifier
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
                    result_item["benchmark_optimal_path_length"] = max(0, optimal_state_count - 1)
            optimal_action_sequence = getattr(benchmark_sample, "optimal_action_sequence", None)
            if optimal_action_sequence is not None:
                result_item["benchmark_has_optimal_action_sequence"] = True
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

        deferred_payload = {
            "result_item": result_item,
            "solve_config": solve_config,
            "initial_state": state,
            "benchmark_sample": benchmark_sample,
        }
        if solved:
            if is_bidirectional:
                from JAxtar.bi_stars.bi_astar import reconstruct_bidirectional_path

                bi_pairs = reconstruct_bidirectional_path(search_result, self.puzzle)
                actions = [a for a, _ in bi_pairs[1:]]
                states_trace = [s for _, s in bi_pairs]
                path_steps = build_path_steps_from_actions(
                    puzzle=self.puzzle,
                    solve_config=solve_config,
                    initial_state=state,
                    actions=actions,
                    heuristic=heuristic_model,
                    q_fn=qfunction_model,
                    states=states_trace,
                )
            elif hasattr(search_result, "solution_trace"):
                (
                    states_trace,
                    costs_trace,
                    dists_trace,
                    actions_trace,
                ) = search_result.solution_trace()
                path_steps = build_path_steps_from_actions(
                    puzzle=self.puzzle,
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
                    puzzle=self.puzzle,
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
                    puzzle=self.puzzle,
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
            costs_list = [step.cost for step in path_steps]
            dists_list = [
                float(step.dist) if step.dist is not None else np.inf for step in path_steps
            ]
            states_concat = xnp.concatenate(states) if states else None

            if (
                self.benchmark is not None
                and states
                and hasattr(self.benchmark, "verify_solution")
                and result_item.get("matches_optimal_path") is None
            ):
                if actual_actions and not result_item.get("path_action_strings"):
                    action_to_string_fn = getattr(self.puzzle, "action_to_string", None)
                    actual_action_labels = []
                    for action_id in actual_actions:
                        if action_to_string_fn is None:
                            label = str(action_id)
                        else:
                            try:
                                label = action_to_string_fn(action_id)
                            except (ValueError, IndexError):
                                label = str(action_id)
                        actual_action_labels.append(label)
                    result_item["path_action_strings"] = actual_action_labels

                try:
                    verification_result = self.benchmark.verify_solution(
                        benchmark_sample,
                        states=states,
                        action_sequence=result_item.get("path_action_strings"),
                    )
                except Exception as exc:  # noqa: BLE001
                    verification_result = None
                    result_item["benchmark_verification_error"] = str(exc)

                result_item["matches_optimal_path"] = verification_result
                deferred_payload["path_action_strings"] = result_item.get("path_action_strings")
                deferred_payload["verify_result"] = verification_result
                deferred_payload["verify_error"] = result_item.get("benchmark_verification_error")
            deferred_payload.update(
                {
                    "path_steps": path_steps,
                    "states": states,
                    "actual_actions": actual_actions,
                    "path_costs": costs_list,
                    "path_dists": dists_list,
                    "path_len": len(costs_list),
                    "states_concat": states_concat,
                }
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
                    analysis_data = {
                        "pop_generation": pop_generations,
                        "cost": costs,
                        "dist": dists,
                    }

                    # Try to extract states and graph structure (parent indices)
                    try:
                        # Extract states
                        all_states = search_result.hashtable.table
                        expanded_states = jax.tree_util.tree_map(
                            lambda x: x[expanded_nodes_mask], all_states
                        )
                        expanded_states_np = jax.tree_util.tree_map(
                            lambda x: np.asarray(x), expanded_states
                        )

                        # Flatten structured states to a single 2D array (N, -1) for visualization/analysis
                        leaves = jax.tree_util.tree_leaves(expanded_states_np)
                        if leaves:
                            N = leaves[0].shape[0]
                            flat_states_np = np.concatenate(
                                [x.reshape(N, -1) for x in leaves], axis=1
                            )
                            analysis_data["states"] = flat_states_np
                        else:
                            analysis_data["states"] = expanded_states_np

                        # Extract parent structure for graph visualization
                        # Original indices in the hashtable (0 to capacity)
                        capacity = expanded_nodes_mask.shape[0]
                        original_indices = np.arange(capacity)[np.asarray(expanded_nodes_mask)]
                        analysis_data["original_indices"] = original_indices

                        # Parent indices (pointing to hashtable slots)
                        if hasattr(search_result, "parent") and hasattr(
                            search_result.parent, "hashidx"
                        ):
                            parents = search_result.parent.hashidx.index
                            expanded_parents = parents[expanded_nodes_mask]
                            analysis_data["parent_indices"] = np.asarray(expanded_parents)
                        if solved:
                            try:
                                solved_hash = int(
                                    np.asarray(
                                        jax.device_get(search_result.solved_idx.hashidx.index)
                                    )
                                )
                                analysis_data["solved_index"] = solved_hash
                            except (
                                AttributeError,
                                KeyError,
                                ValueError,
                                TypeError,
                            ) as exc:
                                print(f"Warning: Could not extract solved index: {exc}")

                    except (AttributeError, KeyError, ValueError, TypeError) as e:
                        print(
                            f"Warning: Could not extract states/parents for expansion analysis: {e}"
                        )

                    result_item["expansion_analysis"] = analysis_data

        return result_item, deferred_payload

    def _finalize_deferred_results(
        self,
        deferred_payloads: list[dict],
        heuristic_model,
        qfunction_model,
    ) -> None:
        batched_payloads = []
        batched_costs = []
        batched_dists = []
        batched_lengths = []
        batched_path_costs = []
        max_len = 0

        verify_jobs = []
        action_to_string_fn = getattr(self.puzzle, "action_to_string", None)
        for idx, payload in enumerate(deferred_payloads):
            result_item = payload["result_item"]
            path_steps = payload.get("path_steps")
            states = payload.get("states") or []
            actual_actions = payload.get("actual_actions") or []

            if payload.get("path_action_strings") is not None:
                result_item["path_action_strings"] = payload.get("path_action_strings")
            elif path_steps and actual_actions and not result_item.get("path_action_strings"):
                actual_action_labels = []
                for action_id in actual_actions:
                    if action_to_string_fn is None:
                        label = str(action_id)
                    else:
                        try:
                            label = action_to_string_fn(action_id)
                        except (ValueError, IndexError):
                            label = str(action_id)
                    actual_action_labels.append(label)
                result_item["path_action_strings"] = actual_action_labels

            if (
                self.benchmark is not None
                and states
                and hasattr(self.benchmark, "verify_solution")
                and result_item.get("matches_optimal_path") is None
            ):
                verify_jobs.append(
                    (
                        idx,
                        payload.get("benchmark_sample"),
                        states,
                        result_item.get("path_action_strings"),
                    )
                )

        for payload in deferred_payloads:
            result_item = payload["result_item"]
            path_steps = payload.get("path_steps")
            states = payload.get("states") or []
            if path_steps and states and not result_item.get("path_analysis"):
                costs = payload.get("path_costs") or [step.cost for step in path_steps]
                dists = payload.get("path_dists") or [
                    float(step.dist) if step.dist is not None else np.inf for step in path_steps
                ]
                length = payload.get("path_len") or len(costs)
                if length == 0:
                    continue
                payload["_batch_index"] = len(batched_payloads)
                batched_payloads.append(payload)
                batched_costs.append(costs)
                batched_dists.append(dists)
                batched_lengths.append(length)
                batched_path_costs.append(float(costs[-1]))
                if length > max_len:
                    max_len = length

        verify_results = {}
        if verify_jobs:
            use_process_pool = False
            try:
                pickle.dumps(self.benchmark)
                sample_job = verify_jobs[0]
                pickle.dumps(sample_job[1])
                pickle.dumps(sample_job[2])
                pickle.dumps(sample_job[3])
                use_process_pool = True
            except (pickle.PickleError, TypeError, AttributeError):
                use_process_pool = False

            max_workers = min(len(verify_jobs), max(1, os.cpu_count() or 1))
            if use_process_pool:
                ctx = mp.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=ctx,
                    initializer=_init_verify_worker,
                    initargs=(self.benchmark,),
                ) as executor:
                    future_map = {
                        executor.submit(
                            _verify_solution_worker,
                            (sample, states, actions),
                        ): idx
                        for idx, sample, states, actions in verify_jobs
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        idx = future_map[future]
                        try:
                            verify_results[idx] = (future.result(), None)
                        except Exception as exc:  # noqa: BLE001
                            verify_results[idx] = (None, str(exc))
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(32, max_workers)
                ) as executor:
                    future_map = {
                        executor.submit(
                            self.benchmark.verify_solution,
                            sample,
                            states=states,
                            action_sequence=actions,
                        ): idx
                        for idx, sample, states, actions in verify_jobs
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        idx = future_map[future]
                        try:
                            verify_results[idx] = (future.result(), None)
                        except Exception as exc:  # noqa: BLE001
                            verify_results[idx] = (None, str(exc))

        batched_actual = None
        batched_estimated = None
        batched_valid = None
        padded_costs = None
        padded_dists = None
        if batched_payloads:
            padded_costs = np.full((len(batched_payloads), max_len), 0.0, dtype=np.float32)
            padded_dists = np.full((len(batched_payloads), max_len), np.inf, dtype=np.float32)
            for i, (costs, dists, length) in enumerate(
                zip(batched_costs, batched_dists, batched_lengths)
            ):
                padded_costs[i, :length] = np.asarray(costs, dtype=np.float32)
                padded_dists[i, :length] = np.asarray(dists, dtype=np.float32)
                if length < max_len:
                    padded_costs[i, length:] = padded_costs[i, length - 1]

        finalize_bar = trange(
            len(deferred_payloads),
            desc="Finalizing Results",
        )
        for idx in finalize_bar:
            payload = deferred_payloads[idx]
            result_item = payload["result_item"]
            path_steps = payload.get("path_steps")
            states = payload.get("states") or []
            actual_actions = payload.get("actual_actions") or []
            benchmark_sample = payload.get("benchmark_sample")
            solve_config = payload.get("solve_config")
            initial_state = payload.get("initial_state")

            if path_steps and states and not result_item.get("path_analysis"):
                batch_index = payload.get("_batch_index")
                if batched_actual is None and batched_payloads:
                    actual_arr, estimated_arr, valid_mask = _bulk_actual_estimated_batch(
                        jnp.asarray(padded_costs),
                        jnp.asarray(padded_dists),
                        jnp.asarray(batched_path_costs, dtype=jnp.float32),
                    )
                    batched_actual = np.asarray(actual_arr)
                    batched_estimated = np.asarray(estimated_arr)
                    batched_valid = np.asarray(valid_mask)

                if batched_actual is not None and batch_index is not None:
                    length = batched_lengths[batch_index]
                    actual_np = batched_actual[batch_index, :length]
                    estimated_np = batched_estimated[batch_index, :length]
                    valid_np = batched_valid[batch_index, :length]
                    actual_dists = [float(v) for v in actual_np[valid_np]]
                    estimated_dists = [float(v) for v in estimated_np[valid_np]]
                else:
                    costs_arr = jnp.asarray(
                        [step.cost for step in path_steps],
                        dtype=jnp.float32,
                    )
                    dists_arr = jnp.asarray(
                        [
                            float(step.dist) if step.dist is not None else np.inf
                            for step in path_steps
                        ],
                        dtype=jnp.float32,
                    )
                    actual_dists_arr, estimated_dists_arr, valid_mask = _bulk_actual_estimated(
                        costs_arr,
                        dists_arr,
                    )
                    actual_np = np.asarray(actual_dists_arr)
                    estimated_np = np.asarray(estimated_dists_arr)
                    valid_np = np.asarray(valid_mask)
                    actual_dists = [float(v) for v in actual_np[valid_np]]
                    estimated_dists = [float(v) for v in estimated_np[valid_np]]

                result_item["path_analysis"] = {
                    "actual": actual_dists,
                    "estimated": estimated_dists,
                    "states": payload.get("states_concat") or xnp.concatenate(states),
                    "actions": actual_actions,
                }

            if result_item.get("matches_optimal_path") is None:
                if (
                    payload.get("verify_result") is not None
                    or payload.get("verify_error") is not None
                ):
                    result_item["matches_optimal_path"] = payload.get("verify_result")
                    if payload.get("verify_error") is not None:
                        result_item["benchmark_verification_error"] = payload.get("verify_error")
                else:
                    verify_result = verify_results.get(idx)
                    if verify_result is not None:
                        verification_result, error_text = verify_result
                        if error_text is not None:
                            result_item["benchmark_verification_error"] = error_text
                        if verification_result is not None:
                            result_item["matches_optimal_path"] = verification_result

            if self.benchmark is not None and solve_config is not None:
                optimal_sequence = result_item.get("benchmark_optimal_action_sequence")
                if optimal_sequence is None and benchmark_sample is not None:
                    optimal_sequence = getattr(benchmark_sample, "optimal_action_sequence", None)

                optimal_path = None
                if benchmark_sample is not None:
                    optimal_path = getattr(benchmark_sample, "optimal_path", None)

                if optimal_sequence or optimal_path:
                    optimal_analysis = extract_heuristic_accuracy_data(
                        puzzle=self.puzzle,
                        solve_config=solve_config,
                        initial_state=initial_state,
                        action_sequence=optimal_sequence,
                        path_states=optimal_path,
                        heuristic_model=heuristic_model,
                        qfunction_model=qfunction_model,
                    )
                    if optimal_analysis:
                        result_item["path_analysis"] = optimal_analysis
                        result_item["used_optimal_path_for_analysis"] = True

    def _run_evaluation(
        self,
        search_fn,
        puzzle: Puzzle,
        eval_inputs: List[int],
    ) -> list[dict]:
        num_puzzles = len(eval_inputs)
        results = []
        deferred_payloads = []

        pbar = trange(
            num_puzzles,
            desc="Running Evaluations",
        )

        heuristic_model = self.search_model if isinstance(self.search_model, Heuristic) else None
        qfunction_model = self.search_model if isinstance(self.search_model, QFunction) else None

        for i in pbar:
            identifier = eval_inputs[i]

            result_item, deferred_payload = self._evaluate_single_sample(
                identifier=identifier,
                search_fn=search_fn,
                heuristic_model=heuristic_model,
                qfunction_model=qfunction_model,
            )

            results.append(result_item)
            if deferred_payload is not None:
                deferred_payloads.append(deferred_payload)

            solved_results = [r for r in results if r["solved"]]
            num_solved = len(solved_results)
            total_completed = i + 1
            success_rate = (num_solved / total_completed) * 100

            optimal_reference_results = [
                r for r in solved_results if r.get("matches_optimal_path") is not None
            ]
            optimal_hits = [r for r in optimal_reference_results if r["matches_optimal_path"]]
            optimal_rate = (
                (len(optimal_hits) / len(optimal_reference_results)) * 100
                if optimal_reference_results
                else None
            )

            success_key = (
                "Success Rate/Optimal Rate" if optimal_rate is not None else "Success Rate"
            )
            if optimal_rate is not None:
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
                solved_with_optimal_cost = [
                    r
                    for r in solved_results
                    if r.get("benchmark_has_optimal_action_sequence")
                    and r.get("benchmark_optimal_path_cost") is not None
                    and r.get("path_cost") is not None
                ]
                optimal_costs = [r["benchmark_optimal_path_cost"] for r in solved_with_optimal_cost]
                has_optimal_cost_info = bool(solved_with_optimal_cost)
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

        if deferred_payloads:
            self._finalize_deferred_results(
                deferred_payloads,
                heuristic_model=heuristic_model,
                qfunction_model=qfunction_model,
            )

        return results


def run_evaluation_sweep(
    puzzle: Puzzle,
    puzzle_name: str,
    search_model: Union[Heuristic, QFunction],
    search_model_name: str,
    search_builder_fn: Callable,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    run_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    logger: Optional[BaseLogger] = None,
    step: int = 0,
    **kwargs,
):
    runner = EvaluationRunner(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=search_model,
        search_model_name=search_model_name,
        run_label=run_label,
        search_builder_fn=search_builder_fn,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        output_dir=output_dir,
        logger=logger,
        step=step,
        **kwargs,
    )

    with tee_console(runner.log_path):
        config_title_label = run_label if run_label else search_model_name
        config_title = f"{config_title_label.replace('_', ' ').title()} Evaluation Configuration"
        print_config(
            config_title,
            enrich_config(
                {
                    "puzzle_options": puzzle_opts.dict(),
                    search_model_name: search_model.__class__.__name__,
                    f"{search_model_name}_metadata": getattr(search_model, "metadata", {}),
                    "eval_options": eval_options.dict(),
                }
            ),
        )
        runner.run()
