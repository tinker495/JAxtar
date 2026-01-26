import click
from puxle import Puzzle

from config.pydantic_models import SearchOptions, VisualizeOptions
from helpers import heuristic_dist_format, qfunction_dist_format
from heuristic.heuristic_base import Heuristic
from JAxtar.beamsearch.heuristic_beam import beam_builder
from JAxtar.beamsearch.q_beam import qbeam_builder
from JAxtar.bi_stars.bi_astar import bi_astar_builder
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder
from JAxtar.bi_stars.bi_qstar import bi_qstar_builder
from JAxtar.id_stars.id_astar import id_astar_builder
from JAxtar.id_stars.id_qstar import id_qstar_builder
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.q_base import QFunction

from .options import (
    heuristic_options,
    puzzle_options,
    qfunction_options,
    search_options,
    visualize_options,
)
from .search_runner import run_search_command


@click.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def astar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        astar_builder,
        "heuristic",
        heuristic,
        heuristic.distance,
        heuristic_dist_format,
        "A* Search Configuration",
    )


@click.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def astar_d(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        astar_d_builder,
        "heuristic",
        heuristic,
        heuristic.distance,
        heuristic_dist_format,
        "A* Deferred Search Configuration",
    )


@click.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def bi_astar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        bi_astar_builder,
        "heuristic",
        heuristic,
        heuristic.distance,
        heuristic_dist_format,
        "Bidirectional A* Search Configuration",
    )


@click.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def bi_astar_d(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        bi_astar_d_builder,
        "heuristic",
        heuristic,
        heuristic.distance,
        heuristic_dist_format,
        "Bidirectional A* Deferred Search Configuration",
    )


@click.command()
@puzzle_options
@search_options(variant="beam")
@heuristic_options
@visualize_options
def beam(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        beam_builder,
        "heuristic",
        heuristic,
        heuristic.distance,
        heuristic_dist_format,
        "Beam Search Configuration",
    )


@click.command()
@puzzle_options
@search_options(variant="beam")
@qfunction_options
@visualize_options
def qbeam(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    qfunction: QFunction,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        qbeam_builder,
        "qfunction",
        qfunction,
        qfunction.q_value,
        qfunction_dist_format,
        "Q-beam Search Configuration",
    )


@click.command()
@puzzle_options
@search_options
@qfunction_options
@visualize_options
def qstar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    qfunction: QFunction,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        qstar_builder,
        "qfunction",
        qfunction,
        qfunction.q_value,
        qfunction_dist_format,
        "Q* Search Configuration",
    )


@click.command()
@puzzle_options
@search_options
@qfunction_options
@visualize_options
def bi_qstar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    qfunction: QFunction,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        bi_qstar_builder,
        "qfunction",
        qfunction,
        qfunction.q_value,
        qfunction_dist_format,
        "Bidirectional Q* Search Configuration",
    )


@click.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def id_astar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        id_astar_builder,
        "heuristic",
        heuristic,
        heuristic.distance,
        heuristic_dist_format,
        "IDA* Search Configuration",
    )


@click.command()
@puzzle_options
@search_options
@qfunction_options
@visualize_options
def id_qstar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    qfunction: QFunction,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    run_search_command(
        puzzle,
        puzzle_name,
        seeds,
        search_options,
        visualize_options,
        id_qstar_builder,
        "qfunction",
        qfunction,
        qfunction.q_value,
        qfunction_dist_format,
        "ID-Q* Search Configuration",
    )
