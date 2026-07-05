"""Source-scan guard: the chunked masked-eval block lives in exactly one place.

The partition -> chunked ``jax.lax.scan`` -> inverse-permutation scatter pattern
used to be inlined verbatim in four deferred-search evaluators. It now lives only
in ``JAxtar/utils/chunked_eval.py``; every consumer must delegate to it. These
lightweight string scans (mirroring the guards in ``test_solution_trace.py``) fail
if anyone re-inlines the primitive.
"""

from pathlib import Path

import JAxtar

_PKG = Path(JAxtar.__file__).resolve().parent

# Consumers that previously inlined the partition/scan/scatter block. Because the
# block was the sole user of ``stable_partition_three`` in each of these files, its
# absence is a precise signal that the block is gone (bi_astar_d's backward
# value-lookahead uses ``jax.lax.sort_key_val`` instead, so it is not caught here).
_MIGRATED_CONSUMERS = (
    "stars/astar_d.py",
    "bi_stars/bi_astar_d.py",
    "bi_stars/bi_qstar.py",
)


def _read(rel: str) -> str:
    return (_PKG / rel).read_text()


def test_consumers_do_not_reinline_partition_scan_scatter():
    for rel in _MIGRATED_CONSUMERS:
        src = _read(rel)
        assert "stable_partition_three" not in src, (
            f"{rel} re-inlines the chunked masked-eval partition; "
            "delegate to chunked_masked_eval instead"
        )
        assert "chunked_masked_eval" in src, f"{rel} should call chunked_masked_eval"


def test_id_astar_delegates_to_primitive():
    src = _read("id_stars/id_astar.py")
    assert "def _build_chunked_heuristic_eval" in src
    assert "chunked_masked_eval" in src, "id_astar must delegate to chunked_masked_eval"
    assert (
        "stable_partition_three" not in src
    ), "id_astar still inlines stable_partition_three instead of delegating"


def test_primitive_defined_only_in_chunked_eval():
    assert "def chunked_masked_eval" in _read("utils/chunked_eval.py")
    for rel in _MIGRATED_CONSUMERS + ("id_stars/id_astar.py",):
        assert "def chunked_masked_eval" not in _read(rel), f"{rel} must not redefine the primitive"
