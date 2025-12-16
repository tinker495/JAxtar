from __future__ import annotations

from typing import Any, Dict

import jax


def _format_bytes(num_bytes: int) -> str:
    """Human-readable binary size string (KiB, MiB, GiB...)."""
    if num_bytes is None:
        return "0B"
    n = int(num_bytes)
    if n < 0:
        return f"{n}B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(n)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{n}B"
    return f"{size:.2f}{units[unit_idx]}"


def jax_param_stats(params: Any) -> Dict[str, Any]:
    """
    Compute basic parameter statistics from a JAX/Flax params pytree.

    Returns a JSON-serializable dict (ints/strings only).
    """
    leaves = jax.tree_util.tree_leaves(params)
    total_params = 0
    total_bytes = 0
    dtype_params: Dict[str, int] = {}

    for leaf in leaves:
        if leaf is None:
            continue
        # JAX arrays / NumPy arrays / DeviceArrays generally have .size and .dtype
        if not hasattr(leaf, "size") or not hasattr(leaf, "dtype"):
            continue

        try:
            n = int(leaf.size)
        except Exception:
            continue

        try:
            itemsize = int(getattr(leaf.dtype, "itemsize", 0))
        except Exception:
            itemsize = 0

        total_params += n
        total_bytes += n * itemsize

        dtype_key = str(getattr(leaf, "dtype", "unknown"))
        dtype_params[dtype_key] = dtype_params.get(dtype_key, 0) + n

    return {
        "total_params": int(total_params),
        "total_bytes": int(total_bytes),
        "total_size": _format_bytes(int(total_bytes)),
        "dtype_params": dtype_params,
        "num_leaves": int(len(leaves)),
    }


def attach_runtime_metadata(
    component: Any,
    *,
    model_type: str | None = None,
    param_path: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Attach a small runtime-only metadata block to a heuristic/qfunction instance.
    This is used by CLI config printing (`*_metadata`) to show model type + param stats.
    """
    if component is None or not hasattr(component, "metadata"):
        return

    md = getattr(component, "metadata", None)
    if not isinstance(md, dict):
        md = {}
        setattr(component, "metadata", md)

    runtime = md.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        md["runtime"] = runtime

    if model_type is not None:
        runtime["model_type"] = model_type
    if param_path is not None:
        runtime["param_path"] = param_path

    # Params can be large; we only compute small aggregated stats.
    if hasattr(component, "params"):
        try:
            runtime["param_stats"] = jax_param_stats(getattr(component, "params"))
        except Exception:
            # Best-effort: don't break CLI for stats issues.
            runtime.setdefault("param_stats", {"error": "failed_to_compute"})

    if extra:
        runtime.update(extra)
