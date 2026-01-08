from __future__ import annotations

from typing import Any, Dict

import jax
from flax.core.frozen_dict import FrozenDict

from .formatting import human_format


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


def jax_param_stats(params: Any, aqt_cfg: str | None = None) -> Dict[str, Any]:
    """
    Compute basic parameter statistics from a JAX/Flax params pytree.

    If aqt_cfg is provided (e.g., 'int8'), it estimates the quantized size
    for multi-dimensional weights (ndim >= 2) in the 'params' collection.

    Returns a JSON-serializable dict.
    """

    def _get_raw_stats(params: Any, aqt_cfg: str | None = None) -> Dict[str, Any]:
        # 1. Detect AQT structure: if it's a dict with 'params' and 'aqt' collections
        is_aqt_structure = (
            isinstance(params, (dict, FrozenDict)) and "params" in params and "aqt" in params
        )

        if is_aqt_structure:
            p_stats = _get_raw_stats(params["params"], aqt_cfg=aqt_cfg)
            a_stats = _get_raw_stats(params["aqt"], aqt_cfg=None)

            total_params = p_stats["total_params"] + a_stats["total_params"]
            total_bytes = p_stats["total_bytes"] + a_stats["total_bytes"]

            res = {
                "total_params": total_params,
                "total_bytes": total_bytes,
            }

            if aqt_cfg:
                orig_p_stats = _get_raw_stats(params["params"], aqt_cfg=None)
                res["orig_total_params"] = orig_p_stats["total_params"]
                res["orig_total_bytes"] = orig_p_stats["total_bytes"]

            return res

        # Standard Pytree leaf summation
        leaves = jax.tree_util.tree_leaves(params)
        total_params = 0
        total_bytes = 0

        quant_itemsize = None
        if aqt_cfg:
            if "int8" in aqt_cfg:
                quant_itemsize = 1
            elif "int4" in aqt_cfg:
                quant_itemsize = 0.5

        for leaf in leaves:
            if leaf is None:
                continue
            if not hasattr(leaf, "size") or not hasattr(leaf, "dtype"):
                continue

            try:
                n = int(leaf.size)
            except (AttributeError, TypeError, ValueError):
                continue

            itemsize = int(getattr(leaf.dtype, "itemsize", 0))
            effective_itemsize = itemsize
            dtype_key = str(getattr(leaf, "dtype", "unknown"))

            if (
                quant_itemsize is not None
                and hasattr(leaf, "ndim")
                and leaf.ndim >= 2
                and ("float" in dtype_key or "bfloat" in dtype_key)
            ):
                effective_itemsize = quant_itemsize

            total_params += n
            total_bytes += int(n * effective_itemsize)

        return {
            "total_params": int(total_params),
            "total_bytes": int(total_bytes),
        }

    raw = _get_raw_stats(params, aqt_cfg)

    # Format the results
    if "orig_total_params" in raw:
        # AQT case with original stats
        return {
            "total_params": human_format(raw["total_params"]),
            "total_bytes": f"{human_format(raw['total_bytes'])} ({human_format(raw['orig_total_bytes'])})",
            "total_size": f"{_format_bytes(raw['total_bytes'])} ({_format_bytes(raw['orig_total_bytes'])})",
        }
    else:
        # Standard case
        return {
            "total_params": human_format(raw["total_params"]),
            "total_bytes": human_format(raw["total_bytes"]),
            "total_size": _format_bytes(raw["total_bytes"]),
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

    # Check for aqt_cfg
    aqt_cfg = getattr(component, "aqt_cfg", None)
    if aqt_cfg is None:
        aqt_cfg = md.get("aqt_cfg")
    if aqt_cfg is None and extra:
        aqt_cfg = extra.get("aqt_cfg")

    if aqt_cfg:
        runtime["aqt_cfg"] = aqt_cfg

    # Params can be large; we only compute small aggregated stats.
    if hasattr(component, "params"):
        try:
            runtime["param_stats"] = jax_param_stats(getattr(component, "params"), aqt_cfg=aqt_cfg)
        except (ValueError, RuntimeError, AttributeError, TypeError):
            # Best-effort: don't break CLI for stats issues.
            runtime.setdefault("param_stats", {"error": "failed_to_compute"})

    if extra:
        runtime.update(extra)
