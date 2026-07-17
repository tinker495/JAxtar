from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import jax
from flax.core.frozen_dict import FrozenDict

from .formatting import human_format

logger = logging.getLogger(__name__)


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


def _quant_itemsize(aqt_cfg: str | None) -> float | None:
    """Bytes per element for AQT-quantized kernels, or None if unknown/absent."""
    if not aqt_cfg:
        return None
    if "int8" in aqt_cfg:
        return 1
    if "int4" in aqt_cfg:
        return 0.5
    return None


def _is_quantizable_kernel(leaf: Any) -> bool:
    """AQT quantizes multi-dimensional float weights; 1-D leaves (bias, norm) stay float."""
    dtype_key = str(getattr(leaf, "dtype", "unknown"))
    return getattr(leaf, "ndim", 0) >= 2 and ("float" in dtype_key or "bfloat" in dtype_key)


def _leaf_stats(tree: Any, kernel_itemsize: float | None = None) -> Tuple[int, int]:
    """
    Sum (total_params, total_bytes) over a pytree.

    kernel_itemsize, if given, overrides the storage size of quantizable
    kernels (see _is_quantizable_kernel); other leaves use their real dtype.
    """
    total_params = 0
    total_bytes = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if leaf is None or not hasattr(leaf, "size") or not hasattr(leaf, "dtype"):
            continue
        try:
            n = int(leaf.size)
        except (AttributeError, TypeError, ValueError):
            logger.debug("Skipping leaf with non-integer size: %r", leaf)
            continue
        itemsize = int(getattr(leaf.dtype, "itemsize", 0))
        if kernel_itemsize is not None and _is_quantizable_kernel(leaf):
            itemsize = kernel_itemsize
        total_params += n
        total_bytes += int(n * itemsize)
    return total_params, total_bytes


def jax_param_stats(params: Any, aqt_cfg: str | None = None) -> Dict[str, Any]:
    """
    Compute parameter statistics from a JAX/Flax params pytree.

    Quantized trees display as "serving (original)":
    - Frozen AQT tree ({'params': ..., 'aqt': ...} after convert_to_serving):
      the 'aqt' collection holds the real quantized kernels, so the float
      kernels left in the other collections are superseded at serve time and
      count 0 bytes. Detected from the tree itself; aqt_cfg is not required.
    - Unconverted tree with aqt_cfg: serving bytes are estimated by pricing
      quantizable kernels at the quantized itemsize.

    total_params is always the logical parameter count — frozen 'aqt' copies
    are not extra parameters. Returns a JSON-serializable dict of strings.
    """
    is_frozen_aqt = (
        isinstance(params, (dict, FrozenDict)) and "params" in params and "aqt" in params
    )

    if is_frozen_aqt:
        base = {col: tree for col, tree in params.items() if col != "aqt"}
        total_params, orig_bytes = _leaf_stats(base)
        _, base_serving_bytes = _leaf_stats(base, kernel_itemsize=0)
        _, aqt_bytes = _leaf_stats(params["aqt"])
        serving_bytes = base_serving_bytes + aqt_bytes
    else:
        total_params, orig_bytes = _leaf_stats(params)
        quant_itemsize = _quant_itemsize(aqt_cfg)
        if quant_itemsize is None:
            return {
                "total_params": human_format(total_params),
                "total_bytes": human_format(orig_bytes),
                "total_size": _format_bytes(orig_bytes),
            }
        _, serving_bytes = _leaf_stats(params, kernel_itemsize=quant_itemsize)

    return {
        "total_params": human_format(total_params),
        "total_bytes": f"{human_format(serving_bytes)} ({human_format(orig_bytes)})",
        "total_size": f"{_format_bytes(serving_bytes)} ({_format_bytes(orig_bytes)})",
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
