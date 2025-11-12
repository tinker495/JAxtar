from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from .modules import (
    ACTIVATION_FN_REGISTRY,
    NORM_FN_REGISTRY,
    RESBLOCK_REGISTRY,
    get_activation_fn,
    get_norm_fn,
    get_resblock_fn,
)

CallableRegistry = Dict[str, Any]


def merge_saved_nn_args(
    current_kwargs: Dict[str, Any], saved_nn_args: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge kwargs provided at construction time with the ones that were stored in metadata.
    Current kwargs take precedence over saved ones so users can override defaults.
    """
    merged = {}
    if saved_nn_args:
        merged.update(saved_nn_args)
    merged.update(current_kwargs)
    return merged


def prepare_model_kwargs(raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize keyword arguments before instantiating a neural module by resolving
    string/callable aliases into concrete functions.
    """
    kwargs = deepcopy(raw_kwargs)
    kwargs["norm_fn"] = get_norm_fn(kwargs.get("norm_fn", "batch"))
    kwargs["activation"] = get_activation_fn(kwargs.get("activation", "relu"))
    kwargs["resblock_fn"] = get_resblock_fn(kwargs.get("resblock_fn", "standard"))
    kwargs["use_swiglu"] = kwargs.get("use_swiglu", False)
    return kwargs


def _callable_to_key(value: Any, registry: CallableRegistry, default_key: str) -> str:
    if isinstance(value, str):
        key = value.lower()
        if key in registry:
            return key
    for key, fn in registry.items():
        if fn == value:
            return key
    return default_key


def serialize_nn_args(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert resolved model kwargs into a metadata-friendly dictionary by replacing
    callables with their registry keys whenever possible.
    """
    encoded = {}
    for key, value in model_kwargs.items():
        if key == "norm_fn":
            encoded[key] = _callable_to_key(value, NORM_FN_REGISTRY, "batch")
        elif key == "activation":
            encoded[key] = _callable_to_key(value, ACTIVATION_FN_REGISTRY, "relu")
        elif key == "resblock_fn":
            encoded[key] = _callable_to_key(value, RESBLOCK_REGISTRY, "standard")
        else:
            encoded[key] = value
    return encoded


def resolve_model_kwargs(
    current_kwargs: Dict[str, Any], saved_nn_args: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience helper that merges saved args, prepares them for model instantiation,
    and returns both the resolved kwargs and their serializable representation.
    """
    merged = merge_saved_nn_args(current_kwargs, saved_nn_args)
    prepared = prepare_model_kwargs(merged)
    serialized = serialize_nn_args(prepared)
    return prepared, serialized
