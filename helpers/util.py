import json
from collections.abc import MutableMapping
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def convert_to_serializable_dict(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        # Recursively process the dict representation
        return convert_to_serializable_dict(obj.dict())
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable_dict(i) for i in obj]
    if isinstance(obj, type):
        return obj.__name__
    if callable(obj):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_hashable(val):
    if isinstance(val, list):
        return tuple(val)
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return val


def display_value(val):
    # Convert tuples back to lists for display, and pretty-print JSON strings
    if isinstance(val, tuple):
        return str(list(val))
    try:
        if isinstance(val, str):
            loaded = json.loads(val)
            if isinstance(loaded, dict) or isinstance(loaded, list):
                return json.dumps(loaded, indent=2)
    except json.JSONDecodeError:
        pass
    return str(val)


def map_kwargs_to_pydantic(
    model_class: Type[T], kwargs: Dict[str, Any], pop: bool = True
) -> Dict[str, Any]:
    """
    Extracts keys from kwargs that match the fields of a Pydantic model.

    Args:
        model_class: The Pydantic model class.
        kwargs: The dictionary of arguments (e.g., from Click).
        pop: If True, removes the matched keys from kwargs.

    Returns:
        A dictionary of arguments suitable for initializing the model.
    """
    model_fields = model_class.model_fields.keys()
    result = {}
    # Iterate over a copy of keys since we might pop
    for k in list(kwargs.keys()):
        if k in model_fields and kwargs[k] is not None:
            if pop:
                result[k] = kwargs.pop(k)
            else:
                result[k] = kwargs[k]
    return result
