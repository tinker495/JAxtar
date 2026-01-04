import os
import pickle
from datetime import datetime
from typing import Any, Dict, Tuple

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze


def save_params_with_metadata(path: str, params: Any, metadata: Dict[str, Any]):
    """
    Saves model parameters along with metadata to a specified path.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "params": params,
        "metadata": metadata,
        "timestamp": datetime.now().isoformat(),  # Add a timestamp automatically
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_params_with_metadata(path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads model parameters and metadata from a specified path.
    Returns a tuple of (params, metadata).
    Returns (None, {}) if loading fails or the format is unrecognized/invalid.
    """
    try:
        with open(path, "rb") as f:
            loaded_content = pickle.load(f)

        # New format: dictionary with 'params' and 'metadata' keys
        if isinstance(loaded_content, dict) and "metadata" in loaded_content:
            return loaded_content.get("params", None), loaded_content.get("metadata", {})
        # Old format: just the parameters directly
        else:
            # Validate if the old format content is a non-empty dictionary (typical Flax params)
            if (
                isinstance(loaded_content, dict) and loaded_content
            ):  # Check if it's a non-empty dict
                return loaded_content, {}
            else:
                # If it's not a dict, or an empty dict, consider it invalid old format
                print(f"Warning: Unrecognized or invalid old format for parameters in {path}.")
                return None, {}
    except Exception as e:
        print(f"Warning: Failed to load parameters from {path}. Error: {e}")
        return None, {}


def merge_params(new_params: Any, old_params: Any) -> Any:
    """
    Merges old_params into new_params.
    Values in old_params take precedence.
    Structure/keys present in new_params but missing in old_params are preserved (new values).
    Useful for initializing new model components (e.g. AQT) while keeping trained weights.
    """
    if old_params is None:
        return new_params

    # Convert to mutable dicts if needed
    target = unfreeze(new_params) if isinstance(new_params, FrozenDict) else new_params
    source = unfreeze(old_params) if isinstance(old_params, FrozenDict) else old_params

    # If standard dicts, we might need deep copy of target to avoid side effects if mutable,
    # but typically new_params is fresh.
    # We'll use a recursive update function.

    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                recursive_update(d[k], v)
            else:
                d[k] = v
        return d

    # We assume 'target' is a dictionary-like structure representing the full model
    if isinstance(target, dict) and isinstance(source, dict):
        merged = recursive_update(target, source)
    else:
        # Fallback for non-dict (though params should be dicts)
        merged = source

    return freeze(merged)
