import os
import pickle
from datetime import datetime
from typing import Any, Dict, Tuple


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
