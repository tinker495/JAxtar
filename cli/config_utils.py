import json
from typing import Any, Dict

from helpers.util import convert_to_serializable_dict


def _expand_json_strings(obj: Any) -> Any:
    """
    Recursively expands strings that look like JSON objects into actual Python dicts/lists.
    """
    if isinstance(obj, dict):
        return {k: _expand_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_json_strings(v) for v in obj]
    if isinstance(obj, str):
        try:
            # Try to parse string as JSON
            # We only care if it parses into a complex type (dict or list)
            # or if it's a string representation of a number/bool/null that was stored as string
            parsed = json.loads(obj)
            if isinstance(parsed, (dict, list)):
                return _expand_json_strings(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    return obj


def enrich_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches the configuration dictionary by expanding JSON strings.
    This ensures that all configuration details are visible and structured.
    """
    # 1. Convert to serializable dict to handle Pydantic models
    # We do this first so we can manipulate the structure easily
    serializable_config = convert_to_serializable_dict(config)

    # 2. Construct new config
    enriched_config = {}

    # Copy existing config
    enriched_config.update(serializable_config)

    # 3. Recursively expand JSON strings
    enriched_config = _expand_json_strings(enriched_config)

    return enriched_config
