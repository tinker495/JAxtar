import json
from typing import Any, Dict

from helpers.util import convert_to_serializable_dict


def _expand_json_strings(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_json_strings(v) for v in obj]
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, (dict, list)):
                return _expand_json_strings(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    return obj


def enrich_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return _expand_json_strings(convert_to_serializable_dict(config))
