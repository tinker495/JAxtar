from cli.config_utils import enrich_config
from config.pydantic_models import SearchOptions


def test_enrich_config_expands_json_strings_recursively():
    config = {
        "top": {
            "json_obj": '{"a": 1, "b": [2, 3]}',
            "json_list": "[1, 2, 3]",
            "plain": "hello",
            "number_like": "10",
        },
        "list": [
            '{"nested": true}',
            "x",
            "[4,5]",
        ],
    }

    out = enrich_config(config)

    assert out["top"]["json_obj"] == {"a": 1, "b": [2, 3]}
    assert out["top"]["json_list"] == [1, 2, 3]
    assert out["top"]["plain"] == "hello"
    assert out["top"]["number_like"] == "10"
    assert out["list"][0] == {"nested": True}
    assert out["list"][1] == "x"
    assert out["list"][2] == [4, 5]


def test_enrich_config_serializes_pydantic_models_before_expansion():
    config = {
        "search": SearchOptions(pop_ratio=2.5),
        "inner": {"search": SearchOptions(batch_size=128).model_dump()},
    }

    out = enrich_config(config)

    assert out["search"]["cost_weight"] == 0.6
    assert out["search"]["pop_ratio"] == 2.5
    assert out["search"]["batch_size"] == 10000
    assert out["search"]["max_node_size"] == 2000000
    assert out["search"]["vmap_size"] == 1
    assert out["inner"]["search"]["batch_size"] == 128
