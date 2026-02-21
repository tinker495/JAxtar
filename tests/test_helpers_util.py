import json

from pydantic import BaseModel

from helpers.util import (
    convert_to_serializable_dict,
    display_value,
    flatten_dict,
    make_hashable,
    map_kwargs_to_pydantic,
)


class SampleModel(BaseModel):
    name: str
    count: int


class ArgsModel(BaseModel):
    alpha: int
    beta: str


def sample_fn(x: int) -> int:
    return x + 1


def test_convert_to_serializable_dict_handles_nested_complex_objects():
    data = {
        "model": SampleModel(name="hello", count=3),
        "callable": sample_fn,
        "type_obj": int,
        "tuple_data": (1, 2, 3),
        "set_data": {"a", "b"},
        "dict_data": {"k": 1},
    }

    out = convert_to_serializable_dict(data)

    assert out["model"] == {"name": "hello", "count": 3}
    assert isinstance(out["tuple_data"], list)
    assert out["tuple_data"] == [1, 2, 3]
    assert out["set_data"] in (["a", "b"], ["b", "a"])
    assert out["dict_data"] == {"k": 1}
    assert out["type_obj"] == "int"
    assert out["callable"].startswith("<function sample_fn")


def test_make_hashable_converts_list_and_dict_inputs():
    assert make_hashable([1, 2, 3]) == (1, 2, 3)

    hashed = make_hashable({"b": 2, "a": 1})
    assert hashed == json.dumps({"a": 1, "b": 2}, sort_keys=True)


def test_display_value_formats_json_like_strings_and_tuples():
    assert display_value((1, 2, 3)) == "[1, 2, 3]"
    assert '"x": 1' in display_value('{"x": 1, "y": [2, 3]}')
    assert '"y": [\n    2,\n    3\n  ]' in display_value('{"x": 1, "y": [2, 3]}')
    assert display_value("plain string") == "plain string"


def test_flatten_dict_flattens_nested_mappings_recursively():
    nested = {
        "outer": {
            "inner": {
                "a": 1,
            },
            "b": 2,
        },
        "c": 3,
    }

    flat = flatten_dict(nested)
    assert flat == {"outer.inner.a": 1, "outer.b": 2, "c": 3}


def test_map_kwargs_to_pydantic_respects_none_values_and_pop_flag():
    kwargs = {"alpha": 7, "beta": None, "gamma": 3}
    out = map_kwargs_to_pydantic(ArgsModel, kwargs, pop=True)

    assert out == {"alpha": 7}
    assert "alpha" not in kwargs
    assert kwargs == {"beta": None, "gamma": 3}

    kwargs = {"alpha": 9, "beta": "z", "gamma": 1}
    out_no_pop = map_kwargs_to_pydantic(ArgsModel, kwargs, pop=False)

    assert out_no_pop == {"alpha": 9, "beta": "z"}
    assert kwargs == {"alpha": 9, "beta": "z", "gamma": 1}
