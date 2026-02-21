from unittest.mock import patch

from rich.columns import Columns
from rich.text import Text
from rich.tree import Tree

from helpers import config_printer as cp


def test_is_single_level_config_dictionary():
    assert cp._is_single_level(123) is True
    assert cp._is_single_level({"a": 1, "b": 2}) is True
    assert cp._is_single_level({"a": {"b": 1}, "c": 2}) is False


def test_format_value_applies_human_format_to_nested_numbers():
    assert cp._format_value(1500) == "1.5K"
    assert cp._format_value(True) is True
    assert cp._format_value([1500, {"x": 1500}, (4, 5)]) == [1500, {"x": 1500}, (4, 5)]
    assert cp._format_value({"x": 1500}) == {"x": "1.5K"}


def test_print_config_chooses_layout_based_on_config_shape():
    captured = []

    class _DummyConsole:
        def print(self, obj):
            captured.append(obj)

    with patch.object(cp, "Console", lambda *args, **kwargs: _DummyConsole()):
        cp.print_config("empty", {})
        cp.print_config("small", {"a": 1, "b": 2})
        cp.print_config("large", {"a": 1, "b": {"c": 2}, "d": 3, "e": 4})

    assert captured[0].title == "[bold green]empty[/bold green]"
    assert isinstance(captured[0].renderable, Text)
    assert isinstance(captured[1].renderable, Tree)
    assert isinstance(captured[2].renderable, Columns)
    assert "Configuration is empty" in str(captured[0].renderable)
