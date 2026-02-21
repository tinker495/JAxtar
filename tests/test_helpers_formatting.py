import click
import pytest

from helpers.formatting import (
    HUMAN_FLOAT,
    HUMAN_INT,
    human_format,
    human_format_to_float,
)


def test_human_format_to_float_supports_power_and_suffix_notation():
    assert human_format_to_float("2^10") == 1024.0
    assert human_format_to_float("1.5K") == 1500.0
    assert human_format_to_float("2.5M") == 2_500_000.0
    assert human_format_to_float("3e3") == 3000.0


def test_human_format_converts_number_ranges_and_power_of_two():
    assert human_format(12) == "12"
    assert human_format(999.9) == "1000"
    assert human_format(1024) == "2^10"
    assert human_format(1536) == "1.54K"
    assert human_format(1_000_000) == "1M"


def test_human_format_preserves_special_numbers():
    assert human_format(float("inf")) == "inf"
    assert human_format(float("-inf")) == "-inf"
    assert human_format(float("nan")) == "nan"


def test_human_int_param_type_supports_human_and_native_values():
    assert HUMAN_INT.convert(11, None, None) == 11
    assert HUMAN_INT.convert("1K", None, None) == 1000.0

    with pytest.raises(click.BadParameter):
        HUMAN_INT.convert("bad", None, None)


def test_human_float_param_type_supports_human_and_native_values():
    assert HUMAN_FLOAT.convert(1.25, None, None) == 1.25
    assert HUMAN_FLOAT.convert("2^8", None, None) == 256.0

    with pytest.raises(click.BadParameter):
        HUMAN_FLOAT.convert("??", None, None)
