import pytest

from helpers.formatting import (
    human_format,
    human_format_to_float,
    parse_human_int,
)


def test_human_format_to_float_supports_power_and_suffix_notation():
    assert human_format_to_float("2^10") == 1024.0
    assert human_format_to_float("1.5K") == 1500.0
    assert human_format_to_float("2.5M") == 2_500_000.0
    assert human_format_to_float("3e3") == 3000.0


def test_human_format_converts_number_ranges_without_power_special_case():
    assert human_format(12) == "12"
    assert human_format(999.9) == "1e+03"
    assert human_format(1024) == "1.02K"
    assert human_format(1536) == "1.54K"
    assert human_format(1_000_000) == "1M"


def test_human_format_preserves_special_numbers():
    assert human_format(float("inf")) == "inf"
    assert human_format(float("-inf")) == "-inf"
    assert human_format(float("nan")) == "nan"


def test_parse_human_int_supports_human_and_native_values():
    assert parse_human_int(11) == 11
    assert parse_human_int("1K") == 1000

    with pytest.raises(ValueError):
        parse_human_int("bad")


def test_human_float_parser_supports_human_and_native_values():
    assert human_format_to_float(1.25) == 1.25
    assert human_format_to_float("2^8") == 256.0

    with pytest.raises(ValueError):
        human_format_to_float("??")
