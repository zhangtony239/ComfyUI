import pytest

from comfy.cli_args import parser
from comfy_execution.validation import format_value_not_in_list_details


def _legacy_details(input_name, invalid_vals, combo_options):
    """The historical details format for short lists, kept byte-identical."""
    return f"{input_name}: {', '.join(repr(v) for v in invalid_vals)} not in {str(combo_options)}"


def test_short_list_includes_full_options_by_default():
    options = ["a.safetensors", "b.safetensors"]
    details, truncated = format_value_not_in_list_details("ckpt_name", ["missing.safetensors"], options)

    assert not truncated
    assert details == _legacy_details("ckpt_name", ["missing.safetensors"], options)
    assert "a.safetensors" in details
    assert "b.safetensors" in details


def test_long_list_is_summarized_by_default():
    options = [f"model_{i}.safetensors" for i in range(21)]
    details, truncated = format_value_not_in_list_details("ckpt_name", ["missing.safetensors"], options)

    assert truncated
    assert details == "ckpt_name: 'missing.safetensors' not in (list of length 21)"
    assert "model_0.safetensors" not in details


def test_max_items_boundary():
    at_limit = [f"m{i}" for i in range(20)]
    details, truncated = format_value_not_in_list_details("ckpt_name", ["nope"], at_limit)
    assert not truncated
    assert details == _legacy_details("ckpt_name", ["nope"], at_limit)

    over_limit = at_limit + ["m20"]
    details, truncated = format_value_not_in_list_details("ckpt_name", ["nope"], over_limit)
    assert truncated
    assert "(list of length 21)" in details


def test_force_truncate_summarizes_short_lists():
    options = ["a.safetensors", "b.safetensors"]
    details, truncated = format_value_not_in_list_details(
        "ckpt_name", ["missing.safetensors"], options, force_truncate=True
    )

    assert truncated
    assert details == "ckpt_name: 'missing.safetensors' not in (list of length 2)"
    # The input name and offending value stay in the error so it remains debuggable.
    assert "ckpt_name" in details
    assert "missing.safetensors" in details
    # None of the valid options appear in the error text.
    assert "a.safetensors" not in details
    assert "b.safetensors" not in details


@pytest.mark.parametrize("force_truncate", [False, True])
def test_multiple_invalid_values_are_preserved(force_truncate):
    options = ["x", "y"]
    details, _ = format_value_not_in_list_details(
        "values", ["bad1", "bad2"], options, force_truncate=force_truncate
    )

    assert "'bad1', 'bad2'" in details


def test_cli_flag_defaults_off():
    args = parser.parse_args([])
    assert args.truncate_validation_error_lists is False


def test_cli_flag_parses_on():
    args = parser.parse_args(["--truncate-validation-error-lists"])
    assert args.truncate_validation_error_lists is True
