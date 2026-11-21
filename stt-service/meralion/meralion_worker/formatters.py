"""Formatter registry. A formatter is a pure text function applied to the FINAL merged transcript only; the raw transcript is always kept next to it.

  none                       identity
  percentages-v2             vendored formatting.format_percentages (English spoken percentage fractions -> decimal percent; FORMATTER_VERSION "2")
  percentages-v2+amount-v1   format_percentages first, then formatting.join_split_amounts on its result (removes ONE comma inside a spelled English amount, only with positive evidence of a
                             single amount; AMOUNT_FORMATTER_VERSION "amount-v1"). The selected pipeline's formatter (Stage 2A); version "2+amount-v1".
Both formatters come from the vendored formatting.py, which is pinned to its own commit (see vendored/SOURCES.json).
"""
from meralion_worker import vendored

vendored.ensure_on_path()
import formatting as _formatting                                  # noqa: E402  (vendored, byte-identical to its pinned blob)


def _identity(text):
    return text


def _percentages_then_amounts(text):
    """Percentage wording first, then the amount comma (the order of the saved fee2_amt1 targets)."""
    return _formatting.join_split_amounts(_formatting.format_percentages(text))


REGISTRY = {"none": _identity, "percentages-v2": _formatting.format_percentages, "percentages-v2+amount-v1": _percentages_then_amounts}


def apply_formatter(name, text):
    try:
        function = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown formatter {name!r}; expected one of {sorted(REGISTRY)}")
    return function(text)


def formatter_version(name):
    return {"none": None, "percentages-v2": _formatting.FORMATTER_VERSION,
            "percentages-v2+amount-v1": f"{_formatting.FORMATTER_VERSION}+{_formatting.AMOUNT_FORMATTER_VERSION}"}[name]
