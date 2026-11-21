import pytest

from meralion_worker import formatters


@pytest.mark.parametrize("text, expected", [
    ("The fee is half a percent.", "The fee is 0.5 percent."),
    ("It is one and a half percent a year.", "It is 1.5 percent a year."),
    ("a quarter of a percent", "0.25 percent"),
    ("three quarters of a percent", "0.75 percent"),
    ("two and a quarter percent", "2.25 percent"),
    ("not half a percent", "not 0.5 percent"),                         # the negation is left in place
])
def test_percentages_v2_rewrites_the_closed_grammar(text, expected):
    assert formatters.apply_formatter("percentages-v2", text) == expected


@pytest.mark.parametrize("text", [
    "between half to one percent",                                    # a range is never rewritten
    "half a percent to one percent",
    "a quarter",                                                      # no unit
    "费用是百分之零点五",                                                 # Chinese is never touched
    "我们收取半个百分点",
    "0.5% fee",                                                       # a written form stays as written
    "",
])
def test_percentages_v2_abstains_on_ambiguous_or_out_of_scope_text(text):
    assert formatters.apply_formatter("percentages-v2", text) == text


def test_mixed_text_only_changes_the_english_percentage():
    assert formatters.apply_formatter("percentages-v2", "我们收取半个百分点 half a percent") == "我们收取半个百分点 0.5 percent"


def test_none_is_the_identity_and_the_registry_is_closed():
    assert formatters.apply_formatter("none", "half a percent") == "half a percent"
    assert set(formatters.REGISTRY) == {"none", "percentages-v2", "percentages-v2+amount-v1"}
    with pytest.raises(ValueError, match="unknown formatter"):
        formatters.apply_formatter("v1", "x")


def test_versions():
    assert formatters.formatter_version("percentages-v2") == "2" and formatters.formatter_version("none") is None


# ---- the combined formatter percentages-v2+amount-v1 (Stage 2A). Examples follow stt-experiments/meralion/test_amount_format.py, written independently of the development data. ----

COMBINED = "percentages-v2+amount-v1"


def both(text):
    return formatters.apply_formatter(COMBINED, text)


@pytest.mark.parametrize("before, after", [
    ("The amount I'm committing to this transaction is two million, five hundred thousand. I will leave space.", "The amount I'm committing to this transaction is two million five hundred thousand. I will leave space."),
    ("the amount is two million, five hundred thousand dollars", "the amount is two million five hundred thousand dollars"),
    ("a limit price of fourteen thousand, two hundred and eighty five Japanese yen per unit", "a limit price of fourteen thousand two hundred and eighty five Japanese yen per unit"),
    ("a limit price at three thousand, three hundred and thirty-three Japanese yen.", "a limit price at three thousand three hundred and thirty-three Japanese yen."),
    ("The quantity is one thousand, five hundred units.", "The quantity is one thousand five hundred units."),
    ("Seven million, five hundred thousand dollars", "Seven million five hundred thousand dollars"),
])
def test_one_comma_inside_one_spelled_amount_is_removed_and_the_words_are_kept(before, after):
    assert both(before) == after


@pytest.mark.parametrize("text", [
    "five, ten, fifteen",
    "two hundred, three hundred",
    "one million, two million",
    "one thousand, five hundred and two hundred units",                          # a list
    "between one million, two hundred thousand and two million dollars",         # a range
    "nineteen ninety, nineteen ninety one",                                      # years
    "Two transactions: one million, two hundred thousand.",                      # can describe two separate amounts
    "the amounts are one million, two hundred thousand",
    "one million, two hundred thousand respectively",
    "one million, two hundred thousand",                                         # no cue on either side
    "both one million, two hundred thousand dollars",
    "from one million, two hundred thousand dollars to two million dollars",
    "the balance is thousand, five hundred dollars",                             # A has no multiplier
    "",
])
def test_ambiguous_or_unsupported_amounts_abstain(text):
    assert both(text) == text


def test_the_percentage_step_runs_first_and_the_amount_step_second(monkeypatch):
    order = []
    original_percentages, original_amounts = formatters._formatting.format_percentages, formatters._formatting.join_split_amounts
    monkeypatch.setattr(formatters._formatting, "format_percentages", lambda t: order.append("percentages") or original_percentages(t))
    monkeypatch.setattr(formatters._formatting, "join_split_amounts", lambda t: order.append("amounts") or original_amounts(t))
    both("the fee is half a percent")
    assert order == ["percentages", "amounts"]


def test_a_text_with_both_kinds_of_wording_gets_both_rewrites():
    text = "The fee is half a percent and the amount is two million, five hundred thousand dollars."
    assert both(text) == "The fee is 0.5 percent and the amount is two million five hundred thousand dollars."


def test_the_legacy_formatter_leaves_amounts_alone_and_the_combined_one_does_not_change_percentages_differently():
    text = "The fee is half a percent and the amount is two million, five hundred thousand dollars."
    assert formatters.apply_formatter("percentages-v2", text) == "The fee is 0.5 percent and the amount is two million, five hundred thousand dollars."
    for percentages_only in ("half a percent", "one and a half percent a year", "a quarter of a percent", "between half to one percent", "not half a percent", "0.5% fee"):
        assert both(percentages_only) == formatters.apply_formatter("percentages-v2", percentages_only)


def test_the_combined_formatter_never_touches_chinese_or_text_without_a_target():
    for text in ("费用是百分之零点五，金额是两百万", "我们收取半个百分点", "Thank you for your time.", "My order is for 5,000 units."):
        assert both(text) == text


def test_the_combined_formatter_is_stable_when_applied_twice():
    for text in ("The fee is half a percent and the amount is two million, five hundred thousand dollars.", "the amount is two million, five hundred thousand dollars", "five, ten, fifteen"):
        assert both(both(text)) == both(text)


def test_the_versions_are_the_two_formatter_versions_joined():
    assert formatters.formatter_version(COMBINED) == "2+amount-v1" and formatters.formatter_version("percentages-v2") == "2" and formatters.formatter_version("none") is None
    assert formatters.formatter_version(COMBINED) == f"{formatters._formatting.FORMATTER_VERSION}+{formatters._formatting.AMOUNT_FORMATTER_VERSION}"


def test_an_unknown_name_is_still_refused_and_the_amount_step_is_not_a_selectable_name_on_its_own():
    for name in ("amount-v1", "v2+amount-v1", "percentages-v2+amount-v2"):
        with pytest.raises(ValueError, match="unknown formatter"):
            formatters.apply_formatter(name, "x")
