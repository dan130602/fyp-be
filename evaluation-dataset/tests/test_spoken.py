"""ARTIFICIAL gold-authoring examples only; never prediction normalization."""

from copy import deepcopy

import pytest

from fyp_evaluation.validation.spoken import verify_spoken_value


@pytest.mark.parametrize("kind,value,text", [
    ("text", "ARTIFICIAL Copper Basket", "An ARTIFICIAL Copper Basket."),
    ("text", "EUR", "E, U. R"), ("text", "EUR", "E-U R"), ("text", "EUR", "EUR"),
    ("text", "S0000000X", "S,0.0-0 0 0 0 0 X"), ("text", "S0000000X", "S0000000X"),
    ("text", "S0000000X", "S zero zero zero zero zero zero zero X"),
    ("text", "F1234567A", "F one two three four five six seven A"),
    ("dob_dd_month_yyyy", "01 May 1982", "01 May 1982"),
    ("dob_dd_month_yyyy", "01 May 1982", "1 May 1982"),
    ("dob_dd_month_yyyy", "01 May 1982", "FIRST MAY NINETEEN EIGHTY TWO"),
    ("dob_dd_month_yyyy", "21 May 1982", "twenty-first May nineteen eighty two"),
    ("dob_dd_month_yyyy", "31 May 1982", "thirty-first May 1982"),
    ("dob_dd_month_yyyy", "04 August 1991", "the fourth of August, nineteen ninety-one"),
    ("dob_dd_month_yyyy", "04 August 1991", "four August nineteen ninety one"),
    ("dob_dd_month_yyyy", "04 August 1991", "4 August, 1991"),
    ("dob_dd_month_yyyy", "21 May 1982", "the twenty first of May, nineteen eighty-two"),
    ("text", None, ""),
    # Regression: a compound day's OWN space- or hyphen-joined form must still
    # match itself; only a DIFFERENT day's word must be rejected as its suffix.
    ("dob_dd_month_yyyy", "24 August 1991", "the twenty fourth of August, nineteen ninety-one"),
    ("dob_dd_month_yyyy", "29 February 1988", "the twenty ninth of February, nineteen eighty-eight"),
    ("dob_dd_month_yyyy", "21 August 1991", "the twenty first of August, nineteen ninety-one"),
    # Regression: boundary anchors must not reject a legitimate value simply
    # because ordinary punctuation or surrounding words are adjacent to it.
    ("text", "S0000000X", "the code is S0000000X exactly"),
    ("text", "S0000000X", "(S0000000X)"),
])
def test_permitted_forms_preserve_expected(kind, value, text):
    expected = {"kind": "extraction", "value_type": kind, "expected_value": value}
    before = deepcopy(expected)
    assert verify_spoken_value(expected, text) is None
    assert expected == before


@pytest.mark.parametrize("kind,value,text", [
    ("text", "ARTIFICIAL Copper Basket", "ARTIFICIAL Silver Basket"),
    ("text", "ARTIFICIAL Copper Basket", "artificial copper basket"),
    ("text", "S0000000X", "S0000001X"), ("text", "S0000000X", "S/0/0/0/0/0/0/0/X"),
    ("text", "S0000000X", "S zero zero zero zero zero zero one X"),
    ("text", "EUR", "GBP"), ("text", "EUR", "E/U/R"), ("text", "EUR", "e u r"),
    ("dob_dd_month_yyyy", "01 May 1982", "1 May 1983"),
    ("dob_dd_month_yyyy", "01 May 1982", "11 May 1982"),
    ("dob_dd_month_yyyy", "01 May 1982", "twenty-first May 1982"),
    ("dob_dd_month_yyyy", "01 May 1982", "first May nineteen eighty three"),
    ("dob_dd_month_yyyy", "01 May 1982", "1 May 19820"),
    ("dob_dd_month_yyyy", "01 May 1982", "1 June 1982"),
    ("dob_dd_month_yyyy", "04 August 1991", "the fifth of August, nineteen ninety-one"),
    ("dob_dd_month_yyyy", "04 August 1991", "the fourth of August, nineteen ninety-two"),
    ("dob_dd_month_yyyy", "04 August 1991", "the twenty-fourth of August, nineteen ninety-one"),
    ("dob_dd_month_yyyy", "31 February 1982", "31 February 1982"),
    ("integer", 7, "7"),  # Other scalar rules remain schema-valid, outside pilot verifier.
    # Regression (reported defect): a compound day-word must not accept a
    # different locked day merely because that day's word is its suffix.
    ("dob_dd_month_yyyy", "04 August 1991", "the twenty fourth of August, nineteen ninety-one"),
    ("dob_dd_month_yyyy", "09 February 1988", "the twenty ninth of February, nineteen eighty-eight"),
    ("dob_dd_month_yyyy", "01 May 1982", "the twenty first of May, nineteen eighty-two"),
    ("dob_dd_month_yyyy", "01 May 1982", "TWENTY FIRST OF MAY, NINETEEN EIGHTY-TWO"),
    # Regression (reported defect): a locked identifier/currency must not
    # accept a longer alphanumeric run merely containing it as a substring.
    ("text", "S0000000X", "My artificial identifier is XS0000000X."),
    ("text", "S0000000X", "S0000000XY is the artificial identifier"),
    ("text", "EUR", "MEUR is not a currency code"),
    ("text", "EUR", "EURO is not the same code"),
])
def test_mismatches_rejected_without_repair(kind, value, text):
    expected = {"kind": "extraction", "value_type": kind, "expected_value": value}
    before = deepcopy(expected)
    assert verify_spoken_value(expected, text)
    assert expected == before
