"""Mandarin numbers, dates and identifiers: hand-written golden examples and rejection tables (no network, no credentials).

Every expected surface below is written by hand from the rules in `normalization.json` `language_rules.zh`; none is produced
by the module under test, so a bug shared by the renderer and the decoder cannot hide behind a round trip. The examples are
PROVISIONAL: they pass as code tests but still need native-speaker review before they count as linguistically accepted.
"""

from decimal import Decimal

import pytest

from fyp_evaluation.check_plans import zh_numbers as ZH
from fyp_evaluation.check_plans.render import RenderError

CARDINALS = {
    2: "二", 10: "十", 12: "十二", 15: "十五", 20: "二十", 110: "一百一十", 112: "一百一十二", 200: "两百", 700: "七百",
    1005: "一千零五", 2000: "两千", 2200: "两千二百", 10050: "一万零五十", 10012: "一万零一十二", 20200: "两万零二百",
    25000: "两万五千", 50000: "五万", 100050: "十万零五十", 120000: "十二万", 1200000: "一百二十万", 2500000: "两百五十万",
    7500000: "七百五十万", 10000000: "一千万", 100000000: "一亿", 200000000: "两亿", 100000050: "一亿零五十",
}


@pytest.mark.parametrize("number,surface", CARDINALS.items())
def test_cardinal_golden_examples(number, surface):
    assert ZH.render_cardinal(number) == surface
    assert ZH.parse_cardinal(surface) == number


@pytest.mark.parametrize("surface,number", [("两千两百", 2200), ("二千二百", 2200), ("二百", 200), ("二万", 20000), ("两万五千", 25000)])
def test_cardinal_decoder_tolerates_two_before_a_unit_only(surface, number):
    assert ZH.parse_cardinal(surface) == number


@pytest.mark.parametrize("surface", [
    "一千二", "两万五", "一百五", "十两", "两十", "两", "两三", "幺", "一十二", "一百十二", "二百零", "一万零", "零五", "百", "万",
    "一二", "二二", "一2", "1200", "一千二百三", "十 二", "", "一千万万",
])
def test_cardinal_decoder_rejects_shortened_ambiguous_and_non_canonical_surfaces(surface):
    with pytest.raises(RenderError):
        ZH.parse_cardinal(surface)


def test_only_non_negative_integers_below_a_trillion_render():
    for bad in (-1, 10 ** 12, 1.5, "7", True):
        with pytest.raises(RenderError):
            ZH.render_cardinal(bad)
    assert ZH.render_cardinal(0) == "零"


@pytest.mark.parametrize("value,surface", [
    (Decimal("37.5"), "三十七点五"), (Decimal("0.75"), "零点七五"), (Decimal("12.05"), "十二点零五"), (Decimal("200"), "两百"),
    (Decimal("0.5"), "零点五"), (Decimal("1200.25"), "一千二百点二五"),
])
def test_decimal_golden_examples(value, surface):
    assert ZH.render_decimal(value) == surface
    assert ZH.parse_decimal(surface) == value


def test_precision_is_never_lost_silently():
    assert ZH.render_decimal(Decimal("0.0625"), max_places=4) == "零点零六二五"
    with pytest.raises(RenderError, match="decimal places"):
        ZH.render_decimal(Decimal("0.125"), max_places=2)
    assert ZH.parse_decimal("零点七五") == Decimal("0.75")
    for bad in ("点五", "三十七点", "三点五零", "三点五点五", "三点两", "三点5", "3点五"):
        with pytest.raises(RenderError):
            ZH.parse_decimal(bad)


def test_percentage_golden_examples():
    assert ZH.render_percent(Decimal("0.75")) == "百分之零点七五"
    assert ZH.render_percent(Decimal("12")) == "百分之十二"
    assert ZH.parse_percent("百分之零点七五") == Decimal("0.75")
    with pytest.raises(RenderError):
        ZH.parse_percent("零点七五")


@pytest.mark.parametrize("year,surface", [(1967, "一九六七"), (2001, "二〇〇一"), (1998, "一九九八"), (1950, "一九五〇"), (2000, "二〇〇〇")])
def test_year_golden_examples(year, surface):
    assert ZH.render_year(year) == surface


@pytest.mark.parametrize("canonical,surface", [
    ("05 January 1967", "一九六七年一月五日"), ("12 October 1952", "一九五二年十月十二日"), ("31 December 2001", "二〇〇一年十二月三十一日"),
    ("01 March 1990", "一九九〇年三月一日"), ("29 February 2000", "二〇〇〇年二月二十九日"),
])
def test_date_golden_examples(canonical, surface):
    assert ZH.render_date(canonical) == surface
    assert ZH.parse_date(surface) == canonical


def test_date_decoder_accepts_either_zero_in_the_year_and_rejects_impossible_dates():
    assert ZH.parse_date("二零零一年五月五日") == ZH.parse_date("二〇〇一年五月五日") == "05 May 2001"
    for bad in ("二〇二三年二月三十日", "二〇二一年二月二十九日", "一九六七年十三月五日", "一九六七年零月五日", "一九六七年一月零五日",
                "一九六七年一月五", "一九六七一月五日", "67年一月五日", "一九六七年一月五日一", "一九六七年一月三十二日", "二〇〇一年一月〇五日"):
        with pytest.raises(RenderError):
            ZH.parse_date(bad)
    for bad in ("5 January 1967", "05 Janvier 1967", "30 February 2001"):
        with pytest.raises(RenderError):
            ZH.render_date(bad)


def test_identifier_golden_examples():
    assert ZH.render_identifier("S8779219E") == "S八七七九二一九E"
    assert ZH.render_identifier("G2036843A") == "G二零三六八四三A"
    assert ZH.parse_identifier("S八七七九二一九E") == "S8779219E"
    assert ZH.parse_identifier("G 二零三六八四三 A") == "G2036843A"  # spaces inside the surface are approved
    assert ZH.parse_identifier("S八七七九二一九E".replace("九E", "九　E")) == "S8779219E"


@pytest.mark.parametrize("surface", [
    "S8779219E", "S八七七九二一九", "八七七九二一九E", "s八七七九二一九e", "S八七七九二一九EE", "S幺七七九二一九E", "S两七七九二一九E",
    "S〇七七九二一九E", "S八七七九二一九O", "O八七七九二一九E", "S八七七九二一E", "S八七七九二一九一E", "SA八七七九二一九E",
])
def test_identifier_decoder_rejects_malformed_surfaces(surface):
    with pytest.raises(RenderError):
        ZH.parse_identifier(surface)
    with pytest.raises(RenderError):
        ZH.render_identifier("S877921E")
