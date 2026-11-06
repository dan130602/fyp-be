"""Scoring-only equivalences added in normaliser 1.3.0: what must now match, and what must stay an error."""
import hashlib
import re

import pytest

from transcribe_eval import normalize as n
from transcribe_eval.normalize import normalize_text, normalize_words


def same(a, b):
    return normalize_text(a) == normalize_text(b)


@pytest.mark.parametrize("a, b", [
    ("My date of birth is 4 August 1991.", "My date of birth is August 4th 1991."),
    ("4 August 1991", "August 4th, 1991"),
    ("the fourth of August nineteen ninety-one", "4 August 1991"),
    ("the fourth of August nineteen ninety-one", "August the fourth, 1991"),
    ("13 May 1990", "May 13th 1990"), ("the 13th of May", "May 13th"),
    ("sixteenth of May nineteen ninety-eight", "May 16th, 1998"),
    ("the 1st of March", "March 1st"),
    ("29 February 2024", "February 29th 2024"),
    ("travelling", "traveling"),
    ("organise the cheque", "organize the check"),
    ("it'll", "it will"), ("we're", "we are"), ("they've", "they have"), ("I'm", "I am"),
    ("don't", "do not"), ("isn't", "is not"), ("won't", "will not"), ("can't", "can not"), ("I’ll", "I will"),
    ("Okay", "OK"), ("okay.", "OK,"), ("cut-off", "cutoff"), ("cut off", "cutoff"),
    ("for $50,000 in SGD", "for 50,000 in SGD"),
    ("$50,000 in USD", "50,000 in USD"), ("$50,000 USD", "50,000 USD"), ("$50 US dollars", "50 US dollars"),
    ("$50,000, in SGD", "50,000 in SGD"),
    ("€200 euros", "200 euros"), ("€200 in EUR", "200 in EUR"),
])
def test_equivalent_forms_match(a, b):
    assert same(a, b), (normalize_text(a), normalize_text(b))


@pytest.mark.parametrize("a, b", [
    ("4 August 1991", "14 August 1991"), ("4 August 1991", "4 September 1991"), ("4 August 1991", "4 August 1992"),
    ("4 August 1991", "4 August"),
    ("30 February 1991", "February 30th 1991"),  # not a date, so neither is canonicalised and the two stay different
    ("29 February 2023", "February 29th 2023"),  # 2023 is not a leap year
    ("won't", "will"), ("can't", "can"), ("don't", "do"), ("it'll", "it"),
    ("200 euros", "200 dollars"), ("$30,000 in EUR", "30,000 in EUR"), ("€200 in SGD", "200 in SGD"),
    ("$200", "200"), ("200", "200 dollars"), ("$50,000", "50,000"),
    ("S6532539A", "S6532593A"),
    ("you may 3 times", "you 3 times"),
    ("tonne", "ton"), ("disc", "disk"),
])
def test_different_information_stays_different(a, b):
    assert not same(a, b), (normalize_text(a), normalize_text(b))


def test_may_and_march_are_only_months_with_an_ordinal_an_of_or_a_year():
    assert normalize_text("you may 3 times") == ["you", "may", "3", "times"]
    assert normalize_text("we march 3 miles") == ["we", "march", "3", "miles"]
    assert same("the 13th of May", "May 13th") and same("3 March 2024", "March 3rd 2024")
    assert not same("13 May", "May 13th")  # documented limitation: a bare "13 May" is not provably a date, so it is left as written


def test_a_date_becomes_one_unit_with_one_origin_and_is_reported():
    tokens, origins, notes, applied = n.normalize_words_traced(["on", "the", "fourth", "of", "August,", "1991."])
    assert tokens == ["on", "august", "4", "1", "9", "9", "1"]
    assert set(origins[1:]) == {(1, 2, 3, 4, 5)}  # the, fourth, of, August, 1991: every raw word stays traceable
    assert [(rule, text) for rule, text, _ in applied] == [("dates", "the fourth of August, 1991.")]
    assert ("applied_dates", "the fourth of August, 1991.") in notes


def test_dates_and_currency_never_match_across_a_sentence_end_a_turn_or_a_speaker():
    def matched(words, **options):
        return [text for rule, text, _ in n.normalize_words_traced(words, **options)[3] if rule == "dates"]

    assert matched("born on 4 August. 1991 was good".split()) == ["4 August."]  # the year after a full stop is a different sentence
    assert matched("born on 4 August, 1991 was good".split()) == ["4 August, 1991"]  # a comma may sit inside a date
    assert matched("born on 4 August 1991".split(), boundaries_after={3}) == ["4 August"]  # a turn/speaker change before the year
    words = ["I", "was", "born", "on", "4", "August", "1991"]
    assert normalize_words(words)[0][4:] == ["august", "4", "1", "9", "9", "1"]
    split = normalize_words(words, boundaries_after={4})[0]  # a turn or speaker change between the day and the month
    assert split == ["i", "was", "born", "on", "4", "august", "1", "9", "9", "1"]
    assert normalize_words(["$50,000", "in", "SGD"], boundaries_after={0})[0] == ["5", "0", "0", "0", "0", "dollars", "in", "sgd"]
    assert normalize_words(["$50,000.", "SGD", "next"])[0][:6] == ["5", "0", "0", "0", "0", "dollars"]  # sentence end in between


def test_currency_symbol_rules_are_conservative():
    assert normalize_text("$30,000 in EUR") == ["3", "0", "0", "0", "0", "dollars", "in", "eur"]  # the conflict stays an error
    assert normalize_text("$200") == ["2", "0", "0", "dollars"] and normalize_text("200") == ["2", "0", "0"]
    notes = normalize_words(["$30,000", "in", "EUR"])[2]
    assert ("currency_conflict", "$30,000 in EUR") in notes and ("currency_symbol", "$30,000") in notes
    notes = normalize_words(["$50,000", "in", "SGD"])[2]
    assert ("applied_currency", "$50,000 in SGD") in notes and not any(category == "currency_symbol" for category, _ in notes)
    assert normalize_text("£50 in GBP") == ["5", "0", "pounds", "in", "gbp"]  # only $ and € are in the closed table


def test_each_rule_can_be_switched_off_which_reproduces_1_2_0():
    for text, rule, expected in [
        ("4 August 1991", "dates", ["4", "august", "1", "9", "9", "1"]),
        ("travelling", "spelling", ["travelling"]),
        ("it'll", "contractions", ["it'll"]),
        ("OK", "variants", ["ok"]),
        ("$50,000 in SGD", "currency", ["5", "0", "0", "0", "0", "dollars", "in", "sgd"]),
    ]:
        assert normalize_text(text, rules=n.ALL_RULES - {rule}) == expected
    assert normalize_text("4 August 1991 travelling it'll OK $5 in SGD", rules=()) == [
        "4", "august", "1", "9", "9", "1", "travelling", "it'll", "ok", "5", "dollars", "in", "sgd"]


def test_the_spelling_data_is_the_pinned_whisper_file_with_reviewed_exclusions():
    assert hashlib.sha256((n.DATA / "english_spelling.json").read_bytes()).hexdigest() == n.SPELLING_SOURCE["sha256"]
    assert n.BRITISH_TO_AMERICAN["travelling"] == "traveling" and len(n.BRITISH_TO_AMERICAN) > 1500
    assert not {"tonne", "disc", "philtre", "mhm", "archaeology", "flyer / flier"} & set(n.BRITISH_TO_AMERICAN)
    assert all(re.fullmatch(r"[a-z]+", k) and re.fullmatch(r"[a-z]+", v) for k, v in n.BRITISH_TO_AMERICAN.items())
    info = n.rules_info()
    assert info["normalizer_version"] == "1.4.0" and info["variants"]["reviewed"] and info["spelling"]["tag"] == "v20240930"


def test_unreviewed_project_variants_are_not_applied(monkeypatch):
    monkeypatch.setattr(n, "PROJECT_VARIANTS", {})
    assert not same("OK", "okay")
