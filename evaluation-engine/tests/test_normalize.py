"""The normaliser changes how things are written, never what was said."""
import pytest

from transcribe_eval.normalize import normalize_text, normalize_words


def same(a, b):
    return normalize_text(a) == normalize_text(b)


@pytest.mark.parametrize("a, b", [
    ("1.2 million", "1,200,000"),
    ("one million", "1,000,000"),
    ("the sixteenth of May", "the 16th of May"),
    ("nineteen ninety-eight", "1998"),
    ("born in nineteen ninety-eight", "born in 1998"),
    ("twenty twenty-four", "2024"),
    ("fifteen thousand", "15,000"),
    ("fifty three", "53"),
    ("two hundred and five", "205"),
    ("S six five three two five three nine A", "S6532539A"),
    ("S six five three two five three nine A", "S 6532539 A"),
    ("T three eight six oh one three seven D", "T3860137D"),
    ("F nine nine three three triple two B", "F9933222B"),
    ("500k", "500,000"),
    ("1.2 million", "one point two million"),
    ("0.5%", "0.5 percent"),
    ("minus five", "-5"),
    ("Hi, Daniel.", "hi daniel"),
    ("thirty first", "31st"),
])
def test_representations_that_say_the_same_thing_match(a, b):
    assert same(a, b), (normalize_text(a), normalize_text(b))


@pytest.mark.parametrize("a, b", [
    ("1.2", "12"),
    ("00123", "123"),
    ("15,000", "50,000"),
    ("not guaranteed", "guaranteed"),
    ("six five", "eleven"),
    ("don't", "do not"),
    ("16th", "16"),
    ("1,00", "100"),
])
def test_different_information_stays_different(a, b):
    assert not same(a, b), (normalize_text(a), normalize_text(b))


def test_numbers_are_written_one_digit_per_token_and_keep_leading_zeros():
    assert normalize_text("00123") == ["0", "0", "1", "2", "3"]
    assert normalize_text("1.2") == ["1", "point", "2"]
    assert normalize_text("S6532539A") == ["s", "6", "5", "3", "2", "5", "3", "9", "a"]


def test_oh_is_zero_only_between_digits():
    assert normalize_text("oh I see") == ["oh", "i", "see"]
    assert normalize_text("six oh one") == ["6", "0", "1"]


def test_origins_point_back_to_the_raw_words():
    tokens, origins, _ = normalize_words(["My", "NRIC", "is", "S6532539A."])
    assert tokens[3:] == ["s", "6", "5", "3", "2", "5", "3", "9", "a"]
    assert set(origins[3:]) == {(3,)}
    tokens, origins, _ = normalize_words(["nineteen", "ninety-eight"])
    assert tokens == ["1", "9", "9", "8"] and origins == [(0, 1)] * 4


def test_ambiguous_forms_are_left_unchanged_and_reported():
    tokens, _, notes = normalize_words(["$50", "1,00", "thousand", "don't"])
    assert tokens[:3] == ["5", "0", "dollars"] and "1,00" in tokens and "don't" in tokens  # mapped (1.2.0) but still reported
    categories = {category for category, _ in notes}
    assert {"currency_symbol", "unrecognised_number", "contraction", "multiplier_without_number"} <= categories


def test_phrase_breaks_stop_numbers_joining_across_sentences():
    tokens, _, _ = normalize_words(["it", "was", "five.", "Thousand", "people"])
    assert tokens[:3] == ["it", "was", "5"] and "thousand" in tokens


def test_documented_limitation_a_cardinal_and_its_digit_reading_share_tokens():
    # One digit per token is what makes "S six five ..." match Transcribe's "S65...". The cost: these two readings become identical.
    assert same("sixty five", "six five")


def test_a_bare_currency_symbol_on_a_number_reads_as_the_number_and_its_word():
    assert same("€200", "200 euros") and same("€150,000", "150,000 euros") and same("$40,000", "40,000 dollars")
    assert not same("€200", "200 dollars") and not same("€200", "2,000 euros")
    assert normalize_text("¥500") == ["¥", "5", "0", "0"]  # yen or yuan: ambiguous, left as a symbol token
    assert normalize_text("S$1,500") == ["s$", "1", "5", "0", "0"]  # prefixed: left as a symbol token


def test_hyphenated_words_are_split():
    assert same("long-term", "long term") and same("seven-year", "7 year")
    assert normalize_text("ninety-eight") == ["9", "8"] and normalize_text("twenty-first") == ["2", "1", "ord"]
    assert not same("cut-off", "cutoff")  # a known, reported spelling difference
