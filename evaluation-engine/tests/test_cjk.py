"""Mandarin / mixed scoring (normaliser 1.4.0): Han characters are scoring units, Chinese numerals equal their digit spelling."""

import pytest

from transcribe_eval import hypothesis, normalize as n
from transcribe_eval.cjk import split_han
from transcribe_eval.reference import raw_words
from transcribe_eval.score import score_call


def tokens(text, rules=n.ALL_RULES):
    words = [piece for word in text.split() for piece, _, _ in split_han(word)]
    return n.normalize_text(" ".join(words), rules=rules)


def test_han_characters_are_individual_units_and_english_stays_per_word():
    assert tokens("您好 private banking") == ["您", "好", "private", "banking"]
    assert tokens("您好private") == ["您", "好", "private"]  # a run with no space is split at the script boundary
    assert tokens("好，谢谢。") == ["好", "谢", "谢"]  # Chinese punctuation is not a unit


def test_the_zero_character_is_kept():
    assert tokens("二〇三一") == ["2", "0", "3", "1"]


@pytest.mark.parametrize("chinese, digits", [
    ("七万五千", "7万5千"), ("二十五万", "25万"), ("两千", "2000"), ("一千五百", "1500"), ("五十万", "500000"),
    ("二〇三一", "2031"), ("F一二九六七一七A", "F1296717A"), ("十四点二九新加坡元", "14.29新加坡元"), ("百分之零点二五", "0.25%"),
    ("百分之5", "5%"), ("五欧元", "5欧元"), ("十个", "10个"), ("一千零五十", "1050"), ("一万零五", "10005"),
    ("一万五", "15000"), ("一亿五千万", "150000000"),
])
def test_chinese_numerals_equal_their_digit_spelling(chinese, digits):
    assert tokens(chinese) == tokens(digits)


def test_the_value_is_right_not_just_equal():
    assert "".join(tokens("七万五千")) == "75000" and "".join(tokens("一万零五")) == "10005" and "".join(tokens("一万五")) == "15000"


def test_a_different_number_stays_an_error():
    assert tokens("七万五千") != tokens("七万六千")
    assert tokens("二十五万") != tokens("25000")


def test_the_rule_can_be_switched_off():
    assert tokens("两千", rules=n.ALL_RULES - {"zh_numbers"}) == ["两", "千"]


def test_english_text_is_unchanged_by_the_chinese_rule():
    sample = "the account number is S6532539A, send 1,200,000 euros on the fourth of August 1991"
    assert tokens(sample) == n.normalize_text(sample) == tokens(sample, rules=n.ALL_RULES - {"zh_numbers"})


def test_reference_offsets_stay_exact_for_entity_spans():
    turns = [{"turn_id": 0, "speaker": "CLIENT", "text": "我要买五万 ETF 吧"}]
    words, positions, _, _ = raw_words(turns)
    assert words == ["我", "要", "买", "五", "万", "ETF", "吧"]
    for word, (turn, start, end) in zip(words, positions):
        assert turns[0]["text"][start:end] == word


def items(*contents, speaker="spk_0"):
    return [{"item": i, "content": c, "speaker": speaker, "confidence": 0.99, "start": None, "end": None} for i, c in enumerate(contents)]


def test_a_mandarin_call_scores_per_character():
    turns = [{"turn_id": 0, "speaker": "CLIENT", "text": "我要买五万的基金"}]
    perfect = score_call(turns, items("我要买", "5万", "的", "基金"), set(), None)
    assert perfect["wer"]["errors"] == 0 and perfect["wer"]["reference_words"] == 11  # 我 要 买 5 0 0 0 0 的 基 金
    wrong = score_call(turns, items("我要买", "5万", "得", "基金"), set(), None)
    assert wrong["wer"]["errors"] == 1 and wrong["wer"]["wer"] < 0.2  # one wrong character, not a whole sentence


def test_chinese_punctuation_in_a_transcript_ends_a_phrase(tmp_path):
    import json
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"results": {"transcripts": [{"transcript": "好。"}], "items": [
        {"type": "pronunciation", "alternatives": [{"content": "好", "confidence": "1.0"}]},
        {"type": "punctuation", "alternatives": [{"content": "。"}]}]}}), encoding="utf-8")
    items_, breaks, commas, _ = hypothesis.load_items(path)
    assert breaks == {0}
