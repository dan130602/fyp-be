"""Display units: reference-anchored, merge-safe, and exactly consistent with the exported scores."""
import json

import pytest

from transcribe_eval.config import ENGINE_ROOT
from transcribe_eval.display import error_totals
from transcribe_eval.score import score_call

from test_score import items_from, parsed, turn


def units_for(reference, hypothesis_words, *, spans=None, turns=None, speaker="spk_0", raw_items=None, turn_times=None):
    ref_turns = turns or [turn(reference)]
    items, breaks = parsed(raw_items if raw_items is not None else items_from(hypothesis_words, speaker=speaker))
    return score_call(ref_turns, items, breaks, spans, turn_times)


def errors_of(result):
    return error_totals(result["units"])


def test_a_perfect_transcript_has_only_equal_units():
    result = units_for("hello there world", "hello there world".split())
    assert all(u["op"] == "equal" and u["errors"] == 0 for u in result["units"])
    assert [u["ref_text"] for u in result["units"]] == ["hello", "there", "world"]


def test_substitution_deletion_and_insertion_units_match_the_scores():
    result = units_for("the cat sat on the mat", "the dog sat on um the mat".split())
    ops = [(u["ref_text"], u["hyp_text"], u["op"]) for u in result["units"] if u["op"] != "equal"]
    assert ("cat", "dog", "substitute") in ops and ("", "um", "insert") in ops
    totals = errors_of(result)
    assert (totals["substitutions"], totals["deletions"], totals["insertions"]) == (1, 0, 1)
    wer = result["wer"]
    assert (wer["substitutions"], wer["deletions"], wer["insertions"]) == (1, 0, 1)
    deleted = units_for("the cat sat", "the sat".split())
    assert [u["op"] for u in deleted["units"]] == ["equal", "delete", "equal"] and deleted["units"][1]["hyp_text"] == ""


def test_a_spoken_identifier_matching_one_aws_item_is_one_unit():
    text = "My NRIC is S six five three A"
    result = units_for(text, "My NRIC is S653A".split())
    merged = [u for u in result["units"] if u["ref_text"].startswith("S ")]
    assert len(merged) == 1 and merged[0]["ref_text"] == "S six five three A" and merged[0]["op"] == "equal"
    assert merged[0]["hyp_text"] == "S653A" and len(merged[0]["items"]) == 1


def test_a_partial_identifier_substitution_is_kept_inside_the_merged_unit():
    result = units_for("My NRIC is S six five three A", "My NRIC is S654A".split())  # one digit wrong inside one AWS item
    merged = next(u for u in result["units"] if u["ref_text"].startswith("S "))
    assert merged["op"] == "substitute" and merged["errors"] == 1 and merged["substitutions"] == 1
    bad = [p for p in merged["parts"] if p["op"] != "equal"]
    assert bad == [{"ref": "3", "hyp": "4", "op": "substitute", "entity": False}]
    assert errors_of(result)["substitutions"] == result["wer"]["substitutions"] == 1


def test_an_item_with_both_hits_and_an_insertion_keeps_the_insertion():
    result = units_for("My NRIC is S six five three A", "My NRIC is S6553A".split())  # one extra digit inside the item
    merged = next(u for u in result["units"] if u["ref_text"].startswith("S "))
    assert merged["errors"] == 1 and merged["insertions"] == 1 and merged["op"] == "insert"
    assert sum(p["op"] == "equal" for p in merged["parts"]) == 5 and any(p["op"] == "insert" for p in merged["parts"])
    assert errors_of(result)["insertions"] == result["wer"]["insertions"] == 1


def test_a_mixed_unit_reports_every_kind_of_error_and_keeps_words_in_order():
    result = units_for("My NRIC is S six five three A", "My NRIC is S64A".split())  # one wrong, one missing
    merged = next(u for u in result["units"] if u["ref_text"].startswith("S "))
    assert merged["ref_text"] == "S six five three A"  # the deleted word stays inside the unit, in order
    assert merged["errors"] == 2 and merged["op"] == "mixed"
    assert (merged["substitutions"], merged["deletions"]) == (1, 1)
    totals, wer = errors_of(result), result["wer"]
    assert (totals["substitutions"], totals["deletions"], totals["insertions"]) == (wer["substitutions"], wer["deletions"], wer["insertions"])


def test_numbers_that_normalise_differently_but_say_the_same_are_equal_with_raw_forms_kept():
    result = units_for("born in nineteen ninety-eight", "born in 1998".split())
    unit = next(u for u in result["units"] if u["ref_text"].startswith("nineteen"))
    assert unit["op"] == "equal" and unit["ref_text"] == "nineteen ninety-eight" and unit["hyp_text"] == "1998"
    assert [p["ref"] for p in unit["parts"]] == ["1", "9", "9", "8"]


def test_entity_counted_errors_follow_the_entity_rules():
    text = "my nric is S six five A thanks"
    start = text.index("S six")
    spans = [{"turn_id": 0, "start": start, "end": start + len("S six five A"), "group": "value:K01", "label": "e", "text": "S six five A"}]
    result = units_for(text, "my nric is um S six extra five A thanks".split(), spans=spans)
    inserted = {u["hyp_text"]: u for u in result["units"] if u["op"] == "insert"}
    assert inserted["um"]["entity_errors"] == 0  # boundary insertion: counted in WER, not in entity WER
    assert inserted["extra"]["entity_errors"] == 1  # interior insertion
    assert errors_of(result)["entity_errors"] == result["entity_wer"]["errors"] == 1
    assert next(u for u in result["units"] if u["ref_text"] == "S")["entity_groups"] == ["value:K01"]


def test_speaker_mismatch_is_flagged_on_an_equal_unit():
    turns = [turn("hello there", "EMPLOYEE", 0), turn("hi", "CLIENT", 1)]
    raw = items_from(["hello"], speaker="spk_0") + items_from(["there"], speaker="spk_1") + items_from(["hi"], speaker="spk_1")
    result = units_for(None, None, turns=turns, raw_items=raw)
    flags = {u["ref_text"]: u["speaker_mismatch"] for u in result["units"]}
    assert flags["there"] is True and flags["hello"] is False


def test_times_come_from_aws_and_deletions_are_approximate():
    raw = []
    for word, start, end in (("the", "0.1", "0.3"), ("sat", "0.9", "1.2")):
        raw.append({"type": "pronunciation", "start_time": start, "end_time": end, "speaker_label": "spk_0",
                    "alternatives": [{"content": word, "confidence": "0.9"}]})
    result = units_for("the cat sat", None, raw_items=raw, turn_times={0: (0.0, 2.0)})
    first, deleted, last = result["units"]
    assert (first["start"], first["end"], first["approximate"]) == (0.1, 0.3, False)
    assert deleted["op"] == "delete" and deleted["approximate"] is True and deleted["time_source"] == "neighbours"
    assert (deleted["start"], deleted["end"]) == (0.3, 0.9)
    only_reference = units_for("lost words", None, raw_items=[], turn_times={0: (0.0, 2.0)})
    assert all(u["time_source"] == "reference_turn" and u["approximate"] for u in only_reference["units"])


# ---- the invariant on the real scored runs: displayed errors == exported errors --------------------------------------------------

REAL_RUNS = sorted((ENGINE_ROOT / "runs").glob("*/scores/units"))


@pytest.mark.skipif(not REAL_RUNS, reason="no scored runs on this machine")
@pytest.mark.parametrize("units_dir", REAL_RUNS)
def test_displayed_errors_equal_the_exported_scores_for_every_call(units_dir):
    calls = {json.loads(line)["call_id"]: json.loads(line) for line in (units_dir.parent / "calls.jsonl").read_text(encoding="utf-8").splitlines()}
    for path in sorted(units_dir.glob("CALL_*.json")):
        totals = error_totals(json.loads(path.read_text(encoding="utf-8"))["units"])
        wer = calls[path.stem]["wer"]
        assert (totals["substitutions"], totals["deletions"], totals["insertions"]) == (wer["substitutions"], wer["deletions"], wer["insertions"])
        entity = calls[path.stem]["entity_wer"]
        if entity is not None:
            assert totals["entity_errors"] == entity["errors"]
