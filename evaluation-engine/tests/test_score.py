"""Scoring rules on small hand-built cases, then the identity check on the real datasets (a perfect transcript must score perfectly)."""
import json
import re
import shutil

import pytest

from transcribe_eval.config import FYP_BE_ROOT
from transcribe_eval.reference import load_turns, selected_spans
from transcribe_eval.score import align, score_call, score_run

DATASETS = FYP_BE_ROOT / "evaluation-dataset" / "datasets"
LEGACY, ADVISORY = DATASETS / "legacy_pilot_v1", DATASETS / "advisory_execution_pilot_v1"


def items_from(words, speaker="spk_0", confidence="0.9"):
    """Transcribe-style items: one pronunciation item per word, with trailing punctuation as its own item."""
    out = []
    for word in words:
        core = word.rstrip(".,?!")
        out.append({"type": "pronunciation", "alternatives": [{"content": core, "confidence": confidence}], "speaker_label": speaker})
        if core != word:
            out.append({"type": "punctuation", "alternatives": [{"content": word[len(core):][0], "confidence": "0.0"}], "speaker_label": speaker})
    return out


def parsed(raw_items):
    items, breaks = [], set()
    for raw in raw_items:
        if raw["type"] == "punctuation":
            breaks.add(len(items) - 1)
            continue
        items.append({"item": len(items), "content": raw["alternatives"][0]["content"], "speaker": raw.get("speaker_label"),
                      "confidence": float(raw["alternatives"][0]["confidence"])})
    return items, breaks


def turn(text, speaker="EMPLOYEE", turn_id=0):
    return {"turn_id": turn_id, "speaker": speaker, "text": text}


# ---- alignment counts -----------------------------------------------------------------------------------------------------------------------

def test_a_known_example_gives_the_expected_substitutions_deletions_and_insertions():
    ref_ops, _, insertions = align(["a", "b", "c", "d"], ["a", "z", "c", "e", "d"])
    assert [op[0] for op in ref_ops] == ["equal", "substitute", "equal", "equal"] and insertions == [(3, 3)]
    items, breaks = parsed(items_from("the cat sat on mat".split()))
    result = score_call([turn("the cat sat on the mat")], items, breaks, None)
    assert result["wer"] == {"reference_words": 6, "substitutions": 0, "deletions": 1, "insertions": 0, "hits": 5, "errors": 1, "wer": 0.1667}


# ---- entity WER rules -----------------------------------------------------------------------------------------------------------------------

def test_overlapping_spans_count_each_word_once_and_only_interior_insertions_count():
    text = "my nric is S six five A thanks"
    ref_turn = turn(text)
    start = text.index("S six")
    spans = [{"turn_id": 0, "start": start, "end": start + len("S six five A"), "group": "value:K01", "label": "e1", "text": "S six five A"},
             {"turn_id": 0, "start": start, "end": start + len("S six"), "group": "event:READBACK", "label": "e2", "text": "S six"}]
    # hypothesis: an extra word inside the identifier, and one at its boundary (after "is")
    items, breaks = parsed(items_from("my nric is um S six extra five A thanks".split()))
    entity = score_call([ref_turn], items, breaks, spans)["entity_wer"]
    assert entity["reference_words"] == 4  # s 6 5 a, counted once despite two overlapping spans
    assert entity["insertions"] == 1 and entity["substitutions"] == 0 and entity["deletions"] == 0  # 'extra' counts, 'um' does not


def test_no_annotations_means_no_entity_wer():
    items, breaks = parsed(items_from("hello".split()))
    assert score_call([turn("hello")], items, breaks, None)["entity_wer"] is None


# ---- speaker attribution and confidence: counted per original AWS item -------------------------------------------------------------------

def test_swapped_speaker_labels_are_mapped_and_one_expanded_item_counts_once():
    turns = [turn("born in 1998", "EMPLOYEE", 0), turn("okay", "CLIENT", 1)]
    raw = items_from("born in 1998.".split(), speaker="spk_1") + items_from(["okay"], speaker="spk_0")
    items, breaks = parsed(raw)
    speaker = score_call(turns, items, breaks, None)["speaker"]
    assert speaker["label_mapping"] == {"spk_0": "CLIENT", "spk_1": "EMPLOYEE"}
    assert speaker["eligible_items"] == 4 and speaker["correct_items"] == 4  # '1998' is ONE item although it became four tokens


def test_an_item_is_only_eligible_when_all_its_tokens_match():
    items, breaks = parsed(items_from(["1999"]))
    speaker = score_call([turn("1998")], items, breaks, None)["speaker"]
    assert speaker["eligible_items"] == 0


def test_confidence_counts_every_pronunciation_item_once_including_zero():
    raw = items_from(["hello"], confidence="0.0") + items_from(["world."], confidence="1.0")
    items, breaks = parsed(raw)
    confidence = score_call([turn("hello world.")], items, breaks, None)["confidence"]
    assert confidence["items"] == 2 and confidence["sum"] == 1.0  # punctuation excluded, the zero-confidence word included


# ---- identity: a transcript equal to the reference scores perfectly on the real data ---------------------------------------------------

def _identity_run(tmp_path, dataset, call_id):
    turns = load_turns(dataset, call_id)
    raw = []
    for t in turns:
        raw += items_from(t["text"].split(), speaker="spk_0" if t["speaker"] == "EMPLOYEE" else "spk_1")
    run = tmp_path / "run"
    (run / "raw" / "transcripts").mkdir(parents=True)
    (run / "raw" / "transcripts" / f"{call_id}.json").write_text(json.dumps({"results": {"items": raw, "transcripts": [{"transcript": ""}]}}),
                                                                   encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps({"run_id": "identity", "dataset_id": dataset.name, "calls": [
        {"call_id": call_id, "status": "COMPLETED", "transcript_file": f"raw/transcripts/{call_id}.json"}]}), encoding="utf-8")
    return score_run(run, dataset_root=dataset)


def test_identity_on_the_legacy_call_scores_zero_wer_and_full_speaker_accuracy(tmp_path):
    summary = _identity_run(tmp_path, LEGACY, "CALL_001")
    assert summary["normalised_wer"]["errors"] == 0 and summary["speaker_attribution"]["accuracy"] == 1.0
    assert summary["entity_wer"] == {"available": False, "reason": "this dataset has no value/event annotations"}


def test_identity_on_the_advisory_call_scores_zero_entity_wer_and_its_spans_match_the_text(tmp_path):
    summary = _identity_run(tmp_path, ADVISORY, "CALL_001")
    assert summary["normalised_wer"]["errors"] == 0 and summary["speaker_attribution"]["accuracy"] == 1.0
    entity = summary["entity_wer"]
    assert entity["errors"] == 0 and entity["reference_words"] > 0
    turns = load_turns(ADVISORY, "CALL_001")
    texts = {t["turn_id"]: t["text"] for t in turns}
    spans = selected_spans(ADVISORY, "CALL_001", turns)
    assert spans and all(texts[s["turn_id"]][s["start"]:s["end"]] == s["text"] for s in spans)
    assert any(s["group"] == "value:K01" and re.search(r"\d|six|five|one|two|three|four|seven|eight|nine|zero", s["text"]) for s in spans)
