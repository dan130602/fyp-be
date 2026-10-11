"""Regression test for the comma-separated-identifier formatting correction.

Reads the real five-call pilot dataset (not ARTIFICIAL fixtures) to guard a
specific past defect: K01 identifiers were once spelled out character-by-
character with commas in dialogue/evidence text (a Polly-pronunciation
convention that predates audio generation), which no longer matches the
compact `expected_value`. This test locks in that the compact form is used
consistently across the expected value, canonical dialogue, gold evidence
and the generated transcript, for every call that has a stated identifier.

No AWS, Polly or LLM call is made; only local deterministic conversion.
"""

import re
from pathlib import Path

import pytest

from fyp_evaluation.transcripts.converter import build_transcript_artifacts
from fyp_evaluation.validation.schemas import load_json, validate
from fyp_evaluation.validation.transcript import validate_transcript_pair

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
CALLS_DIR = ROOT / "datasets/legacy_pilot_v1/calls"

COMMA_SPELLED_IDENTIFIER = re.compile(r"[A-Z], [0-9],(?: [0-9],)* [0-9], [A-Z]")

ALL_CALLS = ["CALL_001", "CALL_002", "CALL_003", "CALL_004", "CALL_005"]
CALLS_WITH_STATED_IDENTIFIER = ["CALL_001", "CALL_002", "CALL_003", "CALL_004"]


def _ground_truth(call_id):
    return load_json(CALLS_DIR / call_id / "ground_truth.json")


def _k01_check(ground_truth):
    return next(check for check in ground_truth["checks"] if check["question_id"] == "K01")


@pytest.mark.parametrize("call_id", ALL_CALLS)
def test_no_comma_spelled_identifier_remains_anywhere_in_ground_truth(call_id):
    ground_truth = _ground_truth(call_id)
    for turn in ground_truth["dialogue"]:
        assert not COMMA_SPELLED_IDENTIFIER.search(turn["text"]), (
            f"{call_id} turn {turn['turn_id']}: still contains a comma-spelled identifier"
        )
    k01 = _k01_check(ground_truth)
    assert not COMMA_SPELLED_IDENTIFIER.search(k01["gold_evidence_text"])
    if k01["evidence_plan"]["mode"] == "presence":
        for item in k01["evidence_plan"]["planned_evidence"]:
            assert not COMMA_SPELLED_IDENTIFIER.search(item["text"])


@pytest.mark.parametrize("call_id", CALLS_WITH_STATED_IDENTIFIER)
def test_k01_expected_value_matches_dialogue_and_evidence_exactly(call_id):
    ground_truth = _ground_truth(call_id)
    k01 = _k01_check(ground_truth)
    expected_value = k01["expected"]["expected_value"]
    assert expected_value is not None

    turn_id = k01["gold_evidence_turns"][0]
    turn = next(t for t in ground_truth["dialogue"] if t["turn_id"] == turn_id)
    assert expected_value in turn["text"]
    assert turn["critical_tokens"] == [expected_value]
    assert expected_value in k01["gold_evidence_text"]
    assert expected_value in k01["evidence_plan"]["planned_evidence"][0]["text"]


def test_call_005_identifier_remains_absent_and_untouched():
    ground_truth = _ground_truth("CALL_005")
    k01 = _k01_check(ground_truth)
    assert k01["expected"]["expected_value"] is None
    assert k01["evidence_plan"]["mode"] == "absence"
    assert k01["gold_evidence_turns"] == []
    assert k01["gold_evidence_text"] == ""


@pytest.mark.parametrize("call_id", CALLS_WITH_STATED_IDENTIFIER)
def test_regenerated_transcript_carries_the_corrected_identifier(call_id):
    ground_truth = _ground_truth(call_id)
    k01 = _k01_check(ground_truth)
    expected_value = k01["expected"]["expected_value"]
    turn_id = k01["gold_evidence_turns"][0]

    transcript, speaker_roles = build_transcript_artifacts(ground_truth, schema_dir=SCHEMAS)
    assert validate_transcript_pair(ground_truth, transcript, speaker_roles, SCHEMAS) == []

    segment = next(s for s in transcript["audio_segments"] if s["id"] == turn_id)
    assert expected_value in segment["transcript"]
    assert not COMMA_SPELLED_IDENTIFIER.search(segment["transcript"])
    assert expected_value in transcript["transcripts"][0]["transcript"]


@pytest.mark.parametrize("call_id", ALL_CALLS)
def test_on_disk_transcript_matches_a_fresh_deterministic_rebuild(call_id):
    ground_truth = _ground_truth(call_id)
    on_disk_transcript = load_json(CALLS_DIR / call_id / "transcript.json")
    on_disk_roles = load_json(CALLS_DIR / call_id / "speaker_roles.json")
    assert validate(on_disk_transcript, "transcript", SCHEMAS) == []
    assert validate(on_disk_roles, "speaker_roles", SCHEMAS) == []
    assert validate_transcript_pair(ground_truth, on_disk_transcript, on_disk_roles, SCHEMAS) == []

    rebuilt_transcript, rebuilt_roles = build_transcript_artifacts(ground_truth, schema_dir=SCHEMAS)
    assert rebuilt_transcript == on_disk_transcript
    assert rebuilt_roles == on_disk_roles
