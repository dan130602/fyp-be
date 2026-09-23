"""ARTIFICIAL evidence fixtures, independent of token annotations."""

from copy import deepcopy

import pytest

from fixtures import documents
from compilation_fixtures import SCHEMAS, compilation_inputs
from fyp_evaluation.validation.evidence import resolve_evidence, validate_dialogue_against_plan
from fyp_evaluation.validation.schemas import validate


def evidence(*texts):
    return {"mode": "presence", "planned_evidence": [
        {"evidence_id": f"FIX_E{i}", "required_speaker": "EMPLOYEE", "text": text}
        for i, text in enumerate(texts, 1)], "content_constraints": [], "forbidden_patterns": []}


def turn(text, speaker="EMPLOYEE"):
    # Deliberately omit annotations: evidence helpers may not depend on them.
    return {"speaker": speaker, "text": text}


@pytest.mark.parametrize("turns,code,candidates", [
    ([], "not_found", []), ([turn("ARTIFICIAL other")], "not_found", []),
    ([turn("marker", "CLIENT")], "not_found", []),
    ([turn("Marker")], "not_found", []),
    ([turn("marker"), turn("marker")], "ambiguous", [0, 1]),
])
def test_match_errors(turns, code, candidates):
    ids, errors = resolve_evidence(evidence("marker"), turns)
    assert ids == []
    assert errors[0]["code"] == code
    assert errors[0]["matching_turn_ids"] == candidates
    assert code.replace("_", " ") in errors[0]["message"]


@pytest.mark.parametrize("phrases,turns,wanted", [
    (("marker",), [turn("marker marker")], [0]),
    (("first", "second"), [turn("first"), turn("second")], [0, 1]),
    (("first", "second"), [turn("first and second")], [0, 0]),
    (("second", "first"), [turn("first"), turn("second")], [1, 0]),
    (("marker",), [turn("marker", "CLIENT"), turn("marker")], [1]),
])
def test_resolved_ids_preserve_plan_order(phrases, turns, wanted):
    assert resolve_evidence(evidence(*phrases), turns) == (wanted, [])


def test_item_specific_speakers():
    plan = evidence("prompt", "response")
    plan["planned_evidence"][1]["required_speaker"] = "CLIENT"
    assert resolve_evidence(plan, [turn("prompt"), turn("response", "CLIENT")]) == ([0, 1], [])


def test_dialogue_identity_before_accessing_any_other_field():
    errors = validate_dialogue_against_plan({"call_id": "CALL_999"}, {"call_id": "CALL_998"})
    assert len(errors) == 1 and "CALL_999" in errors[0] and "CALL_998" in errors[0]


def test_individually_valid_artifacts_with_different_ids_fail():
    _, plan, dialogue, *_ = compilation_inputs()
    plan["call_id"] = "CALL_998"
    assert not validate(plan, "check_plan", SCHEMAS)
    assert not validate(dialogue, "dialogue", SCHEMAS)
    assert validate_dialogue_against_plan(dialogue, plan)


def test_plan_order_must_be_nondecreasing_but_same_turn_is_allowed():
    plan = {"call_id": "CALL_999", "checks": [{"question_id": "FIX", "evidence_plan": evidence("second", "first")}]}
    dialogue = {"call_id": "CALL_999", "turns": [turn("first"), turn("second")]}
    assert "order" in validate_dialogue_against_plan(dialogue, plan)[0]
    dialogue["turns"] = [turn("first and second")]
    assert validate_dialogue_against_plan(dialogue, plan) == []


@pytest.mark.parametrize("mode,speaker,hit_speaker,rejected", [
    ("absence", "EMPLOYEE", "EMPLOYEE", True), ("absence", "EMPLOYEE", "CLIENT", False),
    ("absence", "CLIENT", "EMPLOYEE", False), ("absence", "ANY", "CLIENT", True),
    ("presence", None, "CLIENT", True), ("presence", None, "EMPLOYEE", True),
])
def test_forbidden_pattern_scope(mode, speaker, hit_speaker, rejected):
    plan = evidence("marker") if mode == "presence" else {
        "mode": "absence", "required_speaker": speaker, "scope": "entire_dialogue",
        "absent_event": "ARTIFICIAL forbidden marker", "content_constraints": []}
    plan["forbidden_patterns"] = ["forbidden"]
    dialogue = {"call_id": "CALL_999", "turns": [turn("marker"), turn("forbidden", hit_speaker)]}
    checks = {"call_id": "CALL_999", "checks": [{"question_id": "FIX", "evidence_plan": plan}]}
    assert bool(validate_dialogue_against_plan(dialogue, checks)) is rejected
    plan["forbidden_patterns"] = []
    assert validate_dialogue_against_plan(dialogue, checks) == []


def test_invalid_regex_is_reported():
    _, plan, dialogue, *_ = compilation_inputs()
    plan["checks"][0]["evidence_plan"]["forbidden_patterns"] = ["["]
    assert "invalid forbidden pattern" in validate_dialogue_against_plan(dialogue, plan)[0]


def test_legacy_integrity_keeps_ambiguous_union_and_message():
    ground = documents()["ground_truth"]
    ground["dialogue"].append({**deepcopy(ground["dialogue"][0]), "turn_id": 1})
    ground["checks"][0]["gold_evidence_turns"] = [0, 1]
    assert validate(ground, "ground_truth", SCHEMAS) == ["Planned evidence must match exactly one eligible turn"]
    ground["checks"][0]["gold_evidence_turns"] = [0]
    assert validate(ground, "ground_truth", SCHEMAS) == [
        "Planned evidence must match exactly one eligible turn", "gold_evidence_turns disagrees with planned evidence"]
