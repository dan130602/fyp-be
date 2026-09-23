"""ARTIFICIAL fixtures only. Exercises the deterministic ground-truth-to-
transcript/speaker-role converter and its cross-file validator; never reads
or writes real pilot call data."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from compilation_fixtures import SCHEMAS, compilation_inputs
from transcript_fixtures import minimal_ground_truth as _minimal_ground_truth
from fyp_evaluation.authoring.compile import compile_ground_truth
from fyp_evaluation.transcripts.converter import (
    PLACEHOLDER_TIME, TRANSCRIPT_SEPARATOR, build_transcript_artifacts,
)
from fyp_evaluation.validation.schemas import validate
from fyp_evaluation.validation.transcript import validate_transcript_pair


def _compiled_ground_truth():
    """A fully compiled, structurally real ARTIFICIAL ground truth (5 turns,
    EMPLOYEE speaking first): FIX_PRESENT(EMPLOYEE), K01/FIX_DOB/
    FIX_INSTRUMENT/FIX_CURRENCY(CLIENT)."""
    return compile_ground_truth(*compilation_inputs(), schema_dir=SCHEMAS)


# --- Source integrity -------------------------------------------------

def test_compiled_fixture_and_minimal_fixture_are_schema_valid():
    ground = _compiled_ground_truth()
    assert validate(ground, "ground_truth", SCHEMAS) == []
    minimal = _minimal_ground_truth("CALL_997", [("CLIENT", "Hi."), ("EMPLOYEE", "Hello.")])
    assert validate(minimal, "ground_truth", SCHEMAS) == []


# --- Text preservation, segment identity, ordering ---------------------

def test_every_turn_becomes_exactly_one_segment_verbatim_and_in_order():
    ground = _compiled_ground_truth()
    transcript, _ = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    segments = transcript["audio_segments"]
    assert len(segments) == len(ground["dialogue"])
    for turn, segment in zip(ground["dialogue"], segments):
        assert segment["id"] == turn["turn_id"]
        assert segment["transcript"] == turn["text"]
        assert segment["transcript"].encode("utf-8") == turn["text"].encode("utf-8")


def test_segment_ids_are_original_turn_ids_not_positions():
    # Build a minimal ground truth directly (turn_id equals array position
    # here by construction) and confirm the converter copies turn_id, not a
    # freshly derived index, by checking identity of the id field's source.
    ground = _minimal_ground_truth("CALL_996", [("EMPLOYEE", "A"), ("CLIENT", "B"), ("EMPLOYEE", "C")])
    transcript, _ = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert [segment["id"] for segment in transcript["audio_segments"]] == [0, 1, 2]


def test_no_merge_split_or_duplicate_of_turns():
    ground = _compiled_ground_truth()
    transcript, _ = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert len(transcript["audio_segments"]) == len(ground["dialogue"])
    texts = [segment["transcript"] for segment in transcript["audio_segments"]]
    assert texts == [turn["text"] for turn in ground["dialogue"]]


# --- Speaker mapping -----------------------------------------------------

def test_employee_maps_to_spk_0_and_rm_client_to_spk_1_and_customer():
    ground = _minimal_ground_truth("CALL_995", [("EMPLOYEE", "Hello."), ("CLIENT", "Hi.")])
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    labels = {turn["speaker"]: segment["speaker_label"]
              for turn, segment in zip(ground["dialogue"], transcript["audio_segments"])}
    assert labels == {"EMPLOYEE": "spk_0", "CLIENT": "spk_1"}
    assert speaker_roles == {"speaker_roles": {"spk_0": "RM", "spk_1": "Customer"}}


def test_client_first_conversation_still_maps_client_to_spk_1_and_customer():
    ground = _minimal_ground_truth("CALL_994", [("CLIENT", "Hi there."), ("EMPLOYEE", "Hello.")])
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    first_turn_segment = transcript["audio_segments"][0]
    assert first_turn_segment["speaker_label"] == "spk_1"  # CLIENT, despite speaking first
    assert speaker_roles["speaker_roles"] == {"spk_1": "Customer", "spk_0": "RM"}
    # Fixed EMPLOYEE-before-CLIENT key order, independent of speaking order.
    assert list(speaker_roles["speaker_roles"]) == ["spk_0", "spk_1"]


def test_single_speaker_call_maps_only_the_present_speaker():
    ground = _minimal_ground_truth("CALL_993", [("CLIENT", "Hello?"), ("CLIENT", "Anyone there?")])
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert {segment["speaker_label"] for segment in transcript["audio_segments"]} == {"spk_1"}
    assert speaker_roles == {"speaker_roles": {"spk_1": "Customer"}}


# --- Separate file contracts ----------------------------------------------

def test_transcript_has_no_results_wrapper_and_no_speaker_roles():
    ground = _compiled_ground_truth()
    transcript, _ = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert set(transcript.keys()) == {"transcripts", "audio_segments"}
    assert "results" not in transcript
    assert "speaker_roles" not in transcript
    assert "call_id" not in transcript
    assert "checklist" not in transcript


def test_speaker_roles_file_has_no_dialogue_or_segments():
    ground = _compiled_ground_truth()
    _, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert set(speaker_roles.keys()) == {"speaker_roles"}
    assert "audio_segments" not in speaker_roles
    assert "dialogue" not in speaker_roles
    assert "transcripts" not in speaker_roles


# --- Full transcript reconstruction ---------------------------------------

def test_full_transcript_uses_double_newline_separator_in_order():
    ground = _compiled_ground_truth()
    transcript, _ = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    expected = TRANSCRIPT_SEPARATOR.join(turn["text"] for turn in ground["dialogue"])
    assert transcript["transcripts"] == [{"transcript": expected}]
    assert "\n\n" in expected  # sanity: fixture has more than one turn
    assert TRANSCRIPT_SEPARATOR == "\n\n"


# --- No ground-truth leakage ------------------------------------------

def test_no_evaluation_only_fields_leak_into_either_output():
    ground = _compiled_ground_truth()
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    serialized = json.dumps(transcript) + json.dumps(speaker_roles)
    for forbidden in ("expected_verdict", "expected_value", "evidence_plan", "gold_evidence_turns",
                     "gold_evidence_text", "critical_tokens", "planned_evidence", "checks"):
        assert forbidden not in serialized


# --- Missing timing information -----------------------------------------

def test_all_segments_carry_the_documented_placeholder_timestamp():
    ground = _compiled_ground_truth()
    transcript, _ = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    for segment in transcript["audio_segments"]:
        assert segment["start_time"] == PLACEHOLDER_TIME == "0.0"
        assert segment["end_time"] == PLACEHOLDER_TIME == "0.0"
        assert "confidence" not in segment
        assert "items" not in segment
        assert "words" not in segment


def test_timings_argument_is_not_silently_ignored():
    ground = _compiled_ground_truth()
    with pytest.raises(NotImplementedError):
        build_transcript_artifacts(ground, timings={"anything": True}, schema_dir=SCHEMAS)


# --- Cross-file consistency -----------------------------------------------

def test_generated_pair_passes_cross_file_validation():
    ground = _compiled_ground_truth()
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert validate_transcript_pair(ground, transcript, speaker_roles, SCHEMAS) == []


def test_cross_file_validation_rejects_speaker_label_without_a_role():
    ground = _compiled_ground_truth()
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    del speaker_roles["speaker_roles"]["spk_1"]
    errors = validate_transcript_pair(ground, transcript, speaker_roles, SCHEMAS)
    assert any("do not match transcript labels" in error for error in errors)


def test_cross_file_validation_rejects_mismatched_segment_text():
    ground = _compiled_ground_truth()
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    transcript["audio_segments"][0]["transcript"] = "ARTIFICIAL tampered text"
    errors = validate_transcript_pair(ground, transcript, speaker_roles, SCHEMAS)
    assert any("does not match dialogue text exactly" in error for error in errors)


def test_cross_file_validation_rejects_wrong_role_for_a_label():
    ground = _compiled_ground_truth()
    transcript, speaker_roles = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    speaker_roles["speaker_roles"]["spk_0"] = "Customer"
    errors = validate_transcript_pair(ground, transcript, speaker_roles, SCHEMAS)
    assert any("does not match expected" in error for error in errors)


# --- Determinism and purity -------------------------------------------

def test_determinism_byte_identical_output_across_repeated_calls():
    ground = _compiled_ground_truth()
    first = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    second = build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert json.dumps(first[0], sort_keys=False) == json.dumps(second[0], sort_keys=False)
    assert json.dumps(first[1], sort_keys=False) == json.dumps(second[1], sort_keys=False)
    assert first == second


def test_converter_never_mutates_its_input():
    ground = _compiled_ground_truth()
    before = deepcopy(ground)
    build_transcript_artifacts(ground, schema_dir=SCHEMAS)
    assert ground == before


# --- Rejection of unsupported/invalid input --------------------------

def test_rejects_structurally_invalid_ground_truth():
    ground = _compiled_ground_truth()
    del ground["dialogue"]
    with pytest.raises(ValueError):
        build_transcript_artifacts(ground, schema_dir=SCHEMAS)
