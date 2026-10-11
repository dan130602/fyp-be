"""G8 canonical ground truth: reproducibility, actual-evidence invariants over all 50 calls, and refusal of inconsistent inputs."""

import copy
import json
from pathlib import Path

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.ground_truths.cli import expected_outputs, gate
from fyp_evaluation.ground_truths.compile import compile_ground_truth
from g5_support import RELEASE, SCHEMAS


@pytest.fixture(scope="module")
def outputs():
    return expected_outputs(RELEASE, SCHEMAS)


def _disk(path):
    return (RELEASE / path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_committed_outputs_reproduce_byte_for_byte(outputs):
    assert len(outputs) == 50 * 3 + 2
    assert {p: _disk(p) for p in outputs} == outputs


def test_actual_evidence_invariants_for_every_call(outputs):
    checklist = [item["id"] for item in json.loads(_disk("checklist.json"))]
    for call_id in json.loads(_disk("dataset_manifest.json"))["call_ids"]:
        gt = json.loads(outputs[f"calls/{call_id}/ground_truth.json"])
        transcript = json.loads(outputs[f"calls/{call_id}/transcript.json"])
        plan = json.loads(_disk(f"authoring/check_plans/{call_id}.json"))
        planned = {e["event_id"]: e["authoring"] for e in plan["planned_events"]}
        turns = gt["dialogue"]
        assert [c["question_id"] for c in gt["checks"]] == checklist
        assert gt["status"] == {"g7_dialogue_verification": "PENDING", "timings": "PLACEHOLDER_ZERO"}
        for check in gt["checks"]:
            for span in check["gold_evidence"]:
                assert turns[span["turn_id"]]["text"][span["start"]:span["end"]] == span["text"] == planned[span["event_id"]]["planned_text"]
                assert turns[span["turn_id"]]["speaker"] == span["speaker"]
            assert check["gold_evidence_turns"] == sorted({s["turn_id"] for s in check["gold_evidence"]})
            if check["evidence_mode"] == "ABSENCE":
                assert check["gold_evidence"] == [] and check["gold_evidence_text"] == ""
            if check["kind"] == "EXTRACTION" and check["final_turn_id"] is not None:
                finals = [v for v in check["value_history"] if v["status"] == "FINAL"]
                assert finals and finals[-1]["value"] == check["expected"]["expected_value"]
                assert check["final_turn_id"] in check["gold_evidence_turns"]
        segments = transcript["audio_segments"]
        assert [(s["id"], s["transcript"]) for s in segments] == [(t["turn_id"], t["text"]) for t in turns]
        assert all(s["start_time"] == s["end_time"] == "0.0" for s in segments)
        assert transcript["transcripts"][0]["transcript"] == "\n\n".join(t["text"] for t in turns)


def test_a_dialogue_span_that_differs_from_the_plan_is_refused():
    inputs = gate(RELEASE, SCHEMAS)
    dialogue = parse_json_text(_disk("authoring/dialogues/CALL_021.json"))
    report = _disk("authoring/plan_verifications/CALL_021.json")
    tampered = copy.deepcopy(dialogue)
    alignment = tampered["event_alignment"][0]
    turn = tampered["turns"][alignment["turn_id"]]
    turn["text"] = turn["text"][:alignment["start"]] + "X" + turn["text"][alignment["start"] + 1:]
    with pytest.raises(ValueError, match="differs from its planned text"):
        compile_ground_truth(inputs, "CALL_021", serialize_json(tampered), report)
    missing = copy.deepcopy(dialogue)
    missing["event_alignment"].pop()
    with pytest.raises(ValueError, match="does not align every planned event"):
        compile_ground_truth(inputs, "CALL_021", serialize_json(missing), report)


def test_g8_release_exports_provisionally_and_records_its_status():
    import tempfile

    from fyp_evaluation import exporting
    from g5_support import ROOT

    with tempfile.TemporaryDirectory(prefix="g8-export-") as outside:  # the exporter refuses destinations inside the project
        destination = Path(outside) / "advisory_execution_pilot_v1"
        manifest = exporting.export_snapshot(ROOT, destination, dataset_root=RELEASE, allow_uncommitted=True)
        assert manifest["source"]["uncommitted"] is True and manifest["source"]["dataset_id"] == "advisory_execution_pilot_v1"
        assert manifest["dataset_status"] == {"g7_dialogue_verification": "PENDING", "timings": "PLACEHOLDER_ZERO"}
        assert manifest["timing_status"] == "placeholder_zero" and len(manifest["call_ids"]) == 50
        assert len(manifest["files"]) == 1 + 50 * 3
        assert exporting.verify_snapshot(destination)["dataset_sha256"] == manifest["dataset_sha256"]
        again = exporting.export_snapshot(ROOT, Path(outside) / "second", dataset_root=RELEASE, allow_uncommitted=True)
        assert again["dataset_sha256"] == manifest["dataset_sha256"]  # deterministic
