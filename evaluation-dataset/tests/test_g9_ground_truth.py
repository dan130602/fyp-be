"""G9 canonical ground truth: the FINAL (audio-derived timings) and PROVISIONAL compiles, the gold-unchanged invariant over all 50 calls,
refusal of stale or incomplete renders and inconsistent inputs, and the exporter's recognition of both modes. Offline; no export is made."""

import copy
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.ground_truths.cli import expected_outputs, gate, main
from fyp_evaluation.ground_truths.compile import compile_ground_truth
from fyp_evaluation.ground_truths.timings import recorded_render_id
from fyp_evaluation.transcripts.converter import transcript_artifacts_from_dialogue
from g5_support import RELEASE, ROOT, SCHEMAS

RENDER = "clean_voice_A"


@pytest.fixture(scope="module")
def outputs():
    return expected_outputs(RELEASE, SCHEMAS, recorded_render_id(RELEASE))


@pytest.fixture(scope="module")
def provisional():
    return expected_outputs(RELEASE, SCHEMAS, None)


def _disk(path):
    return (RELEASE / path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def copy_release(destination):
    """An isolated copy without any audio (audio is git-ignored and large; the final compile never reads it when it is absent)."""
    shutil.copytree(RELEASE, destination, ignore=shutil.ignore_patterns("*.lock", "*.mp3"))
    return destination


# ---- the committed FINAL outputs ----------------------------------------------------------------------------------------------------------------

def test_committed_outputs_are_the_final_compile_and_reproduce_byte_for_byte(outputs):
    assert recorded_render_id(RELEASE) == RENDER
    assert len(outputs) == 50 * 3 + 2
    assert {p: _disk(p) for p in outputs} == outputs
    record = json.loads(outputs["authoring/ground_truth_record.json"])
    assert record["status"] == {"g7_dialogue_verification": "VERIFIED", "timings": "AUDIO_DERIVED"} and record["timing"] == {"render_id": RENDER}


def test_actual_evidence_and_timing_invariants_for_every_call(outputs):
    checklist = [item["id"] for item in json.loads(_disk("checklist.json"))]
    for call_id in json.loads(_disk("dataset_manifest.json"))["call_ids"]:
        gt = json.loads(outputs[f"calls/{call_id}/ground_truth.json"])
        transcript = json.loads(outputs[f"calls/{call_id}/transcript.json"])
        plan = json.loads(_disk(f"authoring/check_plans/{call_id}.json"))
        timings = json.loads((RELEASE / "calls" / call_id / "renders" / RENDER / "timings.json").read_text(encoding="utf-8"))
        planned = {e["event_id"]: e["authoring"] for e in plan["planned_events"]}
        turns = gt["dialogue"]
        assert [c["question_id"] for c in gt["checks"]] == checklist
        assert gt["status"] == {"g7_dialogue_verification": "VERIFIED", "timings": "AUDIO_DERIVED"}
        assert set(gt["timing"]) == {"render_id", "timings_sha256", "render_manifest_sha256", "audio_sha256", "g7_report_sha256", "duration_seconds"}
        assert gt["timing"]["render_id"] == RENDER and gt["timing"]["duration_seconds"] == timings["duration_seconds"]
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
        assert transcript["transcripts"][0]["transcript"] == "\n\n".join(t["text"] for t in turns)
        # the times ARE the render's: same intervals as strings, in order, never overlapping, none left at the placeholder
        assert [(s["start_time"], s["end_time"]) for s in segments] == [(f"{r['start_seconds']:.3f}", f"{r['end_seconds']:.3f}") for r in timings["turns"]]
        starts, ends = [float(s["start_time"]) for s in segments], [float(s["end_time"]) for s in segments]
        assert all(e > s for s, e in zip(starts, ends)) and all(starts[i + 1] >= ends[i] for i in range(len(segments) - 1))
        assert ends[-1] == pytest.approx(timings["duration_seconds"], abs=0.001)


def test_extraction_required_evidence_is_only_what_establishes_the_final_answer(outputs):
    """The final event (plus the antecedent of a reference) is required; distractors, superseded values and false starts are not."""
    required_total = 0
    for call_id in json.loads(_disk("dataset_manifest.json"))["call_ids"]:
        gt = json.loads(outputs[f"calls/{call_id}/ground_truth.json"])
        plan = json.loads(_disk(f"authoring/check_plans/{call_id}.json"))
        events = {e["event_id"]: e for e in plan["planned_events"]}
        plan_checks = {c["question_id"]: c for c in plan["checks"]}
        for check in gt["checks"]:
            if check["kind"] != "EXTRACTION":  # compliance evidence is unchanged: every planned evidence event
                assert {s["event_id"] for s in check["gold_evidence"]} == set(plan_checks[check["question_id"]]["evidence_event_ids"])
                continue
            final = events[plan_checks[check["question_id"]]["final_event_id"]]
            expected = [final["event_id"]] + ([final["refers_to_event_id"]] if final["kind"] == "REFERENCE_SELECTION" else [])
            assert sorted(s["event_id"] for s in check["gold_evidence"]) == sorted(expected), (call_id, check["question_id"])
            required_total += 1
    assert required_total == 400
    call_1 = {c["question_id"]: c for c in json.loads(outputs["calls/CALL_001/ground_truth.json"])["checks"]}
    assert call_1["T03"]["gold_evidence_turns"] == [5]  # the illustrative amount in turn 6 is only context
    assert any(v["turn_id"] == 6 and v["status"] == "DISTRACTOR" for v in call_1["T03"]["value_history"])
    assert call_1["T04"]["gold_evidence_turns"] == [3, 15]  # "the quantity I mentioned earlier" needs its antecedent


def test_the_final_compile_leaves_gold_evidence_and_dialogue_exactly_as_the_provisional_compile(outputs, provisional):
    """The regression check for the timing step: only status, the timing provenance and the transcript times may differ."""
    assert set(outputs) == set(provisional)
    differing = {p for p in outputs if outputs[p] != provisional[p]}
    assert differing == {p for p in outputs if p.endswith(("ground_truth.json", "transcript.json"))} | {"authoring/ground_truth_record.json"}
    for path in outputs:
        if path.endswith("ground_truth.json"):
            final, early = json.loads(outputs[path]), json.loads(provisional[path])
            assert early["status"] == {"g7_dialogue_verification": "PENDING", "timings": "PLACEHOLDER_ZERO"} and "timing" not in early
            for key in ("status", "timing"):
                final.pop(key, None), early.pop(key, None)
            assert final == early  # expected answers, evidence, value histories, challenge tags, sources and dialogue are identical
        elif path.endswith("transcript.json"):
            final, early = json.loads(outputs[path]), json.loads(provisional[path])
            assert all(s["start_time"] == s["end_time"] == "0.0" for s in early["audio_segments"])
            for segment in final["audio_segments"] + early["audio_segments"]:
                segment.pop("start_time"), segment.pop("end_time")
            assert final == early
        elif path.endswith("speaker_roles.json"):
            assert outputs[path] == provisional[path]


# ---- inconsistent inputs ----------------------------------------------------------------------------------------------------------------------------

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


def test_the_converter_only_accepts_complete_ordered_non_overlapping_timings():
    dialogue = [{"turn_id": 0, "speaker": "EMPLOYEE", "text": "Hi."}, {"turn_id": 1, "speaker": "CLIENT", "text": "Hello."}]
    rows = [{"turn_id": 0, "start_seconds": 0, "end_seconds": 1.5}, {"turn_id": 1, "start_seconds": 1.9, "end_seconds": 3.25}]
    transcript, _ = transcript_artifacts_from_dialogue("CALL_999", dialogue, timings=rows)
    assert [(s["start_time"], s["end_time"]) for s in transcript["audio_segments"]] == [("0.000", "1.500"), ("1.900", "3.250")]
    for bad, message in ((rows[:1], "cover every dialogue turn"), (rows[::-1], "cover every dialogue turn"),
                         ([rows[0], {**rows[1], "start_seconds": 1.0}], "overlapping"), ([rows[0], {**rows[1], "end_seconds": 1.9}], "overlapping")):
        with pytest.raises(ValueError, match=message):
            transcript_artifacts_from_dialogue("CALL_999", dialogue, timings=bad)


# ---- refusals of the FINAL compile (isolated copies) ------------------------------------------------------------------------------------------------

def test_the_final_compile_needs_a_complete_current_render(tmp_path):
    root = copy_release(tmp_path / "dataset")
    with pytest.raises(ValueError, match="complete, current G8 render"):
        expected_outputs(root, SCHEMAS, "no_such_render")
    timings_path = root / "calls" / "CALL_005" / "renders" / RENDER / "timings.json"
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    timings["turns"].pop()
    timings_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"CALL_005: render clean_voice_A: .*(timings\.json|every dialogue turn)"):
        expected_outputs(root, SCHEMAS, RENDER)
    shutil.copy2(RELEASE / "calls" / "CALL_005" / "renders" / RENDER / "timings.json", timings_path)
    assert expected_outputs(root, SCHEMAS, RENDER) == expected_outputs(RELEASE, SCHEMAS, RENDER)  # restored: identical to the committed compile
    manifest_path = root / "calls" / "CALL_006" / "renders" / RENDER / "render_manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8").replace('"dialogue_sha256": "', '"dialogue_sha256": "0'), encoding="utf-8")
    with pytest.raises(ValueError, match="CALL_006"):
        expected_outputs(root, SCHEMAS, RENDER)


def test_the_command_line_final_provisional_and_check_flow(tmp_path, capsys):
    root = copy_release(tmp_path / "dataset")
    base = ["--dataset-root", str(root), "--schema-dir", str(SCHEMAS)]
    assert main([*base, "--check"]) == 0 and "FINAL from render clean_voice_A" in capsys.readouterr().out  # auto-detected from the record
    assert main([*base, "--write", "--overwrite"]) == 2  # a provisional write over FINAL outputs is refused
    assert "refusing to write a PROVISIONAL compile over the FINAL outputs" in capsys.readouterr().err
    (root / "authoring" / "ground_truth_record.json").unlink()  # a release that was never finalised
    assert main([*base, "--write", "--overwrite"]) == 0 and "PROVISIONAL" in capsys.readouterr().out
    assert recorded_render_id(root) is None and main([*base, "--check"]) == 0
    assert main([*base, "--write", "--overwrite", "--render-id", RENDER]) == 0 and "FINAL" in capsys.readouterr().out
    assert recorded_render_id(root) == RENDER and main([*base, "--check"]) == 0


# ---- the exporter understands both modes (no export is made to fyp-ai) -------------------------------------------------------------------------------

def test_g9_release_exports_final_and_provisional_with_their_status():
    from fyp_evaluation import exporting

    with tempfile.TemporaryDirectory(prefix="g9-export-") as outside:  # the exporter refuses destinations inside the project
        destination = Path(outside) / "advisory_execution_pilot_v1"
        manifest = exporting.export_snapshot(ROOT, destination, dataset_root=RELEASE, allow_uncommitted=True)
        assert manifest["source"]["uncommitted"] is True and manifest["source"]["dataset_id"] == "advisory_execution_pilot_v1"
        assert manifest["dataset_status"] == {"g7_dialogue_verification": "VERIFIED", "timings": "AUDIO_DERIVED"}
        assert manifest["timing_status"] == "audio_derived" and len(manifest["call_ids"]) == 50
        assert len(manifest["files"]) == 1 + 50 * 3
        assert exporting.verify_snapshot(destination)["dataset_sha256"] == manifest["dataset_sha256"]
        again = exporting.export_snapshot(ROOT, Path(outside) / "second", dataset_root=RELEASE, allow_uncommitted=True)
        assert again["dataset_sha256"] == manifest["dataset_sha256"]  # deterministic


def test_the_exporter_refuses_final_outputs_that_no_longer_match_their_render(tmp_path):
    from fyp_evaluation import exporting

    root = copy_release(tmp_path / "dataset")
    transcript = root / "calls" / "CALL_007" / "transcript.json"
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["audio_segments"][0]["end_time"] = "9.999"  # a hand-edited time
    transcript.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="g9-export-") as outside:
        with pytest.raises(ValueError, match="differs from a fresh G9 compilation"):
            exporting.export_snapshot(ROOT, Path(outside) / "x", dataset_root=root, allow_uncommitted=True)


def test_an_extraction_check_whose_final_answer_cannot_be_verified_is_refused_not_guessed():
    dialogue = _disk("authoring/dialogues/CALL_001.json")
    report = _disk("authoring/plan_verifications/CALL_001.json")

    def refused(mutate, match):
        inputs = gate(RELEASE, SCHEMAS)
        plan = copy.deepcopy(inputs.plans["CALL_001"])
        mutate(plan)
        inputs.plans["CALL_001"] = plan
        with pytest.raises(ValueError, match=match):
            compile_ground_truth(inputs, "CALL_001", dialogue, report)

    def check(plan, question_id):
        return next(c for c in plan["checks"] if c["question_id"] == question_id)

    def event(plan, event_id):
        return next(e for e in plan["planned_events"] if e["event_id"] == event_id)

    refused(lambda plan: check(plan, "T03").update(final_event_id=None), "needs a final event")
    refused(lambda plan: check(plan, "T03").update(final_event_id="EV_014"), "is not the final answer")
    refused(lambda plan: event(plan, "EV_013")["authoring"].update(value_mentions=[]), "does not explicitly mention")
    refused(lambda plan: event(plan, "EV_017").update(refers_to_event_id="EV_015"), "does not point to an explicit mention")
    refused(lambda plan: check(plan, "T03").update(evidence_event_ids=["EV_014", "EV_015"]), "not part of the planned evidence")
