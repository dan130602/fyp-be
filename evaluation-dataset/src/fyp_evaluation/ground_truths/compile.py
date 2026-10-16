"""G9: deterministic compilation of the canonical per-call ground truth (no LLM, no network).

Inputs (all already accepted and validated upstream): the frozen gold (G2A call spec), the accepted G4 check plan and the
accepted G6 dialogue. The check plan holds PLANNED evidence; the ground truth holds ACTUAL evidence: for every evidence event
the G6 event alignment gives the turn and the exact character span, and Python re-checks that the span equals the event's
planned text. Extraction questions also get their value history (every turn that states a value for the question, with the
event status) and every question gets challenge tags from its assigned edge cases. The REQUIRED evidence of an extraction question
is only what establishes the final answer (see `_required_extraction_events`); distractors, superseded values, rejected readbacks
and false starts stay in the value history and the dialogue but are not required evidence. Expected answers are copied from the
frozen gold, never derived from the dialogue.

Each call yields `calls/CALL_XXX/ground_truth.json`, `transcript.json` and `speaker_roles.json` (the transcript pair comes
from the shared transcript converter). Without `timing` the compile is PROVISIONAL (placeholder "0.0" times, G7 PENDING). With `timing`
(the audio-derived per-turn times of a verified G8 render, see timings.py) it is FINAL: the times go into `transcript.json` only, and
`ground_truth.json` records the status and the timing provenance. Expected answers, evidence, value histories, challenge tags and the
dialogue are identical in both modes.
"""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.transcripts.converter import transcript_artifacts_from_dialogue

STAGE, SCHEMA_VERSION, GENERATOR_VERSION = "G9", "1.0", "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"
CALLS_DIR = "calls"
RECORD = "authoring/ground_truth_record.json"
DIALOGUE_DIR, REPORT_DIR = "authoring/dialogues", "authoring/plan_verifications"
CALL_FILES = ("ground_truth.json", "transcript.json", "speaker_roles.json")


def schema_path(schema_dir):
    return Path(schema_dir) / "stage9" / "ground_truth.schema.json"


def challenge_tag(edge_case_id):
    return edge_case_id.removeprefix("EC_").lower()


def _span(event, alignment, turns):
    turn = turns[alignment["turn_id"]]
    text = turn["text"][alignment["start"]:alignment["end"]]
    if text != event["authoring"]["planned_text"]:
        raise ValueError(f"{event['event_id']}: dialogue span differs from its planned text")
    if turn["speaker"] != event["authoring"]["speaker"]:
        raise ValueError(f"{event['event_id']}: spoken by {turn['speaker']}, planned for {event['authoring']['speaker']}")
    return {"event_id": event["event_id"], "turn_id": turn["turn_id"], "speaker": turn["speaker"], "start": alignment["start"],
            "end": alignment["end"], "text": text}


def _value_history(question_id, events, spans):
    """Every event that states a value for the question (readback errors included), in dialogue order."""
    history = []
    for event in events:
        if "values" in event and question_id in event["values"]:
            value = event["values"][question_id]["value"]
        elif (event.get("readback") or {}).get("wrong_question_id") == question_id:
            value = event["readback"]["wrong_value"]
        else:
            continue
        span = spans[event["event_id"]]
        history.append({"turn_id": span["turn_id"], "event_id": event["event_id"], "speaker": span["speaker"], "kind": event["kind"],
                        "status": event["status"], "value": value, "_order": (span["turn_id"], span["start"])})
    history.sort(key=lambda entry: entry["_order"])
    return [{key: value for key, value in entry.items() if key != "_order"} for entry in history]


# Final-event kinds whose own turn explicitly states the value (a reference needs its antecedent as well).
EXPLICIT_FINAL_KINDS = frozenset({"VALUE_STATEMENT", "VALUE_CORRECTION", "ORDER_STATEMENT"})


def _mentions_value(event, question_id, value):
    return any(m["question_id"] == question_id and m["value"] == value for m in event["authoring"].get("value_mentions", []))


def _required_extraction_events(call_id, question_id, check, events_by_id, expected_value):
    """The planned events a model must cite to answer the extraction question: the final event and, for a reference, its antecedent.

    Deterministic: nothing is guessed. Anything the rule cannot verify raises instead of falling back to the whole planned set.
    """
    where = f"{call_id}/{question_id}"
    final_id = check.get("final_event_id")
    if not final_id:
        raise ValueError(f"{where}: an extraction check needs a final event to derive its required evidence")
    final = events_by_id[final_id]
    if final["status"] != "FINAL" or final["values"][question_id]["value"] != expected_value:
        raise ValueError(f"{where}: final event {final_id} is not the final answer {expected_value!r}")
    if final["kind"] in EXPLICIT_FINAL_KINDS:
        if not _mentions_value(final, question_id, expected_value):
            raise ValueError(f"{where}: final event {final_id} does not explicitly mention the expected value")
        required = [final_id]
    elif final["kind"] == "REFERENCE_SELECTION":
        antecedent = events_by_id[final["refers_to_event_id"]]
        if antecedent["kind"] != "ANTECEDENT_MENTION" or not _mentions_value(antecedent, question_id, expected_value):
            raise ValueError(f"{where}: reference {final_id} does not point to an explicit mention of the expected value")
        required = [antecedent["event_id"], final_id]
    else:
        raise ValueError(f"{where}: unsupported final event kind {final['kind']}")
    if not set(required) <= set(check["evidence_event_ids"]):
        raise ValueError(f"{where}: required evidence {required} is not part of the planned evidence")
    return required


def compile_ground_truth(inputs, call_id, dialogue_text, report_text, timing=None):
    """The canonical ground-truth dict for one call (raises on any inconsistency; never repairs)."""
    dialogue = parse_json_text(dialogue_text)
    spec, plan = inputs.specs[call_id], inputs.plans[call_id]
    if dialogue["check_plan_sha256"] != inputs.plan_hashes[call_id]:
        raise ValueError(f"{call_id}: the dialogue was written for a different check plan")
    turns = dialogue["turns"]
    if [t["turn_id"] for t in turns] != list(range(len(turns))):
        raise ValueError(f"{call_id}: turn IDs are not contiguous from 0")
    events = plan["planned_events"]
    alignment = {a["event_id"]: a for a in dialogue["event_alignment"]}
    if set(alignment) != {e["event_id"] for e in events}:
        raise ValueError(f"{call_id}: the dialogue does not align every planned event exactly once")
    spans = {e["event_id"]: _span(e, alignment[e["event_id"]], turns) for e in events}
    gold = {entry["question_id"]: entry["expected"] for entry in spec["gold"]}
    plan_checks = {c["question_id"]: c for c in plan["checks"]}
    events_by_id = {e["event_id"]: e for e in events}
    checks = []
    for item in inputs.documents["checklist"]:
        question_id = item["id"]
        check = plan_checks[question_id]
        if check["kind"] == "EXTRACTION":
            evidence_ids = _required_extraction_events(call_id, question_id, check, events_by_id, gold[question_id]["expected_value"])
        else:
            evidence_ids = check["evidence_event_ids"]
        evidence = sorted((spans[e] for e in evidence_ids), key=lambda s: (s["turn_id"], s["start"]))
        final_event = check.get("final_event_id")
        history = _value_history(question_id, events, spans) if check["kind"] == "EXTRACTION" else []
        expected = gold[question_id]
        if final_event and expected["kind"] == "extraction":
            stated = next(e for e in events if e["event_id"] == final_event)["values"][question_id]["value"]
            if stated != expected["expected_value"]:
                raise ValueError(f"{call_id}/{question_id}: final event value {stated!r} differs from gold {expected['expected_value']!r}")
        tags = []
        for edge in spec["edge_cases"]:
            if question_id in edge["target_question_ids"] and challenge_tag(edge["edge_case_id"]) not in tags:
                tags.append(challenge_tag(edge["edge_case_id"]))
        checks.append({
            "question_id": question_id, "kind": check["kind"], "expected": expected, "evidence_mode": check["evidence_mode"],
            "presence_pattern": check.get("presence_pattern"), "absence_scope": (check.get("absence") or {}).get("scope"),
            "gold_evidence_turns": sorted({s["turn_id"] for s in evidence}), "gold_evidence": evidence,
            "gold_evidence_text": "\n".join(s["text"] for s in evidence),
            "final_turn_id": spans[final_event]["turn_id"] if final_event else None,
            "value_history": history, "challenge_tags": tags,
        })
    source = inputs.upstream.source_hashes()
    final = timing is not None
    ground_truth = {
        "schema_version": SCHEMA_VERSION, "stage": STAGE, "dataset_id": inputs.dataset_id, "call_id": call_id,
        "status": {"g7_dialogue_verification": "VERIFIED", "timings": "AUDIO_DERIVED"} if final
        else {"g7_dialogue_verification": "PENDING", "timings": "PLACEHOLDER_ZERO"},
        "source": {"call_spec_sha256": inputs.spec_hashes[call_id], "check_plan_skeleton_sha256": inputs.skeleton_hashes[call_id],
                   "call_config_sha256": inputs.config_hashes[call_id], "check_plan_sha256": inputs.plan_hashes[call_id],
                   "plan_verification_sha256": text_hash(report_text), "dialogue_sha256": text_hash(dialogue_text), **source},
        "metadata": spec["metadata"],
        "checks": checks,
        "dialogue": [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"]} for t in turns],
    }
    if final:
        ground_truth["timing"] = dict(timing["provenance"])
    return ground_truth


def build_call_files(inputs, call_id, dialogue_text, report_text, schema_dir, timing=None):
    """{file name: canonical text} for one call, every document schema-validated. `timing` makes it the FINAL compile."""
    ground_truth = compile_ground_truth(inputs, call_id, dialogue_text, report_text, timing)
    validator = Draft202012Validator(json.loads(schema_path(schema_dir).read_text(encoding="utf-8")))
    if errors := [f"{e.json_path}: {e.message}" for e in validator.iter_errors(ground_truth)]:
        raise ValueError(f"{call_id}: ground truth fails its schema: {errors[:5]}")
    transcript, speaker_roles = transcript_artifacts_from_dialogue(call_id, ground_truth["dialogue"], schema_dir=schema_dir,
                                                                   timings=timing["turns"] if timing else None)
    return {"ground_truth.json": serialize_json(ground_truth), "transcript.json": serialize_json(transcript),
            "speaker_roles.json": serialize_json(speaker_roles)}


def build_record(inputs, files_by_call, render_id=None):
    calls = {call_id: {name: text_hash(text) for name, text in files.items()} for call_id, files in files_by_call.items()}
    record = {"schema_version": SCHEMA_VERSION, "stage": STAGE, "dataset_id": inputs.dataset_id,
              "generator": {"package": "fyp_evaluation.ground_truths", "version": GENERATOR_VERSION}, "hashing": HASHING,
              "status": {"g7_dialogue_verification": "VERIFIED", "timings": "AUDIO_DERIVED"} if render_id
              else {"g7_dialogue_verification": "PENDING", "timings": "PLACEHOLDER_ZERO"},
              "calls": calls, "summary": {"calls": len(calls)}}
    if render_id:  # the render the timings came from; the per-call provenance lives in each ground_truth.json
        record["timing"] = {"render_id": render_id}
    return record
