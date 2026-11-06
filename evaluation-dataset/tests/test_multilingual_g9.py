"""Phase 3B: G9 v2 (source schema only) and the shared answer-support rule on the multilingual release (offline, deterministic)."""

import json

import pytest
from jsonschema import Draft202012Validator

import multilingual_support as M
from fyp_evaluation import answer_support
from fyp_evaluation.ground_truths.cli import expected_outputs, main

SCHEMA_V1 = json.loads((M.SCHEMAS / "stage9/ground_truth.schema.json").read_text(encoding="utf-8"))
SCHEMA_V2 = json.loads((M.SCHEMAS / "stage9/ground_truth.v2.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def world(verified250_root, tmp_path_factory):
    root, _ = verified250_root
    return M.copy_chain(root, tmp_path_factory.mktemp("g9world") / "dataset")


@pytest.fixture(scope="module")
def outputs(world):
    return expected_outputs(world.root, M.SCHEMAS)


def test_the_provisional_compile_covers_all_250_calls_with_v2_ground_truths(outputs, world):
    truths = {path: json.loads(text) for path, text in outputs.items() if path.endswith("ground_truth.json")}
    assert len(truths) == 250 and len([p for p in outputs if p.endswith("transcript.json")]) == 250
    specs = M.upstream_inputs().specs
    for path, truth in truths.items():
        call_id = truth["call_id"]
        profile = specs[call_id]["metadata"]["language_profile"]
        assert truth["schema_version"] == "2.0" and truth["source"]["language_profiles_sha256"], path
        assert ("register_guide_sha256" in truth["source"]) is (profile == "SINGLISH"), path
        assert truth["metadata"]["language_profile"] == profile and truth["status"] == {"g7_dialogue_verification": "PENDING", "timings": "PLACEHOLDER_ZERO"}
        assert not list(Draft202012Validator(SCHEMA_V2).iter_errors(truth)), path
        gold = {e["question_id"]: e["expected"] for e in specs[call_id]["gold"]}
        assert {c["question_id"]: c["expected"] for c in truth["checks"]} == gold, "expected answers are copied from the frozen gold"


def test_only_the_source_block_differs_from_the_historical_schema(outputs):
    truth = next(json.loads(text) for path, text in outputs.items() if path.endswith("ground_truth.json"))
    assert list(Draft202012Validator(SCHEMA_V1).iter_errors(truth)), "the historical schema rejects the v2 document"
    assert SCHEMA_V1["properties"]["checks"] == SCHEMA_V2["properties"]["checks"] and SCHEMA_V1["properties"]["dialogue"] == SCHEMA_V2["properties"]["dialogue"]
    assert set(SCHEMA_V2["properties"]["source"]["properties"]) - set(SCHEMA_V1["properties"]["source"]["properties"]) == {
        "language_profiles_sha256", "register_guide_sha256"}


def test_write_then_check_reproduces_byte_for_byte_and_the_pilot_schema_name_is_unchanged(world, capsys):
    argv = ["--dataset-root", str(world.root), "--schema-dir", str(M.SCHEMAS)]
    assert main([*argv, "--write"]) == 0
    assert main([*argv, "--check"]) == 0 and "reproduce byte-for-byte" in capsys.readouterr().out


def test_every_extraction_answer_is_fully_supported_by_its_required_events_in_all_250_plans(chain250_root):
    """Answer support (shared with G9): the final event, and for a reference its antecedent, state the expected value in full."""
    root, _ = chain250_root
    specs = M.upstream_inputs().specs
    seen = 0
    for call_id in specs:
        plan = json.loads((root / "authoring/check_plans" / f"{call_id}.json").read_text(encoding="utf-8"))
        events = {e["event_id"]: e for e in plan["planned_events"]}
        gold = {g["question_id"]: g["expected"] for g in specs[call_id]["gold"]}
        for check in plan["checks"]:
            if check["kind"] != "EXTRACTION":
                continue
            value = gold[check["question_id"]]["expected_value"]
            required = answer_support.required_extraction_events(call_id, check["question_id"], check, events, value)
            assert required and all(r in check["evidence_event_ids"] for r in required)
            carrier = events[required[0]]  # the final event, or for a reference its antecedent: the explicit mention of the value
            assert answer_support.mentions_value(carrier, check["question_id"], value), (call_id, check["question_id"])
            seen += 1
    assert seen > 250 * 4


def test_value_history_orders_every_stated_value_for_a_question(world):
    dialogue = json.loads((world.root / "authoring/dialogues/CALL_001.json").read_text(encoding="utf-8"))
    plan = json.loads((world.root / "authoring/check_plans/CALL_001.json").read_text(encoding="utf-8"))
    spans = {a["event_id"]: {"turn_id": a["turn_id"], "start": a["start"], "speaker": dialogue["turns"][a["turn_id"]]["speaker"]}
             for a in dialogue["event_alignment"]}
    history = answer_support.value_history("T04", plan["planned_events"], spans)
    assert history and history == sorted(history, key=lambda h: (h["turn_id"], spans[h["event_id"]]["start"]))


def test_a_tampered_dialogue_is_refused_not_compiled(world, tmp_path):
    copy = M.copy_chain(world.root, tmp_path / "dataset")
    path = copy.root / "authoring/dialogues/CALL_002.json"
    dialogue = json.loads(path.read_text(encoding="utf-8"))
    span = dialogue["event_alignment"][0]
    turn = dialogue["turns"][span["turn_id"]]
    turn["text"] = turn["text"][:span["start"]] + "X" + turn["text"][span["start"] + 1:]  # truncate/alter the planned text inside its span
    path.write_text(json.dumps(dialogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        expected_outputs(copy.root, M.SCHEMAS)
