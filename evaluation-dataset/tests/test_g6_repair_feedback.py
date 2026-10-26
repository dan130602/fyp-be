"""Optional G6 repair feedback: guidance derived from a call's INVALID G7 report goes into that call's request brief and is recorded on its
attempt; nothing else (acceptance rules, other calls' requests, artifacts) changes. Offline; isolated tmp copies."""

import json

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.dialogues.cli import main
from fyp_evaluation.dialogues.request import MAX_GUIDANCE_CHARS, RequestBuilder, load_repair_feedback
from fyp_evaluation.planning import outputs as g1
from g6_support import SCHEMAS, Clock, make_runner, shared_inputs, submit_and_finish

X, Y = "CALL_016", "CALL_027"
PROMPTS = SCHEMAS.parent / "prompts" / "stage6"
GUIDANCE = ["In your own words, never state a transaction fee or charge, including a rate such as a percentage.",
            "Keep the closing turns to scheduling and thanks only."]


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def g7_report(root, call_id=X, *, verdict="INVALID", attempt="A001"):
    """A minimal stand-in for an accepted G7 report (only what the loader reads); returns its hash."""
    path = root / "authoring/dialogue_verifications" / f"{call_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_json({"verdict": verdict, "generation": {"attempt_id": attempt}}).encode("utf-8"))
    return g1.sha256(path.read_bytes())


def feedback_file(tmp_path, root, **overrides):
    entry = {"source_g7_report_sha256": g7_report(root), "source_g7_attempt_id": "A001", "round": 2, "guidance": list(GUIDANCE)}
    entry.update(overrides)
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps({X: entry}), encoding="utf-8")
    return path


def test_the_loader_accepts_guidance_tied_to_the_current_invalid_report(tmp_path):
    root = tmp_path / "dataset"
    loaded = load_repair_feedback(feedback_file(tmp_path, root), root)
    assert loaded[X]["guidance"] == GUIDANCE and loaded[X]["round"] == 2


@pytest.mark.parametrize("change, text", [
    ({"source_g7_report_sha256": "0" * 64}, "does not match the current G7 report"),
    ({"source_g7_attempt_id": "A009"}, "must be the INVALID report of attempt A009"),
    ({"guidance": []}, "guidance must be"),
    ({"guidance": ["  "]}, "guidance must be"),
    ({"guidance": ["x" * (MAX_GUIDANCE_CHARS + 1)]}, "guidance must be"),
    ({"guidance": ["ok"] * 9}, "guidance must be"),
])
def test_the_loader_rejects_stale_or_malformed_feedback(tmp_path, change, text):
    root = tmp_path / "dataset"
    path = feedback_file(tmp_path, root, **change)
    with pytest.raises(ValueError, match=text):
        load_repair_feedback(path, root)


def test_the_loader_refuses_feedback_for_a_valid_report_or_a_missing_report(tmp_path):
    root = tmp_path / "dataset"
    digest = g7_report(root, verdict="VALID")
    path = tmp_path / "f.json"
    path.write_text(json.dumps({X: {"source_g7_report_sha256": digest, "source_g7_attempt_id": "A001", "round": 2, "guidance": ["a"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be the INVALID report"):
        load_repair_feedback(path, root)
    (root / "authoring/dialogue_verifications" / f"{X}.json").unlink()
    with pytest.raises(ValueError, match="cannot be read"):
        load_repair_feedback(path, root)
    path.write_text(json.dumps({X: {"guidance": ["a"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="needs exactly"):
        load_repair_feedback(path, root)


def test_only_the_repaired_call_gets_guidance_and_every_other_request_is_byte_identical(tmp_path):
    root = tmp_path / "dataset"
    feedback = load_repair_feedback(feedback_file(tmp_path, root), root)
    plain = RequestBuilder(shared_inputs(), SCHEMAS, PROMPTS)
    guided = RequestBuilder(shared_inputs(), SCHEMAS, PROMPTS, feedback)
    assert guided.build(Y, "A002") == plain.build(Y, "A002")
    body, base = guided.build(X, "A002"), plain.build(X, "A002")
    brief = json.loads(body["input"][1]["content"])
    assert brief["repair_guidance"] == GUIDANCE and "repair_guidance" not in json.loads(base["input"][1]["content"])
    assert {**brief, "repair_guidance": None}.keys() == {**json.loads(base["input"][1]["content"]), "repair_guidance": None}.keys()
    assert body["input"][0] == base["input"][0] and body != base  # same developer message, different user brief
    assert "source_g7_report_sha256" not in body["input"][1]["content"]  # hashes stay in Python-owned records


def test_the_g6_prompt_explains_repair_guidance_without_loosening_any_rule():
    import re
    text = re.sub(r"\s+", " ", (PROMPTS / "dialogue.v1.md").read_text(encoding="utf-8"))
    for required in ("Repair guidance (only when the brief has `repair_guidance`)", "Write a fresh, complete dialogue",
                     "it never changes an event, a marker rule, a placement constraint or any other rule above",
                     "never asks you to mention the verification", "follow the rule above"):
        assert required in text, required


def test_a_repair_attempt_records_its_source_and_guidance_and_is_accepted_like_any_other(tmp_path):
    root = tmp_path / "release"
    feedback = load_repair_feedback(feedback_file(tmp_path, root), root)
    runner, fake = make_runner(tmp_path, repair_feedback=feedback)
    submit_and_finish(runner, fake, [X, Y])
    runner.collect()
    attempt = runner.store.read_attempt(X, "A001")
    assert attempt["state"] == "ACCEPTED" and attempt["repair_feedback"]["guidance"] == GUIDANCE and attempt["repair_feedback"]["round"] == 2
    assert attempt["repair_feedback"]["source_g7_report_sha256"] == feedback[X]["source_g7_report_sha256"]
    assert "repair_feedback" not in runner.store.read_attempt(Y, "A001")
    sent = parse_json_text(runner.store.read_request(X, "A001"))
    assert json.loads(sent["input"][1]["content"])["repair_guidance"] == GUIDANCE  # the ACTUAL request carries the guidance
    committed = parse_json_text((runner.root / runner.spec.provenance_path(X)).read_text(encoding="utf-8"))
    assert committed["attempts"][0]["repair_feedback"]["source_g7_attempt_id"] == "A001"
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


def test_the_cli_accepts_feedback_only_with_dry_run_or_submit_and_a_dataset_root(tmp_path, capsys):
    root = tmp_path / "dataset"
    path = feedback_file(tmp_path, root)
    assert main(["--repair-feedback", str(path), "--collect"]) == 2
    assert "--dataset-root" in capsys.readouterr().err
    assert main(["--dataset-root", str(root), "--repair-feedback", str(path), "--status"]) == 2
    assert "--submit, --sync or --dry-run" in capsys.readouterr().err
    path.write_text("{}", encoding="utf-8")
    assert main(["--dataset-root", str(root), "--repair-feedback", str(path), "--dry-run"]) == 2
    assert "repair feedback" in capsys.readouterr().err
