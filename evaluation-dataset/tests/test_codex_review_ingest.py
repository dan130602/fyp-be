"""Adapter compatibility, not a test of a model's semantic judgement.

Responses in these plumbing tests are artificial. The separate qualification
exercise contains actual Codex reviews of existing speech and seeded defects.
"""

import json

import pytest
from jsonschema import Draft202012Validator

import multilingual_support as M
from fyp_evaluation.llm import exposure, review_policy
from fyp_evaluation.llm.stage_runner import RunError
from fyp_evaluation.plan_verifications.cli import main as g5_main
from fyp_evaluation.dialogue_verifications.cli import main as g7_main
from fyp_evaluation.plan_verifications.run import synthesize_response as g5_response
from fyp_evaluation.dialogue_verifications.run import synthesize_response as g7_response

CALLS = [M.first_call(p) for p in M.PROFILES]
REVIEWER = {"kind": "codex-harness", "model": "fixture-model", "note": "Artificial adapter test, not a semantic review"}


@pytest.mark.parametrize("stage,name", [("stage5", "plan_verification"), ("stage7", "dialogue_verification")])
def test_codex_metadata_cannot_downgrade_to_the_legacy_manual_branch(stage, name):
    params = {"method": "manual_review", "reviewer": "codex-harness", "reviewer_model": "fixture-model", "request_sha256": "a" * 64}
    for filename in (f"{name}.v2.schema.json", f"{name}_provenance.schema.json"):
        schema = json.loads((M.SCHEMAS / stage / filename).read_text(encoding="utf-8"))
        definition = schema["properties"]["generation"]["properties"]["params"] if ".v2." in filename else schema["$defs"]["params"]
        validator = Draft202012Validator(definition)
        assert not validator.is_valid(params)
        assert validator.is_valid({**params, "review_format": "codex-review/1", "review_policy_version": "codex-review-v1",
                                   "review_policy_sha256": "b" * 64})


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    chain = M.chain(tmp_path_factory.mktemp("codex_base"), CALLS)
    assert set(chain.run_g3().values()) == set(chain.run_g4().values()) == {"ACCEPTED"}
    return chain


def review_file(runner, call_id, response, path, *, reopen=False):
    packet = runner.prepare_review(call_id, reviewer_kind="codex-harness", reopen=reopen)
    data = {"format": packet["format"], "stage": packet["stage"], "call_id": call_id,
            "reviewer": dict(REVIEWER), "request_sha256": packet["request_sha256"],
            "review_policy_sha256": packet["review_policy"]["sha256"], "response": response}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def read(chain, folder, call_id):
    return json.loads((chain.root / "authoring" / folder / f"{call_id}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("call_id", CALLS)
def test_codex_g5_and_g7_use_the_normal_lifecycle_and_record_the_scope(base, tmp_path, call_id):
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=CALLS)
    for stage in ("G5", "G7"):
        if stage == "G7":
            assert chain.run_g6(call_ids=[call_id]) == {call_id: "ACCEPTED"}
        runner = chain.open_g5() if stage == "G5" else chain.open_g7()
        synthesize = g5_response if stage == "G5" else g7_response
        response = synthesize(runner.builder.context(call_id))
        packet = runner.prepare_review(call_id, reviewer_kind="codex-harness")
        legacy = runner.prepare_review(call_id)
        assert legacy["format"] == "claude-review/1" and "review_policy" not in legacy
        assert packet["request"] == legacy["request"] and packet["request_sha256"] == legacy["request_sha256"]
        assert packet["review_policy"] == review_policy.codex_review_policy()
        path = review_file(runner, call_id, response, tmp_path / f"{stage}.json")
        attempt = runner.ingest_review(call_id, path)
        assert attempt["state"] == "ACCEPTED" and attempt["model"] == "codex-harness:fixture-model"
        assert attempt["transport"] == "manual" and attempt["estimated_cost_usd"] == 0
        assert attempt["params"]["review_policy_sha256"] == packet["review_policy"]["sha256"]
        folder = "plan_verifications" if stage == "G5" else "dialogue_verifications"
        report = read(chain, folder, call_id)
        assert report["verdict"] == "VALID" and report["generation"]["params"] == attempt["params"]
        assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
        raw = runner.store.read_raw_response(call_id, attempt["attempt_id"])
        assert raw["response"]["manual_review"]["review_policy"] == packet["review_policy"]
        assert runner.transport().calls == []
        if stage == "G5":
            assert runner.g6_eligibility([call_id])[call_id]["eligible"]
        else:
            assert runner.g8_eligibility([call_id])[call_id]["eligible"]
    assert exposure.build(chain.root, chain.audit_root, runner.inputs.dataset_id)["live_attempts"] == []


@pytest.mark.parametrize("fault", ["missing_policy", "wrong_policy", "wrong_request", "wrong_kind", "wrong_format", "fabricated_quote"])
def test_codex_review_cannot_bypass_identity_scope_or_grounding(base, tmp_path, fault):
    call_id = CALLS[0]
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=CALLS)
    runner = chain.open_g5()
    path = review_file(runner, call_id, g5_response(runner.builder.context(call_id)), tmp_path / "review.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if fault == "missing_policy":
        del data["review_policy_sha256"]
    elif fault == "wrong_policy":
        data["review_policy_sha256"] = "0" * 64
    elif fault == "wrong_request":
        data["request_sha256"] = "0" * 64
    elif fault == "wrong_kind":
        data["reviewer"]["kind"] = "claude-code-harness"
    elif fault == "wrong_format":
        data["format"] = "claude-review/1"
        del data["review_policy_sha256"]
    else:
        next(c for c in data["response"]["checks"] if c["evidence"])["evidence"][0]["quote"] = "fabricated citation"
    path.write_text(json.dumps(data), encoding="utf-8")
    if fault == "fabricated_quote":
        attempt = runner.ingest_review(call_id, path)
        assert attempt["state"] == "REJECTED" and any("substring" in e for e in attempt["validation_errors"])
    else:
        with pytest.raises(RunError):
            runner.ingest_review(call_id, path)
        assert runner.store.attempt_ids(call_id) == []


def test_scope_change_during_review_is_refused_and_history_rebuilds_from_recorded_scope(base, tmp_path, monkeypatch):
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=CALLS)
    runner = chain.open_g5()
    call_id = CALLS[0]
    path = review_file(runner, call_id, g5_response(runner.builder.context(call_id)), tmp_path / "review.json")
    original = review_policy.POLICY_PATH
    changed = tmp_path / "changed.md"
    changed.write_bytes(original.read_bytes() + b"\nChanged policy.\n")
    monkeypatch.setattr(review_policy, "POLICY_PATH", changed)
    with pytest.raises(RunError, match="review policy changed"):
        runner.ingest_review(call_id, path)
    assert runner.store.attempt_ids(call_id) == []
    monkeypatch.setattr(review_policy, "POLICY_PATH", original)
    assert runner.ingest_review(call_id, path)["state"] == "ACCEPTED"
    monkeypatch.setattr(review_policy, "POLICY_PATH", changed)
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


@pytest.mark.parametrize("raw_saved", [False, True])
def test_interrupted_codex_import_recovers_without_claiming_provider_billing(base, tmp_path, monkeypatch, raw_saved):
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=CALLS)
    runner = chain.open_g5()
    call_id = CALLS[0]
    path = review_file(runner, call_id, g5_response(runner.builder.context(call_id)), tmp_path / "review.json")
    hook = "_adjudicate" if raw_saved else "_store_response"
    original = getattr(runner, hook)

    def interrupt(*args, **kwargs):
        raise RuntimeError("simulated local interruption")

    monkeypatch.setattr(runner, hook, interrupt)
    with pytest.raises(RuntimeError, match="simulated local interruption"):
        runner.ingest_review(call_id, path)
    saved = runner.store.read_attempt(call_id, "A001")
    assert saved["state"] == ("RESPONDED" if raw_saved else "PREPARED")
    assert saved["model"] == "codex-harness:fixture-model"
    assert saved["params"]["review_policy_sha256"] == review_policy.codex_review_policy()["sha256"]
    monkeypatch.setattr(runner, hook, original)
    with runner.store.lock():
        runner._recover()
    recovered = runner.store.read_attempt(call_id, "A001")
    assert recovered["possibly_billed"] is False and not recovered.get("ambiguous")
    if raw_saved:
        assert recovered["state"] == "ACCEPTED" and runner.store.attempt_ids(call_id) == ["A001"]
    else:
        assert recovered["state"] == "TRANSPORT_FAILED" and recovered["estimated_cost_usd"] == 0
        assert runner.ingest_review(call_id, path)["attempt_id"] == "A002"
    assert runner.transport().calls == [] and runner.validate() == []
    assert runner.check() == {"missing": [], "changed": [], "extra": []}


@pytest.mark.parametrize("assessment,verdict", [("MISMATCH", "INVALID"), ("UNCLEAR", "UNCLEAR")])
def test_gold_preserving_authoring_issue_still_blocks_g7(base, tmp_path, assessment, verdict):
    """A whole-dialogue issue need not lie about the checklist to block acceptance."""
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=CALLS)
    call_id = CALLS[0]
    g5 = chain.open_g5()
    path = review_file(g5, call_id, g5_response(g5.builder.context(call_id)), tmp_path / "g5.json")
    assert g5.ingest_review(call_id, path)["state"] == "ACCEPTED"
    assert chain.run_g6(call_ids=[call_id]) == {call_id: "ACCEPTED"}
    runner = chain.open_g7()
    ctx = runner.builder.context(call_id)
    response = g7_response(ctx)
    turn = ctx.turns[0]
    response["issues"] = [{"issue_id": "I1", "assessment": assessment, "code": "CROSS_CHECK_INTERFERENCE",
                           "question_ids": [], "assignment_ids": [], "language_row_ids": [],
                           "evidence": [{"turn_id": turn["turn_id"], "quote": turn["text"][:40]}],
                           "explanation": "Artificial plumbing assertion; not a semantic judgement about this turn.",
                           "repair_scope": "G6_ONLY"}]
    path = review_file(runner, call_id, response, tmp_path / "g7.json")
    assert runner.ingest_review(call_id, path)["state"] == "ACCEPTED"
    assert read(chain, "dialogue_verifications", call_id)["verdict"] == verdict
    assert not runner.g8_eligibility([call_id])[call_id]["eligible"]


def test_both_clis_prepare_codex_packets_and_do_not_allow_the_flag_on_other_modes(base, tmp_path, capsys):
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=CALLS)
    call_id = CALLS[0]
    args = ["--dataset-root", str(chain.root), "--audit-dir", str(chain.audit_root), "--call-id", call_id]
    for main in (g5_main, g7_main):
        if main is g7_main:
            runner = chain.open_g5()
            path = review_file(runner, call_id, g5_response(runner.builder.context(call_id)), tmp_path / "review.json")
            runner.ingest_review(call_id, path)
            chain.run_g6(call_ids=[call_id])
        packet = tmp_path / "packet.json"
        assert main([*args, "--prepare-review", "--reviewer-kind", "codex-harness", "--output", str(packet)]) == 0
        assert json.loads(packet.read_text(encoding="utf-8"))["format"] == "codex-review/1"
        with pytest.raises(SystemExit):
            main([*args, "--status", "--reviewer-kind", "codex-harness"])
        capsys.readouterr()
