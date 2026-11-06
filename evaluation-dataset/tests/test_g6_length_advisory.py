"""Length is advisory for new multilingual attempts; stored paid responses can be reused offline without rewriting history."""

import json
from copy import deepcopy

import pytest

import multilingual_support as M
from fyp_evaluation.dialogues import config
from fyp_evaluation.dialogues.feasibility import FILLER_BY_PROFILE, synthesize_dialogue
from fyp_evaluation.dialogues.request import DialogueContext, RequestBuilder
from fyp_evaluation.dialogues.validate_response import DialogueValidator
from fyp_evaluation.llm import provenance
from fyp_evaluation.llm.stage_runner import RunError

CID = M.first_call("MANDARIN")


def compact(ctx):
    parsed = synthesize_dialogue(ctx)
    for turn in parsed["turns"]:
        if "[[" not in turn["text"]:
            turn["text"] = "嗯。"
    return parsed


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    chain = M.chain(tmp_path_factory.mktemp("length_base"), [CID])
    assert set(chain.run_g3().values()) == set(chain.run_g4().values()) == set(chain.run_g5().values()) == {"ACCEPTED"}
    return chain


def rejected(base, tmp_path, monkeypatch):
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=[CID])
    def strict(spec, documents):
        return config.LENGTH_POLICY_V2["MANDARIN"]
    with monkeypatch.context() as old:
        old.setattr("fyp_evaluation.dialogues.request.length_policy_for", strict)
        old.setattr("fyp_evaluation.dialogues.run.length_policy_for", strict)
        old.setattr(config, "PROMPT_TEMPLATE_MULTILINGUAL", "dialogue.v3.md")
        runner = chain.open_g6({CID: compact})
        result = runner.run_sync(CID)
        assert result["state"] == "REJECTED", result
        assert result["validation_errors"] and all(e.startswith("length:") for e in result["validation_errors"])
    runner = chain.open_g6()
    return chain, runner


@pytest.mark.parametrize("profile", M.PROFILES)
def test_under_and_over_length_do_not_hide_other_errors(profile):
    cid = M.first_call(profile)
    builder = RequestBuilder(M.verification_inputs([cid]), M.SCHEMAS)
    ctx = builder.context(cid)
    validator = DialogueValidator(builder.response_schema)
    parsed = synthesize_dialogue(ctx)
    # Change only the advisory bounds to exercise both sides without generating thousands of filler turns.
    ctx.bounds = {**ctx.bounds, "accepted_words": [1, 2]}
    assert validator.errors(parsed, ctx) == []
    ctx.bounds = {**ctx.bounds, "accepted_words": [100000, 200000]}
    assert validator.errors(parsed, ctx) == []
    parsed["turns"][0]["speaker"] = "CLIENT" if parsed["turns"][0]["speaker"] == "EMPLOYEE" else "EMPLOYEE"
    assert validator.errors(parsed, ctx), "speaker/alternation errors must still block"
    assert config.length_is_advisory(ctx.length_policy)
    brief = json.loads(builder.build(cid, "A001")["input"][1]["content"])
    assert brief["call"]["length"]["enforcement"] == "advisory"


def test_offline_reuse_preserves_rejection_cost_and_exact_response(base, tmp_path, monkeypatch):
    chain, runner = rejected(base, tmp_path, monkeypatch)
    source = deepcopy(runner.store.read_attempt(CID, "A001"))
    request = runner.store.read_request(CID, "A001")
    raw = runner.store.read_raw_response(CID, "A001")
    calls = list(runner.transport().calls)
    result = runner.revalidate_length_only(CID, "A001", "User approved advisory length")
    assert result["state"] == "ACCEPTED" and result["transport"] == "revalidation"
    assert result["estimated_cost_usd"] == 0 and result["usage"] is None and result["exposure"]["run_kind"] == "offline"
    assert runner.store.read_attempt(CID, "A001") == source
    assert runner.store.read_request(CID, "A001") == request
    assert runner.store.read_raw_response(CID, "A001") == raw
    assert runner.transport().calls == calls
    prov = json.loads((chain.root / "authoring/dialogue_provenance" / f"{CID}.json").read_text(encoding="utf-8"))
    assert [a["state"] for a in prov["attempts"]] == ["REJECTED", "ACCEPTED"]
    assert prov["accepted"]["parsed_response"] == runner._parse_response(raw["response"])[1]
    assert prov["accepted"]["prompt_template_sha256"] == source["prompt_template_sha256"]
    artifact = json.loads((chain.root / "authoring/dialogues" / f"{CID}.json").read_text(encoding="utf-8"))
    assert artifact["length"]["enforcement"] == "advisory" and artifact["length"]["within_range"] is False
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    assert runner.g7_eligibility([CID])[CID]["eligible"]
    with pytest.raises(RunError):
        runner.revalidate_length_only(CID, "A001", "cannot accept twice")


@pytest.mark.parametrize("problem", ["nonlength", "ambiguous", "request", "response", "stale"])
def test_reuse_refuses_other_failures_and_modified_or_stale_sources(base, tmp_path, monkeypatch, problem):
    _, runner = rejected(base, tmp_path, monkeypatch)
    source = runner.store.read_attempt(CID, "A001")
    if problem in ("nonlength", "ambiguous"):
        if problem == "nonlength":
            source["validation_errors"].append("EV_001: missing from the dialogue")
        else:
            source["possibly_billed"] = True
        runner.store.write_attempt(source)
        prov = provenance.read(runner.spec, runner.root, runner.inputs.dataset_id, CID)
        provenance.upsert_attempt(prov, source)
        provenance.write(runner.spec, runner.root, prov)
    elif problem == "request":
        runner.store.write_request(CID, "A001", "{}")
    elif problem == "response":
        raw = runner.store.read_raw_response(CID, "A001")
        raw["response"]["id"] = "different-response"
        runner.store.write_raw_response(CID, "A001", raw)
    else:
        original = runner._current_upstream_hashes
        monkeypatch.setattr(runner, "_current_upstream_hashes", lambda c: {**original(c), "call_config": "0" * 64})
    with pytest.raises(RunError):
        runner.revalidate_length_only(CID, "A001", "must fail closed")
    assert runner.store.attempt_ids(CID) == ["A001"]


def test_interrupted_local_acceptance_resumes_without_provider_or_rewriting_source(base, tmp_path, monkeypatch):
    _, runner = rejected(base, tmp_path, monkeypatch)
    source = deepcopy(runner.store.read_attempt(CID, "A001"))
    with monkeypatch.context() as interrupted:
        interrupted.setattr(runner, "_accept", lambda *a: (_ for _ in ()).throw(RuntimeError("interrupted")))
        with pytest.raises(RuntimeError, match="interrupted"):
            runner.revalidate_length_only(CID, "A001", "resume offline")
    assert runner.store.read_attempt(CID, "A002")["state"] == "RESPONDED"
    assert runner._finalize_responded()[0]["state"] == "ACCEPTED"
    assert runner.store.read_attempt(CID, "A001") == source
    assert runner.transport().calls == [] and runner.validate() == []
