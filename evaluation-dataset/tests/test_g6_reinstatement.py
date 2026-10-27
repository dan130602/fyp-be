"""Reopen then reinstate: an earlier accepted G6 dialogue can be made active again offline, with every attempt, response, cost and
exposure record preserved, and with G7 following the active dialogue (stale until re-reviewed). Fake transports only; nothing is sent."""

import json

import pytest

import multilingual_support as M
from fyp_evaluation.dialogues import cli
from fyp_evaluation.dialogues.feasibility import synthesize_dialogue
from fyp_evaluation.llm import exposure, provenance
from fyp_evaluation.llm.stage_runner import RunError

CID = M.first_call("MANDARIN")


def compact(ctx):
    """A valid but different dialogue: every supporting turn is a single word, planned events untouched."""
    parsed = synthesize_dialogue(ctx)
    for turn in parsed["turns"]:
        if "[[" not in turn["text"]:
            turn["text"] = "嗯。"
    return parsed


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    chain = M.chain(tmp_path_factory.mktemp("reinstate_base"), [CID])
    assert set(chain.run_g3().values()) == set(chain.run_g4().values()) == set(chain.run_g5().values()) == {"ACCEPTED"}
    return chain


def artifact(chain):
    return json.loads((chain.root / "authoring/dialogues" / f"{CID}.json").read_text(encoding="utf-8"))


def artifact_text(chain):
    return (chain.root / "authoring/dialogues" / f"{CID}.json").read_text(encoding="utf-8")


def provenance_of(chain, runner):
    return provenance.read(runner.spec, chain.root, runner.inputs.dataset_id, CID)


def baseline_then_candidate(base, tmp_path):
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=[CID])
    first = chain.open_g6()
    assert first.run_sync(CID)["state"] == "ACCEPTED"
    baseline = artifact(chain)
    second = chain.open_g6({CID: compact})
    result = second.run_sync(CID, reopen_reason="G6 v5 candidate")
    assert result["state"] == "ACCEPTED" and result["attempt_id"] == "A002", result
    assert artifact(chain)["turns"] != baseline["turns"]
    return chain, second, baseline


def test_reinstating_the_baseline_restores_its_turns_and_keeps_every_attempt(base, tmp_path):
    chain, runner, baseline = baseline_then_candidate(base, tmp_path)
    before = {a: (runner.store.read_request(CID, a), runner.store.read_raw_response(CID, a)) for a in ("A001", "A002")}
    calls = list(runner.transport().calls)
    result = runner.reinstate_accepted(CID, "A001", "v5 candidate judged worse")
    assert result["state"] == "ACCEPTED" and result["attempt_id"] == "A003" and result["transport"] == "reinstatement"
    assert result["estimated_cost_usd"] == 0 and result["usage"] is None and result["exposure"]["run_kind"] == "offline"
    assert result["provider_response_id"] is None and result["possibly_billed"] is False
    assert runner.transport().calls == calls  # nothing was sent
    restored = artifact(chain)
    assert restored["turns"] == baseline["turns"] and restored["generation"]["attempt_id"] == "A003"
    prov = provenance_of(chain, runner)
    assert [a["state"] for a in prov["attempts"]] == ["SUPERSEDED", "SUPERSEDED", "ACCEPTED"]
    assert [e["attempt_id"] for e in prov["accepted_history"]] == ["A001", "A002"]
    assert prov["accepted"]["attempt_id"] == "A003"
    assert prov["attempts"][-1]["reinstatement"]["source_attempt_id"] == "A001"
    assert prov["attempts"][-1]["reopen_reason"] == "v5 candidate judged worse"
    for attempt_id, (request, raw) in before.items():  # the earlier requests and responses are untouched
        assert runner.store.read_request(CID, attempt_id) == request and runner.store.read_raw_response(CID, attempt_id) == raw
    assert runner.store.read_request(CID, "A003") == before["A001"][0] and runner.store.read_raw_response(CID, "A003") == before["A001"][1]
    assert runner.store.attempt_ids(CID) == ["A001", "A002", "A003"]
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


def test_reinstating_twice_and_chaining_back_to_the_candidate_both_work_and_never_reuse_ids(base, tmp_path):
    chain, runner, baseline = baseline_then_candidate(base, tmp_path)
    candidate = artifact(chain)
    assert runner.reinstate_accepted(CID, "A001", "back to baseline")["attempt_id"] == "A003"
    assert runner.reinstate_accepted(CID, "A002", "back to the candidate")["attempt_id"] == "A004"
    assert artifact(chain)["turns"] == candidate["turns"]
    prov = provenance_of(chain, runner)
    assert [e["attempt_id"] for e in prov["accepted_history"]] == ["A001", "A002", "A003"]
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


@pytest.mark.parametrize("problem", ["current", "never_accepted", "reason", "tampered", "request", "stale", "g5", "unresolved"])
def test_reinstatement_fails_closed(base, tmp_path, monkeypatch, problem):
    chain, runner, _ = baseline_then_candidate(base, tmp_path)
    source, reason = "A001", "restore the baseline"
    if problem == "current":
        source = "A002"
    elif problem == "never_accepted":
        source = "A009"
    elif problem == "reason":
        reason = "  "
    elif problem == "tampered":
        prov = provenance_of(chain, runner)
        prov["accepted_history"][0]["parsed_response"]["turns"][0]["text"] = "changed after acceptance"
        provenance.write(runner.spec, chain.root, prov)
    elif problem == "request":
        runner.store.write_request(CID, "A001", "{}")
    elif problem == "stale":
        original = runner._current_upstream_hashes
        monkeypatch.setattr(runner, "_current_upstream_hashes", lambda call_id: {**original(call_id), "call_config": "0" * 64})
    elif problem == "g5":
        monkeypatch.setattr(runner, "_g5_gate", lambda call_id: "G5 approval is no longer intact")
    else:
        monkeypatch.setattr(runner, "_awaiting_acknowledgement", lambda latest: True)
    text = artifact_text(chain)
    with pytest.raises(RunError):
        runner.reinstate_accepted(CID, source, reason)
    assert runner.store.attempt_ids(CID) == ["A001", "A002"] and artifact_text(chain) == text


def test_live_exposure_of_both_generations_survives_a_reinstatement_and_the_reinstatement_adds_none(base, tmp_path):
    chain, runner, _ = baseline_then_candidate(base, tmp_path)
    for attempt_id in ("A001", "A002"):  # mark the two generations as live, as a paid run would have
        attempt = runner.store.read_attempt(CID, attempt_id)
        attempt["exposure"] = {**attempt["exposure"], "run_kind": "live"}
        runner.store.write_attempt(attempt)
        prov = provenance_of(chain, runner)
        provenance.upsert_attempt(prov, attempt)
        provenance.write(runner.spec, chain.root, prov)
    dataset_id = runner.inputs.dataset_id
    before = exposure.collect(chain.root, chain.audit_root, dataset_id)
    assert [(row["call_id"], row["attempt_id"], row["state"]) for row in before] == [(CID, "A001", "SUPERSEDED"), (CID, "A002", "ACCEPTED")]
    runner.reinstate_accepted(CID, "A001", "restore the baseline")
    after = exposure.collect(chain.root, chain.audit_root, dataset_id)
    # both paid generations stay in the ledger with their request, cost and exposure; only the lifecycle state of A002 moved on
    assert [(row["attempt_id"], row["state"]) for row in after] == [("A001", "SUPERSEDED"), ("A002", "SUPERSEDED")]
    keep = ("stage", "call_id", "attempt_id", "request_sha256", "estimated_cost_usd", "possibly_billed", "created_at", "code_commit")
    assert [{key: row[key] for key in keep} for row in after] == [{key: row[key] for key in keep} for row in before]
    assert all(row["attempt_id"] != "A003" for row in after)  # the offline reinstatement is not provider exposure


def test_g7_follows_the_active_dialogue_through_reopen_and_reinstatement(base, tmp_path):
    chain = M.copy_chain(base.root, tmp_path / "dataset", call_ids=[CID])
    assert chain.open_g6().run_sync(CID)["state"] == "ACCEPTED"
    g7 = chain.open_g7()
    assert g7.run_sync(CID)["state"] == "ACCEPTED"
    assert g7.dependency_status([CID])[CID]["status"] == "current" and g7.g8_eligibility([CID])[CID]["eligible"]
    g6 = chain.open_g6({CID: compact})
    assert g6.run_sync(CID, reopen_reason="G6 v5 candidate")["state"] == "ACCEPTED"
    g7 = chain.open_g7()
    assert g7.dependency_status([CID])[CID]["status"] == "stale" and not g7.g8_eligibility([CID])[CID]["eligible"]
    assert g7.run_sync(CID, reopen_reason="review of the v5 candidate")["state"] == "ACCEPTED"
    assert g7.dependency_status([CID])[CID]["status"] == "current" and g7.g8_eligibility([CID])[CID]["eligible"]
    prov = json.loads((chain.root / "authoring/dialogue_verification_provenance" / f"{CID}.json").read_text(encoding="utf-8"))
    assert len(prov["accepted_history"]) == 1  # the review of the baseline is retained, not erased
    chain.open_g6().reinstate_accepted(CID, "A001", "restore the baseline")
    g7 = chain.open_g7()
    assert g7.dependency_status([CID])[CID]["status"] == "stale" and not g7.g8_eligibility([CID])[CID]["eligible"]


def test_the_command_line_action_reinstates_offline(base, tmp_path, capsys):
    chain, runner, baseline = baseline_then_candidate(base, tmp_path)
    code = cli.main(["--dataset-root", str(chain.root), "--reinstate-accepted", "--call-id", CID, "--attempt-id", "A001",
                     "--reason", "restore the baseline", "--audit-dir", str(chain.audit_root), "--schema-dir", str(M.SCHEMAS)])
    assert code == 0
    reported = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert reported["state"] == "ACCEPTED" and reported["attempt_id"] == "A003" and reported["source_attempt_id"] == "A001"
    assert artifact(chain)["turns"] == baseline["turns"]
    assert cli.main(["--dataset-root", str(chain.root), "--reinstate-accepted", "--revalidate-length-only"]) == 2
