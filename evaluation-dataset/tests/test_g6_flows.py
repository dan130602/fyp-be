"""G6 lifecycle with fake transports: Batch acceptance, schema-valid provenance, the G5 gate and acceptance ordering,
history, length-policy reproduction, recovery and the CLI. Offline; every mutation runs on an isolated copy."""

import json

import pytest
from jsonschema import Draft202012Validator

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.dialogues import config
from fyp_evaluation.dialogues.cli import main
from fyp_evaluation.dialogues.feasibility import synthesize_dialogue
from fyp_evaluation.dialogues.run import DialogueRunner, RunError
from fyp_evaluation.llm.errors import AmbiguousError
from g6_support import RELEASE, SCHEMAS, Clock, attempt_state, make_runner, make_world, shared_inputs, snapshot, submit_and_finish

X, Y = "CALL_027", "CALL_010"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def artifact(runner, call_id=X):
    return parse_json_text((runner.root / runner.spec.artifact_path(call_id)).read_text(encoding="utf-8"))


def provenance(runner, call_id=X):
    return parse_json_text((runner.root / runner.spec.provenance_path(call_id)).read_text(encoding="utf-8"))


def g5_path(runner, suffix, call_id=X):
    return runner.root / f"authoring/{suffix}" / f"{call_id}.json"


def rewrite(path, mutate):
    data = parse_json_text(path.read_bytes().decode("utf-8"))
    mutate(data)
    path.write_bytes(serialize_json(data).encode("utf-8"))


def bad_response(ctx):
    response = synthesize_dialogue(ctx)
    response["turns"][0]["text"] = "No markers here."  # drops every event in the first turn
    return response


# ============================================================================================ acceptance and artifacts
def test_batch_acceptance_writes_a_valid_dialogue_provenance_and_record(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = submit_and_finish(runner, fake, [X, Y])
    result = runner.collect()
    assert result["collected"] == ["SUB_0001"] and result["anomalies"] == []
    assert [attempt_state(runner, c) for c in (X, Y)] == ["ACCEPTED", "ACCEPTED"] and report["submitted"] == 2
    dialogue = artifact(runner)
    assert dialogue["stage"] == "G6" and dialogue["generation"]["params"]["length_policy"]["version"] == "g6-length-v1"
    assert dialogue["plan_verification_sha256"] == shared_inputs().report_hashes[X]
    assert dialogue["check_plan_sha256"] == shared_inputs().plan_hashes[X]
    assert list(Draft202012Validator(runner.builder.artifact_schema).iter_errors(dialogue)) == []
    prov = provenance(runner)
    prov_schema = parse_json_text((SCHEMAS / "stage6" / "dialogue_provenance.schema.json").read_text(encoding="utf-8"))
    assert list(Draft202012Validator(prov_schema).iter_errors(prov)) == []  # Batch transport fields and G6 params
    attempt = prov["attempts"][0]
    assert attempt["transport"] == "batch" and attempt["submission_id"] == "SUB_0001" and attempt["params"]["max_output_tokens"] == 32768
    assert attempt["upstream_sha256"]["plan_verification"] == shared_inputs().report_hashes[X]
    record = parse_json_text((runner.root / config.RECORD).read_text(encoding="utf-8"))
    assert record["summary"]["accepted_calls"] == 2 and record["summary"]["words_by_profile"]["SHORT"]["min"] >= 510
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


def test_dry_run_is_deterministic_feasible_and_reports_a_batch_estimate(tmp_path):
    runner, fake = make_runner(tmp_path)
    first = runner.dry_run()
    second = runner.dry_run()
    assert [r["request_sha256"] for r in first["calls"]] == [r["request_sha256"] for r in second["calls"]]
    assert first["feasibility"]["infeasible"] == {} and first["feasibility"]["checked"] == 50
    assert first["estimate"]["eligible_calls"] == 50 and first["estimate"]["batch_cost_usd"] > 0
    assert fake.calls == [] and not (runner.root / "authoring").exists()


def test_sync_is_refused_before_anything_is_sent(tmp_path, capsys):
    runner, fake = make_runner(tmp_path)
    with pytest.raises(RunError, match="Batch-only"):
        runner.run_sync(X)
    argv = ["--dataset-root", str(runner.root), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS), "--sync", "--call-id", X]
    assert main(argv, clock=Clock()) == 2 and "Batch-only" in capsys.readouterr().err and fake.calls == []


def test_structural_rejection_is_recorded_and_retried_only_on_request(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={X: bad_response})
    submit_and_finish(runner, fake, [X, Y])
    runner.collect()
    assert [attempt_state(runner, c) for c in (X, Y)] == ["REJECTED", "ACCEPTED"]
    assert any("missing from the dialogue" in e for e in runner.store.read_attempt(X, "A001")["validation_errors"])
    with pytest.raises(RunError, match="--retry-rejected"):
        runner.submit([X])
    fake.responder = lambda body: __import__("fyp_evaluation.llm.fake_transport", fromlist=["x"]).response_body(
        json.dumps(synthesize_dialogue(runner.builder.context(X))))
    retry = runner.submit([X], retry_rejected=True)
    fake.finish_batch(retry["batch_id"])
    runner.collect()
    assert attempt_state(runner, X, "A002") == "ACCEPTED" and runner.validate() == []


def test_truncated_response_is_rejected_not_accepted(tmp_path):
    from fyp_evaluation.llm.fake_transport import response_body
    runner, fake = make_runner(tmp_path)
    fake.responder = lambda body: response_body("", status="incomplete", incomplete_reason="max_output_tokens")
    submit_and_finish(runner, fake, [X])
    runner.collect()
    assert attempt_state(runner, X) == "REJECTED" and not (runner.root / runner.spec.artifact_path(X)).exists()


def test_shuffled_dropped_and_duplicated_batch_lines_are_handled_per_call(tmp_path):
    runner, fake = make_runner(tmp_path)
    ids = {c: f"{c}-G6-A001" for c in (X, Y, "CALL_017")}
    report = runner.submit(list(ids))
    fake.finish_batch(report["batch_id"], shuffle=True, drop={ids[X]}, duplicate={ids[Y]})
    result = runner.collect()
    assert any("duplicate result lines" in a for a in result["anomalies"])
    assert attempt_state(runner, X) == "TRANSPORT_FAILED" and attempt_state(runner, Y) == "TRANSPORT_FAILED"
    assert attempt_state(runner, "CALL_017") == "ACCEPTED"
    again = runner.submit([X, Y])  # technical failures retry without --retry-rejected
    fake.finish_batch(again["batch_id"])
    runner.collect()
    assert attempt_state(runner, X, "A002") == "ACCEPTED" and attempt_state(runner, Y, "A002") == "ACCEPTED"


def test_ambiguous_submission_blocks_its_calls_until_reviewed(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.create_error = AmbiguousError("timeout creating the batch")
    report = runner.submit([X, Y])
    assert report["state"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(RunError, match="PREPARED"):
        runner.submit([X])
    runner.mark_not_submitted("SUB_0001", acknowledge_duplicate_risk=True)
    assert runner.submit([X])["state"] == "SUBMITTED"


# ============================================================================================ the G5 gate (before sending)
def test_a_tampered_missing_or_invalid_g5_approval_blocks_only_that_call(tmp_path):
    runner, fake = make_world(tmp_path)
    rewrite(g5_path(runner, "plan_verification_provenance"), lambda d: d["accepted"].update(artifact_sha256="0" * 64))
    ok, reason = runner.eligibility(X)
    assert not ok and "not G6-eligible per G5" in reason
    g5_path(runner, "plan_verifications", Y).unlink()
    ok, reason = runner.eligibility(Y)
    assert not ok and "no accepted G5 report" in reason
    assert runner.eligibility("CALL_017")[0] is True
    assert runner._ensure_fresh_inputs(runner.inputs.call_ids) is None  # one call's G5 report never blocks the whole run
    report = runner.dry_run()
    assert report["estimate"]["eligible_calls"] == 48
    assert runner.submit(["CALL_017"])["state"] == "SUBMITTED"
    with pytest.raises(RunError):
        runner.submit([X])


def test_an_invalid_g5_verdict_makes_the_call_ineligible(tmp_path, monkeypatch):
    runner, _ = make_runner(tmp_path)
    monkeypatch.setattr(type(runner._g5()), "g6_eligibility",
                        lambda self, ids: {i: {"eligible": False, "reason": "verdict INVALID", "verdict": "INVALID"} for i in ids})
    ok, reason = runner.eligibility(X)
    assert not ok and "verdict INVALID" in reason


# ============================================================================================ acceptance ordering (under the lock)
def one_call_down(tmp_path):
    runner, fake = make_world(tmp_path)
    report = runner.submit([X, Y])
    fake.finish_batch(report["batch_id"])
    return runner, fake, report


def assert_no_resends(fake):
    assert [c[0] for c in fake.calls if c[0] in ("create_response", "create_batch")] == ["create_batch"]


def test_a_g5_report_deleted_after_submission_is_terminal_stale_and_keeps_the_response(tmp_path):
    runner, fake, _ = one_call_down(tmp_path)
    g5_path(runner, "plan_verifications").unlink()
    runner.collect()
    assert attempt_state(runner, X) == "STALE_UPSTREAM" and attempt_state(runner, Y) == "ACCEPTED"
    assert "plan_verification" in runner.store.read_attempt(X, "A001")["stale_upstream"]["changed"]
    assert runner.store.read_raw_response(X, "A001") is not None
    assert runner.store.read_submission("SUB_0001")["state"] == "COLLECTED"
    assert_no_resends(fake)


def test_an_unreadable_g5_report_is_a_resumable_hold(tmp_path, monkeypatch):
    runner, fake, _ = one_call_down(tmp_path)
    original = __import__("fyp_evaluation.dialogues.inputs", fromlist=["x"]).g5_report_fingerprint

    def unreadable(root, call_id):
        return config.G5_UNREADABLE if call_id == X else original(root, call_id)
    monkeypatch.setattr("fyp_evaluation.dialogues.inputs.g5_report_fingerprint", unreadable)
    monkeypatch.setattr("fyp_evaluation.dialogues.run.g5_report_fingerprint", unreadable)
    result = runner.collect()
    assert attempt_state(runner, X) == "RESPONDED" and attempt_state(runner, Y) == "ACCEPTED"
    assert runner.store.read_attempt(X, "A001")["blocked"]["kind"] == "UPSTREAM_INVALID"
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMITTED" and any("kept unjudged" in a for a in result["anomalies"])
    monkeypatch.undo()
    runner.collect()
    assert attempt_state(runner, X) == "ACCEPTED" and runner.store.read_submission("SUB_0001")["state"] == "COLLECTED"
    assert_no_resends(fake)


def test_g5_provenance_corruption_with_an_unchanged_report_hash_is_a_resumable_hold(tmp_path):
    runner, fake, _ = one_call_down(tmp_path)
    path = g5_path(runner, "plan_verification_provenance")
    original = path.read_bytes()
    rewrite(path, lambda d: d["accepted"].update(artifact_sha256="0" * 64))
    runner.collect()
    assert attempt_state(runner, X) == "RESPONDED" and attempt_state(runner, Y) == "ACCEPTED"
    assert runner.store.read_attempt(X, "A001")["blocked"]["kind"] == "UPSTREAM_INVALID"
    path.write_bytes(original)
    runner.collect()
    assert attempt_state(runner, X) == "ACCEPTED" and runner.validate() == []
    assert_no_resends(fake)


@pytest.mark.parametrize("path_of", [lambda r: r.root / "authoring/check_plans" / f"{X}.json",
                                     lambda r: r.root / "authoring/call_configs" / f"{X}.json"])
def test_g3_or_g4_changed_after_submission_with_the_old_g5_report_is_terminal_stale(tmp_path, path_of):
    runner, fake, _ = one_call_down(tmp_path)
    path = path_of(runner)
    path.write_bytes(path.read_bytes() + b"\n")
    runner.collect()
    assert attempt_state(runner, X) == "STALE_UPSTREAM" and attempt_state(runner, Y) == "ACCEPTED"
    assert runner.store.read_raw_response(X, "A001") is not None and runner.store.read_submission("SUB_0001")["state"] == "COLLECTED"
    assert_no_resends(fake)


# ============================================================================================ history and length-policy reproduction
def test_reopen_keeps_history_and_rebuilds_it_against_the_recorded_versions(tmp_path):
    runner, fake = make_runner(tmp_path)
    submit_and_finish(runner, fake, [X])
    runner.collect()
    first = (runner.root / runner.spec.artifact_path(X)).read_text(encoding="utf-8")
    retry = runner.submit([X], reopen_reason="re-generate for a test")
    fake.finish_batch(retry["batch_id"])
    runner.collect()
    (history,) = provenance(runner)["accepted_history"]
    assert history["reason"] == "re-generate for a test" and attempt_state(runner, X, "A002") == "ACCEPTED"
    assert runner._rebuild_text(X, history) == first.replace("\r\n", "\n")
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


def recalibrated():
    return {**config.LENGTH_POLICY, "version": "g6-length-v2", "assumed_wpm": 150,
            "profiles": {"SHORT": {"accepted_words": [540, 800], "target_words": [600, 700]},
                         "LONG": config.LENGTH_POLICY["profiles"]["LONG"]}}


def test_a_historical_dialogue_rebuilds_with_its_own_length_policy_after_recalibration(tmp_path, monkeypatch):
    runner, fake = make_runner(tmp_path)
    submit_and_finish(runner, fake, [X])
    runner.collect()
    before = (runner.root / runner.spec.artifact_path(X)).read_bytes()
    monkeypatch.setattr("fyp_evaluation.dialogues.run.LENGTH_POLICY", recalibrated())
    fresh = DialogueRunner(runner.root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, inputs=shared_inputs(), clock=Clock())
    assert fresh.validate() == [] and fresh.check() == {"missing": [], "changed": [], "extra": []}
    assert (runner.root / runner.spec.artifact_path(X)).read_bytes() == before
    assert artifact(fresh)["length"]["assumed_wpm"] == 160  # the recorded policy, not the recalibrated constant


def test_an_attempt_built_under_an_older_length_policy_goes_stale_when_the_policy_changes(tmp_path, monkeypatch):
    runner, fake = make_runner(tmp_path)
    submit_and_finish(runner, fake, [X, Y])
    monkeypatch.setattr("fyp_evaluation.dialogues.run.LENGTH_POLICY", recalibrated())
    runner.collect()
    assert attempt_state(runner, X) == "STALE_UPSTREAM"
    assert "settings changed" in runner.store.read_attempt(X, "A001")["rejection_reason"]
    assert runner.store.read_raw_response(X, "A001") is not None
    assert_no_resends(fake)


# ============================================================================================ historical G5 verification
def accepted_world(tmp_path):
    runner, fake = make_world(tmp_path)
    submit_and_finish(runner, fake, [X])
    runner.collect()
    assert attempt_state(runner, X) == "ACCEPTED"
    return runner


def fresh_validate(runner):
    other = DialogueRunner(runner.root, audit_root=runner.audit_root, schema_dir=SCHEMAS, inputs=runner.inputs, clock=Clock())
    return other.validate()


def test_a_corrupted_historical_g5_entry_with_an_unchanged_hash_is_rejected(tmp_path):
    runner = accepted_world(tmp_path)
    assert fresh_validate(runner) == []
    rewrite(g5_path(runner, "plan_verification_provenance"),
            lambda d: d["accepted"]["parsed_response"]["checks"][0].update(rationale="Edited after acceptance."))
    assert any("G5" in e for e in fresh_validate(runner))


def test_a_g5_approval_for_different_g3_g4_versions_is_rejected(tmp_path):
    runner = accepted_world(tmp_path)

    def other_plan(data):
        accepted_id = data["accepted"]["attempt_id"]
        next(a for a in data["attempts"] if a["attempt_id"] == accepted_id)["upstream_sha256"]["check_plan"] = "f" * 64
    rewrite(g5_path(runner, "plan_verification_provenance"), other_plan)
    assert any("G3/G4" in e or "G5" in e for e in fresh_validate(runner))


def test_dependency_status_never_advertises_a_tampered_or_missing_approval(tmp_path):
    runner = accepted_world(tmp_path)
    assert runner.dependency_status([X])[X]["status"] == "current"
    rewrite(g5_path(runner, "plan_verification_provenance"), lambda d: d["accepted"].update(artifact_sha256="0" * 64))
    assert runner.dependency_status([X])[X]["status"] == "upstream_invalid"
    g5_path(runner, "plan_verifications").unlink()
    status = runner.dependency_status([X])[X]
    assert status["status"] == "stale" and "plan_verification" in status["changed"]


# ============================================================================================ upstream and production safety
def test_read_only_commands_never_mutate_state(tmp_path):
    runner, fake = make_runner(tmp_path)
    submit_and_finish(runner, fake, [X])
    runner.collect()
    before = snapshot(runner.root, tmp_path / "audit")
    sends = len(fake.calls)
    argv = ["--dataset-root", str(runner.root), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS)]
    for mode in ("--status", "--validate", "--check"):
        assert main(argv + [mode], clock=Clock(), runner_class=lambda root, **kw: DialogueRunner(root, inputs=shared_inputs(), **kw)) in (0, 1)
    assert snapshot(runner.root, tmp_path / "audit") == before and len(fake.calls) == sends


def test_tests_run_against_the_committed_g5_reports_and_write_nothing_to_the_release():
    assert len(list((RELEASE / "authoring/plan_verifications").glob("CALL_*.json"))) == 50


# ============================================================================================ prompt/schema fingerprints are read fresh
def runner_with_private_prompt_and_schema(tmp_path):
    """A runner whose prompt directory and schema directory are private copies, so a test can change them."""
    import shutil

    from fyp_evaluation.llm.fake_transport import FakeTransport
    from g6_support import good_responder
    prompts, schemas = tmp_path / "prompts_stage6", tmp_path / "schemas"
    shutil.copytree(SCHEMAS.parent / "prompts" / "stage6", prompts)
    shutil.copytree(SCHEMAS, schemas)
    holder = {}
    fake = FakeTransport(lambda body: holder["responder"](body))
    root = tmp_path / "release"
    root.mkdir()
    runner = DialogueRunner(root, audit_root=tmp_path / "audit", schema_dir=schemas, prompt_dir=prompts, transport=fake, clock=Clock(),
                            inputs=shared_inputs())
    holder["responder"] = good_responder(runner)
    return runner, fake, prompts / "dialogue.v1.md", schemas / "stage6" / "dialogue_response.schema.json"


@pytest.mark.parametrize("which, key", [(2, "prompt_template_sha256"), (3, "response_schema_sha256")])
def test_a_prompt_or_schema_changed_while_the_process_is_alive_makes_the_response_stale(tmp_path, which, key):
    runner, fake, *files = runner_with_private_prompt_and_schema(tmp_path)
    path = files[which - 2]
    submit_and_finish(runner, fake, [X, Y])
    assert runner.builder.prompt_template_sha256  # the builder (and its cached hashes) already exists in this process
    path.write_bytes(path.read_bytes() + b"\n")
    runner.collect()
    attempt = runner.store.read_attempt(X, "A001")
    assert attempt["state"] == "STALE_UPSTREAM" and key in attempt["stale_upstream"]["changed"]
    assert "settings changed" in attempt["rejection_reason"] and runner.store.read_raw_response(X, "A001") is not None
    assert attempt_state(runner, Y) == "STALE_UPSTREAM"  # every in-flight response built from the old prompt/schema is terminal
    assert_no_resends(fake)


def test_an_unreadable_prompt_is_a_resumable_hold_and_unchanged_files_still_accept(tmp_path):
    runner, fake, prompt, _ = runner_with_private_prompt_and_schema(tmp_path)
    submit_and_finish(runner, fake, [X, Y])
    original = prompt.read_bytes()
    prompt.unlink()
    runner.collect()
    assert attempt_state(runner, X) == "RESPONDED" and runner.store.read_attempt(X, "A001")["blocked"]["kind"] == "UPSTREAM_INVALID"
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMITTED"
    prompt.write_bytes(original)
    runner.collect()
    assert attempt_state(runner, X) == "ACCEPTED" and attempt_state(runner, Y) == "ACCEPTED"
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    assert_no_resends(fake)
