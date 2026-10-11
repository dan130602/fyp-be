"""G3 workflow with fake transports: sync, Batch, recovery, reopen, validate/check, CLI, env and transport mapping.

No test needs OPENAI_API_KEY or network access; the first fixture removes the key to prove it.
"""

import json
from copy import copy

import pytest

from fyp_evaluation.call_configs.cli import main
from fyp_evaluation.call_configs.feasibility import synthesize_response
from fyp_evaluation.call_configs.run import RunError
from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.llm.errors import AmbiguousError, CredentialsError, FatalRequestError, RetryableError
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body
from fyp_evaluation.llm.store import LockHeldError
from g3_support import BAD, RELEASE, SCHEMAS, copy_release, good_responder, make_runner, shared_inputs, snapshot

CALLS = ["CALL_001", "CALL_002", "CALL_003", "CALL_004", "CALL_005", "CALL_006"]


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def state_of(runner, call_id):
    return runner.status()["calls"][call_id]["status"]


def provenance_of(runner, call_id):
    return parse_json_text((runner.root / f"authoring/call_config_provenance/{call_id}.json").read_text(encoding="utf-8"))


def attempt_state(runner, call_id, attempt_id="A001"):
    return runner.store.read_attempt(call_id, attempt_id)["state"]


# =============================================================================================== sync
def test_sync_success_writes_artifact_provenance_and_record_and_everything_validates(tmp_path):
    runner, fake = make_runner(tmp_path)
    attempt = runner.run_sync("CALL_005")
    assert attempt["state"] == "ACCEPTED" and attempt["transport"] == "sync"
    assert attempt["usage"]["input_tokens"] == 6000 and attempt["estimated_cost_usd"] == 0.037
    root = runner.root
    assert (root / "authoring/call_configs/CALL_005.json").is_file() and (root / "authoring/call_config_record.json").is_file()
    prov = provenance_of(runner, "CALL_005")
    assert prov["accepted"]["attempt_id"] == "A001" and prov["attempts"][0]["state"] == "ACCEPTED"
    assert prov["accepted"]["parsed_response_sha256"] and prov["accepted_history"] == []
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    assert fake.calls == [("create_response", "CALL_005")]
    assert state_of(runner, "CALL_005") == "ACCEPTED"


def test_nothing_credential_like_is_ever_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value-123")
    runner, _ = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    for path in list(tmp_path.rglob("*")):
        if path.is_file():
            assert b"sk-test-secret-value-123" not in path.read_bytes() and b"Authorization" not in path.read_bytes()


def test_sync_validation_rejection_is_recorded_and_needs_an_explicit_retry(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={"CALL_003": BAD})
    attempt = runner.run_sync("CALL_003")
    assert attempt["state"] == "REJECTED" and any("must not be empty" in e for e in attempt["validation_errors"])
    assert not (runner.root / "authoring/call_configs/CALL_003.json").exists()
    prov = provenance_of(runner, "CALL_003")
    assert prov["accepted"] is None and prov["attempts"][0]["state"] == "REJECTED" and prov["attempts"][0]["validation_errors"]
    assert runner.store.read_raw_response("CALL_003", "A001")["response"]["status"] == "completed"  # rejected output is kept for audit
    with pytest.raises(RunError, match="--retry-rejected"):
        runner.run_sync("CALL_003")
    fake.responder = lambda body: response_body(json.dumps(synthesize_response(runner.builder.context("CALL_003"))))
    second = runner.run_sync("CALL_003", retry_rejected=True)
    assert second["state"] == "ACCEPTED" and second["attempt_id"] == "A002"
    assert [a["state"] for a in provenance_of(runner, "CALL_003")["attempts"]] == ["REJECTED", "ACCEPTED"]
    assert runner.validate() == []


def test_rejection_budget_ends_in_awaiting_human(tmp_path):
    runner, _ = make_runner(tmp_path, overrides={"CALL_004": BAD})
    for _ in range(3):
        assert runner.run_sync("CALL_004", retry_rejected=True)["state"] == "REJECTED"
    assert state_of(runner, "CALL_004") == "AWAITING_HUMAN"
    with pytest.raises(RunError, match="AWAITING_HUMAN"):
        runner.run_sync("CALL_004", retry_rejected=True)


@pytest.mark.parametrize("body, fragment", [
    (response_body("", status="incomplete", incomplete_reason="max_output_tokens"), "incomplete response: max_output_tokens"),
    (response_body("", refusal="I cannot help with that"), "model refusal"),
    (response_body("{not json"), "unparseable structured output"),
    (response_body("[1, 2]"), "not a JSON object"),
    (response_body('{"a": 1, "a": 2}'), "unparseable"),
])
def test_incomplete_refusal_and_unparseable_outputs_are_rejected(tmp_path, body, fragment):
    runner, _ = make_runner(tmp_path, responder=lambda _: body)
    attempt = runner.run_sync("CALL_001")
    assert attempt["state"] == "REJECTED" and fragment in attempt["rejection_reason"]
    assert not (runner.root / "authoring/call_configs/CALL_001.json").exists()


def test_provider_failed_status_and_retryable_errors_are_transport_failures_with_their_own_cap(tmp_path):
    runner, fake = make_runner(tmp_path, responder=lambda _: response_body("", status="failed", error={"code": "server_error"}))
    first = runner.run_sync("CALL_001")
    assert first["state"] == "TRANSPORT_FAILED" and not first["fatal"] and first["possibly_billed"]
    fake.responder = good_responder(runner)
    fake.sync_error = RetryableError("RateLimitError: slow down")
    second = runner.run_sync("CALL_001")
    assert second["state"] == "TRANSPORT_FAILED" and second["attempt_id"] == "A002"
    fake.sync_error = RetryableError("timeout")
    assert runner.run_sync("CALL_001")["state"] == "TRANSPORT_FAILED"
    assert state_of(runner, "CALL_001") == "AWAITING_HUMAN"
    with pytest.raises(RunError, match="AWAITING_HUMAN"):
        runner.run_sync("CALL_001")
    assert runner.validate() == []  # nothing accepted, nothing inconsistent


def test_fatal_transport_error_stops_the_run_and_is_not_charged_to_the_budget(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.sync_error = FatalRequestError("AuthenticationError: bad key")
    with pytest.raises(FatalRequestError):
        runner.run_sync("CALL_001")
    attempt = runner.store.read_attempt("CALL_001", "A001")
    assert attempt["state"] == "TRANSPORT_FAILED" and attempt["fatal"] and not attempt["possibly_billed"]
    assert runner.call_state("CALL_001")["transport_failed"] == 0
    assert runner.run_sync("CALL_001")["state"] == "ACCEPTED"


def test_missing_credentials_fail_before_any_attempt_or_request(tmp_path):
    def factory():
        raise CredentialsError("OPENAI_API_KEY is not set. No request was made.")
    runner, _ = make_runner(tmp_path)
    runner._transport, runner._transport_factory = None, factory
    with pytest.raises(CredentialsError):
        runner.run_sync("CALL_001")
    with pytest.raises(CredentialsError):
        runner.submit(["CALL_001"])
    assert not (tmp_path / "audit").exists() and not list((tmp_path / "release").iterdir())


def test_accepted_calls_are_never_regenerated_without_reopen(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    before = snapshot(tmp_path)
    with pytest.raises(RunError, match="already accepted"):
        runner.run_sync("CALL_001")
    assert fake.calls == [("create_response", "CALL_001")]
    assert snapshot(tmp_path) == before


def test_reopen_preserves_the_previous_accepted_version_and_history(tmp_path):
    runner, fake = make_runner(tmp_path)
    first = runner.run_sync("CALL_002")
    old_artifact = (runner.root / "authoring/call_configs/CALL_002.json").read_text(encoding="utf-8")
    old_prov = provenance_of(runner, "CALL_002")["accepted"]
    ctx = runner.builder.context("CALL_002")
    changed = synthesize_response(ctx)
    changed["client"]["name"] = "Sam Wong"
    fake.responder = lambda body: response_body(json.dumps(changed))
    with pytest.raises(RunError, match="--reopen"):
        runner.run_sync("CALL_002")
    second = runner.run_sync("CALL_002", reopen_reason="Prefer a different client persona")
    assert second["state"] == "ACCEPTED" and second["attempt_id"] == "A002"
    prov = provenance_of(runner, "CALL_002")
    assert prov["accepted"]["attempt_id"] == "A002"
    (history,) = prov["accepted_history"]
    assert history["attempt_id"] == first["attempt_id"] and history["reason"] == "Prefer a different client persona"
    assert history["parsed_response"] == old_prov["parsed_response"] and history["artifact_sha256"] == old_prov["artifact_sha256"]
    assert [a["state"] for a in prov["attempts"]] == ["SUPERSEDED", "ACCEPTED"] and prov["attempts"][1]["reopen_reason"]
    assert runner.store.read_attempt("CALL_002", "A001")["state"] == "SUPERSEDED"
    assert (runner.root / "authoring/call_configs/CALL_002.json").read_text(encoding="utf-8") != old_artifact
    record = parse_json_text((runner.root / "authoring/call_config_record.json").read_text(encoding="utf-8"))
    assert record["calls"]["CALL_002"]["previous_artifact_sha256"] == [old_prov["artifact_sha256"]] and record["summary"]["reopened_calls"] == 1
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


def test_a_rejected_reopen_leaves_the_accepted_version_untouched(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_002")
    before = (runner.root / "authoring/call_configs/CALL_002.json").read_bytes()
    fake.responder = lambda body: response_body("{bad")
    assert runner.run_sync("CALL_002", reopen_reason="try again")["state"] == "REJECTED"
    assert (runner.root / "authoring/call_configs/CALL_002.json").read_bytes() == before
    assert provenance_of(runner, "CALL_002")["accepted"]["attempt_id"] == "A001" and runner.validate() == []
    assert state_of(runner, "CALL_002") == "ACCEPTED"


def test_reopen_requires_an_accepted_call(tmp_path):
    runner, _ = make_runner(tmp_path)
    with pytest.raises(RunError, match="only applies to a call with an accepted config"):
        runner.run_sync("CALL_001", reopen_reason="x")


def test_a_stored_but_unjudged_response_is_adjudicated_on_the_next_run_without_a_new_request(tmp_path):
    runner, fake = make_runner(tmp_path)
    attempt, body = runner._new_attempt("CALL_001", "sync")
    runner._store_response(attempt, fake.responder(body), {"transport": "sync"})  # simulate a crash right after the paid response
    assert attempt["state"] == "RESPONDED" and state_of(runner, "CALL_001").startswith("IN_FLIGHT")
    assert runner.eligibility("CALL_001")[0] is False
    runner.collect()  # any mutating command finalises pending responses first
    assert state_of(runner, "CALL_001") == "ACCEPTED" and fake.calls == []
    assert runner.validate() == []


def test_repairing_an_interrupted_acceptance_is_idempotent(tmp_path):
    runner, _ = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    (runner.root / "authoring/call_configs/CALL_001.json").unlink()  # crash after provenance, before the artifact
    attempt = runner.store.read_attempt("CALL_001", "A001")
    attempt["state"] = "RESPONDED"
    runner.store.write_attempt(attempt)
    runner._finalize_responded()
    assert runner.validate() == [] and provenance_of(runner, "CALL_001")["accepted_history"] == []


def test_the_run_lock_blocks_concurrent_mutation(tmp_path):
    runner, _ = make_runner(tmp_path)
    runner.store.root.mkdir(parents=True)
    (runner.store.root / ".lock").write_text("pid 1\n")
    with pytest.raises(LockHeldError, match="Another run holds"):
        runner.run_sync("CALL_001")
    (runner.store.root / ".lock").unlink()
    assert runner.run_sync("CALL_001")["state"] == "ACCEPTED" and not (runner.store.root / ".lock").exists()


def test_an_in_flight_attempt_blocks_a_second_one(tmp_path):
    runner, _ = make_runner(tmp_path)
    runner._new_attempt("CALL_001", "sync")
    ok, reason = runner.eligibility("CALL_001")
    assert not ok and "PREPARED" in reason


# ============================================================================================== batch
def submit_all(runner, fake, calls=CALLS):
    report = runner.submit(list(calls))
    assert report["state"] == "SUBMITTED", report
    return report


def test_submission_intent_and_attempts_are_durable_before_any_network_call(tmp_path):
    seen = {}

    class Spy(FakeTransport):
        def upload_batch_file(self, data, filename):
            ids = runner.store.submission_ids()
            seen["state"] = runner.store.read_submission(ids[0])["state"]
            seen["attempts"] = [runner.store.read_attempt(c, "A001")["state"] for c in CALLS[:2]]
            seen["jsonl"] = runner.store.read_input_jsonl(ids[0]) == data.decode("utf-8")
            return super().upload_batch_file(data, filename)

    spy = Spy()
    runner, fake = make_runner(tmp_path, transport=spy)
    spy.responder = good_responder(runner)
    submit_all(runner, fake, CALLS[:2])
    assert seen == {"state": "INTENT_RECORDED", "attempts": ["PREPARED", "PREPARED"], "jsonl": True}


def test_batch_roundtrip_with_shuffled_output_and_repeat_safe_submit_and_collect(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = submit_all(runner, fake)
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMITTED"
    assert [state_of(runner, c).split(" ")[0] for c in CALLS] == ["IN_FLIGHT"] * 6
    second = runner.submit()
    assert second["submitted"] == 44 and second["submission_id"] == "SUB_0002"  # the 6 in flight are never resubmitted
    third = runner.submit()
    assert third["submitted"] == 0 and third["submission_id"] is None and len(third["skipped"]) == 50
    assert len([c for c in fake.calls if c[0] == "create_batch"]) == 2
    first_batch = report["batch_id"]
    assert runner.collect()["pending"] == ["SUB_0001 (in_progress)", "SUB_0002 (in_progress)"]
    fake.finish_batch(first_batch, shuffle=True)
    result = runner.collect()
    assert result["collected"] == ["SUB_0001"] and result["anomalies"] == [] and not result["fatal"]
    assert [state_of(runner, c) for c in CALLS] == ["ACCEPTED"] * 6
    for call_id in CALLS:
        artifact = parse_json_text((runner.root / f"authoring/call_configs/{call_id}.json").read_text(encoding="utf-8"))
        assert artifact["call_id"] == call_id and artifact["generation"]["attempt_id"] == "A001"
        assert runner.store.read_attempt(call_id, "A001")["batch_id"] == first_batch
        assert runner.store.read_attempt(call_id, "A001")["estimated_cost_usd"] == 0.0185  # batch tier price
    downloads = len([c for c in fake.calls if c[0] == "download_file"])
    again = runner.collect()
    assert again["collected"] == [] and len([c for c in fake.calls if c[0] == "download_file"]) == downloads
    assert runner.validate() == []
    assert runner.store.read_submission("SUB_0001")["state"] == "COLLECTED"


def test_partial_expired_and_failed_requests_are_handled_per_call(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={"CALL_002": BAD})
    report = submit_all(runner, fake)
    ids = {c: f"{c}-G3-A001" for c in CALLS}
    fake.finish_batch(report["batch_id"], status="expired", drop={ids["CALL_006"]}, errors={
        ids["CALL_003"]: (429, "rate_limit_exceeded", "slow"), ids["CALL_004"]: (400, "invalid_request_error", "bad"),
        ids["CALL_005"]: (None, "batch_expired", "expired")})
    result = runner.collect()
    assert result["fatal"] is True and any("permanent request error" in a for a in result["anomalies"])
    outcomes = {c: attempt_state(runner, c) for c in CALLS}
    assert outcomes == {"CALL_001": "ACCEPTED", "CALL_002": "REJECTED", "CALL_003": "TRANSPORT_FAILED",
                        "CALL_004": "TRANSPORT_FAILED", "CALL_005": "TRANSPORT_FAILED", "CALL_006": "TRANSPORT_FAILED"}
    assert runner.store.read_attempt("CALL_004", "A001")["fatal"] is True
    assert runner.store.read_attempt("CALL_003", "A001")["fatal"] is False
    assert "no result line" in runner.store.read_attempt("CALL_006", "A001")["rejection_reason"]
    assert runner.call_state("CALL_003")["transport_failed"] == 1 and runner.call_state("CALL_004")["transport_failed"] == 0
    assert runner.eligibility("CALL_003")[0] and runner.eligibility("CALL_006")[0] and not runner.eligibility("CALL_002")[0]
    assert runner.validate() == []


def test_unknown_missing_and_duplicate_custom_ids(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = submit_all(runner, fake, CALLS[:3])
    ids = [f"{c}-G3-A001" for c in CALLS[:3]]
    fake.finish_batch(report["batch_id"], duplicate={ids[0]}, drop={ids[2]}, unknown=["CALL_099-G3-A001"])
    result = runner.collect()
    assert any("duplicate result lines" in a for a in result["anomalies"]) and any("unknown custom_id" in a for a in result["anomalies"])
    assert attempt_state(runner, "CALL_001") == "TRANSPORT_FAILED" and "duplicate" in runner.store.read_attempt("CALL_001", "A001")["rejection_reason"]
    assert attempt_state(runner, "CALL_002") == "ACCEPTED"
    assert attempt_state(runner, "CALL_003") == "TRANSPORT_FAILED"
    assert not (runner.root / "authoring/call_configs/CALL_001.json").exists()
    assert runner.store.read_submission("SUB_0001")["anomalies"]


def test_a_failed_batch_fails_every_attempt_as_a_configuration_defect(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = submit_all(runner, fake, CALLS[:2])
    fake.finish_batch(report["batch_id"], status="failed", drop={f"{c}-G3-A001" for c in CALLS[:2]})
    result = runner.collect()
    assert result["fatal"] is True
    assert [runner.store.read_attempt(c, "A001")["fatal"] for c in CALLS[:2]] == [True, True]
    assert runner.call_state("CALL_001")["transport_failed"] == 0


def test_request_hashes_are_recomputed_from_the_locally_persisted_canonical_request(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = submit_all(runner, fake, CALLS[:3])
    request = runner.store.attempt_dir("CALL_001", "A001") / "request.json"
    request.write_text(request.read_text(encoding="utf-8").replace("CALL_001", "CALL_00X", 1), encoding="utf-8")
    jsonl = runner.store.root / "submissions" / "SUB_0001.input.jsonl"
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("private", "privatE", 1)
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fake.finish_batch(report["batch_id"], shuffle=True)
    result = runner.collect()
    assert sum("local request hash mismatch" in a for a in result["anomalies"]) == 2
    assert [attempt_state(runner, c) for c in CALLS[:3]] == ["REJECTED", "REJECTED", "ACCEPTED"]
    assert not (runner.root / "authoring/call_configs/CALL_001.json").exists()


def test_submission_unknown_blocks_resubmission_until_recovered_manually(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.create_error, fake.create_succeeds_despite_error = AmbiguousError("timeout while creating"), True
    report = runner.submit(CALLS[:3])
    assert report["state"] == "SUBMISSION_UNKNOWN" and "--attach-batch" in report["advice"]
    sub = runner.store.read_submission("SUB_0001")
    assert sub["state"] == "SUBMISSION_UNKNOWN" and sub["file_id"] and sub["batch_id"] is None
    assert [attempt_state(runner, c) for c in CALLS[:3]] == ["PREPARED"] * 3
    calls_before = list(fake.calls)
    assert runner.submit(CALLS[3:5])["state"] == "SUBMITTED"  # other calls are still fine
    with pytest.raises(RunError, match="attempt A001 is PREPARED"):
        runner.submit(["CALL_001"])
    assert len([c for c in fake.calls if c[0] == "create_batch"]) == 2 and len(fake.calls) == len(calls_before) + 2  # SUB_0002 only: upload + create
    assert runner.call_state("CALL_001")["in_flight"] and runner._blocked_calls()["CALL_001"] == "SUB_0001 is SUBMISSION_UNKNOWN"
    assert runner.status()["submissions"][0]["state"] == "SUBMISSION_UNKNOWN"
    assert runner.collect()["collected"] == []  # an unknown submission is never polled
    hidden = fake.last_hidden_batch
    runner.attach_batch("SUB_0001", hidden)
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMITTED" and attempt_state(runner, "CALL_001") == "SUBMITTED"
    fake.finish_batch(hidden)
    assert runner.collect()["collected"] == ["SUB_0001"]
    assert [state_of(runner, c) for c in CALLS[:3]] == ["ACCEPTED"] * 3


def test_attach_refuses_a_batch_that_does_not_match_the_submission(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.create_error = AmbiguousError("connection reset")
    runner.submit(CALLS[:2])
    other = fake.create_batch(fake.upload_batch_file(b'{"custom_id":"x","method":"POST","url":"/v1/responses","body":{}}\n', "other.jsonl"),
                              {"submission_id": "SUB_0042", "stage": "G3", "dataset_id": "advisory_execution_pilot_v1"})
    with pytest.raises(RunError, match="metadata submission_id"):
        runner.attach_batch("SUB_0001", other)
    with pytest.raises(RunError, match="unknown submission"):
        runner.attach_batch("SUB_0777", other)
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMISSION_UNKNOWN"
    fake.create_error = None
    runner.submit(CALLS[2:3])
    sub2 = runner.store.read_submission("SUB_0002")
    with pytest.raises(RunError, match="applies only to SUBMISSION_UNKNOWN"):
        runner.attach_batch("SUB_0002", sub2["batch_id"])


def test_marking_not_submitted_needs_acknowledgement_and_then_allows_a_new_submission(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.create_error = AmbiguousError("5xx while creating")
    runner.submit(CALLS[:2])
    with pytest.raises(RunError, match="--acknowledge-duplicate-risk"):
        runner.mark_not_submitted("SUB_0001")
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMISSION_UNKNOWN"
    runner.mark_not_submitted("SUB_0001", acknowledge_duplicate_risk=True)
    assert runner.store.read_submission("SUB_0001")["state"] == "NOT_SUBMITTED"
    assert [attempt_state(runner, c) for c in CALLS[:2]] == ["TRANSPORT_FAILED"] * 2
    assert runner.store.read_attempt("CALL_001", "A001")["possibly_billed"] is True
    with pytest.raises(RunError, match="applies only to SUBMISSION_UNKNOWN"):
        runner.mark_not_submitted("SUB_0001", acknowledge_duplicate_risk=True)
    report = runner.submit(CALLS[:2])
    assert report["state"] == "SUBMITTED" and report["submission_id"] == "SUB_0002"
    assert runner.store.read_attempt("CALL_001", "A002")["submission_id"] == "SUB_0002"


def test_definite_creation_failures_never_become_unknown(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.upload_error = RetryableError("connection reset while uploading")
    assert runner.submit(CALLS[:1])["state"] == "NOT_SUBMITTED"
    fake.create_error = RetryableError("RateLimitError")
    assert runner.submit(CALLS[:1])["state"] == "NOT_SUBMITTED"
    fake.create_error = FatalRequestError("BadRequestError: invalid")
    with pytest.raises(FatalRequestError):
        runner.submit(CALLS[:1])
    states = [runner.store.read_submission(s)["state"] for s in runner.store.submission_ids()]
    assert states == ["NOT_SUBMITTED"] * 3
    assert runner.store.read_attempt("CALL_001", "A003")["fatal"] is True and runner.call_state("CALL_001")["transport_failed"] == 2


def test_batch_creation_failure_with_unexpected_exception_is_protected_as_unknown(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.create_error = RuntimeError("unexpected bug in the SDK")
    with pytest.raises(RuntimeError):
        runner.submit(CALLS[:1])
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMISSION_UNKNOWN"
    assert not runner.eligibility("CALL_001")[0]


def test_explicitly_named_ineligible_calls_are_refused_while_the_default_selection_skips_them(tmp_path):
    runner, _ = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    with pytest.raises(RunError, match="already accepted"):
        runner.submit(["CALL_001", "CALL_002"])
    report = runner.submit()
    assert report["submitted"] == 49 and "CALL_001" in report["skipped"]


def test_batch_reopen_keeps_history(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    report = runner.submit(["CALL_001"], reopen_reason="second opinion")
    fake.finish_batch(report["batch_id"])
    runner.collect()
    prov = provenance_of(runner, "CALL_001")
    assert prov["accepted"]["attempt_id"] == "A002" and prov["accepted_history"][0]["attempt_id"] == "A001"
    assert runner.validate() == []


# ====================================================================================== validate/check
def accepted_runner(tmp_path, calls=("CALL_001", "CALL_005")):
    runner, fake = make_runner(tmp_path)
    for call_id in calls:
        runner.run_sync(call_id)
    return runner, fake


def test_check_rebuilds_from_the_stored_accepted_responses_without_any_audit_data(tmp_path):
    runner, _ = accepted_runner(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "audit")
    assert runner.check() == {"missing": [], "changed": [], "extra": []} and runner.validate() == []


def test_validate_detects_edited_stale_missing_extra_and_orphaned_files(tmp_path):
    runner, _ = accepted_runner(tmp_path)
    artifact = runner.root / "authoring/call_configs/CALL_001.json"
    original = artifact.read_bytes()
    artifact.write_bytes(original.replace(b"Alex Tan", b"Alex Teo"))
    errors = runner.validate()
    assert any("artifact hash differs from provenance" in e for e in errors) and any("differs from the rebuild" in e for e in errors)
    artifact.write_bytes(original)
    assert runner.validate() == []
    artifact.unlink()
    assert any("disagree about acceptance" in e for e in runner.validate())
    artifact.write_bytes(original)
    (runner.root / "authoring/call_configs/notes.txt").write_text("x")
    (runner.root / "authoring/call_config_record.json.tmp").write_text("x")
    (runner.root / "authoring/call_config_provenance/CALL_077.json").write_text("{}")
    errors = runner.validate()
    assert any("unexpected file authoring/call_configs/notes.txt" in e for e in errors)
    assert any("orphaned temporary file" in e for e in errors) and any("CALL_077" in e for e in errors)


def test_validate_detects_stale_upstream_hashes(tmp_path):
    runner, _ = accepted_runner(tmp_path)
    stale = copy(shared_inputs())
    stale.spec_hashes = {**stale.spec_hashes, "CALL_001": "0" * 64}
    runner._inputs = stale
    errors = runner.validate()
    assert any("stale call_spec_sha256" in e for e in errors)


def test_validate_detects_record_and_provenance_tampering(tmp_path):
    runner, _ = accepted_runner(tmp_path)
    record = runner.root / "authoring/call_config_record.json"
    original = record.read_bytes()
    record.write_bytes(original.replace(b'"accepted_calls": 2', b'"accepted_calls": 3'))
    assert any("authoring/call_config_record.json" in e for e in runner.validate())
    record.write_bytes(original)
    prov = runner.root / "authoring/call_config_provenance/CALL_001.json"
    saved = prov.read_bytes()
    prov.write_bytes(saved.replace(b"Alex Tan", b"Alex Teo", 1))
    assert any("parsed_response_sha256" in e for e in runner.validate())
    prov.write_bytes(saved)
    record.unlink()
    assert any("missing authoring/call_config_record.json" in e for e in runner.validate())


def test_validate_rejects_an_accepted_response_that_no_longer_satisfies_the_rules(tmp_path):
    runner, _ = accepted_runner(tmp_path)
    prov_path = runner.root / "authoring/call_config_provenance/CALL_001.json"
    prov = parse_json_text(prov_path.read_text(encoding="utf-8"))
    from fyp_evaluation.call_configs.compile import response_hash
    from fyp_evaluation.call_specs.compile import serialize_json

    prov["accepted"]["parsed_response"]["employee"]["name"] = ""
    prov["accepted"]["parsed_response_sha256"] = response_hash(prov["accepted"]["parsed_response"], runner.builder.response_schema)
    prov_path.write_text(serialize_json(prov), encoding="utf-8")
    assert any("accepted response" in e and "must not be empty" in e for e in runner.validate())


def test_require_complete(tmp_path):
    runner, _ = accepted_runner(tmp_path)
    assert runner.validate() == []
    errors = runner.validate(require_complete=True)
    assert len([e for e in errors if "no accepted call config" in e]) == 48


# ================================================================================================ CLI
@pytest.fixture()
def fast_gate(monkeypatch):
    """Skip re-running the multi-second upstream gate for every CLI call; one test below exercises the real gate."""
    monkeypatch.setattr("fyp_evaluation.call_configs.run.load_inputs", lambda root, schema_dir: shared_inputs())


@pytest.fixture()
def cli_args(tmp_path):
    return ["--dataset-root", str(RELEASE), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS)]


def test_two_dry_runs_print_identical_requests_and_mutate_nothing(tmp_path, cli_args, capsys):
    watched = (tmp_path, RELEASE / "authoring", RELEASE / "checklist.json", RELEASE / "check_semantics.json")
    before = snapshot(*watched)
    outputs = []
    for _ in range(2):
        assert main([*cli_args, "--dry-run", "--call-id", "CALL_005", "--call-id", "CALL_011", "--print-request"]) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] and "feasibility: 50/50" in outputs[0]
    requests = [line for line in outputs[0].splitlines() if line.startswith('{"input"')]
    assert len(requests) == 2 and all('"model":"gpt-6.1-sol"' in line for line in requests)
    assert snapshot(*watched) == before and not (tmp_path / "audit").exists()


def test_cli_usage_errors(cli_args):
    for extra in (["--sync"], ["--sync", "--call-id", "CALL_001", "--call-id", "CALL_002"], ["--reopen", "--call-id", "CALL_001", "--sync"],
                  ["--reason", "x", "--status"], ["--acknowledge-duplicate-risk", "--status"], ["--print-request", "--status"],
                  ["--require-complete", "--status"], ["--status", "--check"], []):
        with pytest.raises(SystemExit) as raised:
            main([*cli_args, *extra])
        assert raised.value.code == 2, extra


def test_cli_fake_sync_status_validate_check_and_recovery_commands(tmp_path, capsys, fast_gate):
    release = copy_release(tmp_path / "release")
    args = ["--dataset-root", str(release), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS)]
    holder = {}
    fake = FakeTransport(lambda body: holder["responder"](body))
    from fyp_evaluation.call_configs.run import ConfigRunner
    helper = ConfigRunner(release, audit_root=tmp_path / "unused", schema_dir=SCHEMAS, inputs=shared_inputs())
    holder["responder"] = good_responder(helper)
    assert main([*args, "--sync", "--call-id", "CALL_001"], transport=fake) == 0
    assert "CALL_001 A001 ACCEPTED" in capsys.readouterr().out
    assert main([*args, "--sync", "--call-id", "CALL_001"], transport=fake) == 2
    assert "already accepted" in capsys.readouterr().err
    assert main([*args, "--status"]) == 0 and "CALL_001 ACCEPTED" in capsys.readouterr().out
    assert main([*args, "--validate"]) == 0 and main([*args, "--check"]) == 0
    assert main([*args, "--validate", "--require-complete"]) == 1
    capsys.readouterr()
    fake.create_error = AmbiguousError("lost")
    assert main([*args, "--submit", "--call-id", "CALL_002"], transport=fake) == 1
    assert "SUBMISSION_UNKNOWN" in capsys.readouterr().out
    assert main([*args, "--mark-not-submitted", "SUB_0001"], transport=fake) == 2
    assert "--acknowledge-duplicate-risk" in capsys.readouterr().err
    assert main([*args, "--mark-not-submitted", "SUB_0001", "--acknowledge-duplicate-risk"], transport=fake) == 0
    assert main([*args, "--submit", "--call-id", "CALL_002"], transport=fake) == 0
    fake.finish_batch(next(reversed(fake.batches)))
    assert main([*args, "--collect"], transport=fake) == 0
    assert "CALL_002 A002 ACCEPTED" in capsys.readouterr().out
    assert main([*args, "--validate"]) == 0


def test_cli_reports_missing_credentials_cleanly_and_never_prints_a_key(tmp_path, capsys, monkeypatch, fast_gate):
    release = copy_release(tmp_path / "release")
    args = ["--dataset-root", str(release), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS)]

    def factory():
        raise CredentialsError("OPENAI_API_KEY is not set. No request was made.")
    assert main([*args, "--sync", "--call-id", "CALL_001"], transport_factory=factory) == 2
    err = capsys.readouterr().err
    assert "credentials:" in err and "No request was made" in err and not (tmp_path / "audit").exists()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-print-me")
    assert main([*args, "--status"]) == 0
    captured = capsys.readouterr()
    assert "sk-do-not-print-me" not in captured.out + captured.err


# ===================================================================================== env / transport
def test_dotenv_is_read_from_the_project_root_not_the_cwd_and_the_os_environment_wins(tmp_path, monkeypatch):
    import os

    from fyp_evaluation.llm.env import get_api_key

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("OPENAI_API_KEY=from-dotenv\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("OPENAI_API_KEY=wrong-cwd-file\n")
    monkeypatch.chdir(elsewhere)
    assert get_api_key(project) == "from-dotenv" and "OPENAI_API_KEY" not in os.environ  # os.environ is never mutated
    monkeypatch.setenv("OPENAI_API_KEY", "from-os")
    assert get_api_key(project) == "from-os"
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert get_api_key(project) == "from-dotenv"
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(CredentialsError, match="No request was made"):
        get_api_key(tmp_path / "empty")


def test_gitignore_and_env_example_protect_secrets():
    from g3_support import ROOT

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and ".llm_audit/" in ignored and "!.env.example" in ignored
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=\n" in example and "gpt-" not in example.replace("model ID is NOT configured", "")
    assert not any(line.split("=", 1)[1].strip() for line in example.splitlines() if line.strip() and not line.startswith("#"))


def test_openai_transport_maps_provider_errors_and_never_retries_batch_creation():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    from fyp_evaluation.llm.openai_transport import OpenAITransport

    request = httpx.Request("POST", "https://api.openai.com/v1/batches")

    def status_error(cls, code):
        return cls("boom", response=httpx.Response(code, request=request), body=None)

    class FakeBatches:
        def __init__(self, error):
            self.error, self.kwargs = error, None

        def create(self, **kwargs):
            self.kwargs = kwargs
            raise self.error

    class FakeClient:
        def __init__(self, error):
            self.batches, self.options = FakeBatches(error), None
            self.files = self.responses = None

        def with_options(self, **options):
            self.options = options
            return self

    for error, expected in ((openai.APITimeoutError(request=request), AmbiguousError),
                            (openai.APIConnectionError(request=request), AmbiguousError),
                            (status_error(openai.InternalServerError, 500), AmbiguousError),
                            (status_error(openai.RateLimitError, 429), RetryableError),
                            (status_error(openai.AuthenticationError, 401), FatalRequestError),
                            (status_error(openai.BadRequestError, 400), FatalRequestError)):
        client = FakeClient(error)
        transport = OpenAITransport(client)
        assert client.options == {"max_retries": 0}
        with pytest.raises(expected):
            transport.create_batch("file_1", {"submission_id": "SUB_0001"})
        assert client.batches.kwargs["endpoint"] == "/v1/responses" and client.batches.kwargs["completion_window"] == "24h"
    # reads and generation keep the SDK's own bounded retries; timeouts there are plain retryable errors
    class Responses:
        def create(self, **_):
            raise openai.APITimeoutError(request=request)
    client = FakeClient(None)
    client.responses = Responses()
    with pytest.raises(RetryableError):
        OpenAITransport(client).create_response({"model": "x"})


def test_openai_transport_requires_credentials_before_building_a_client(tmp_path):
    from fyp_evaluation.llm.openai_transport import OpenAITransport

    with pytest.raises(CredentialsError, match="No request was made"):
        OpenAITransport.from_env(tmp_path)
