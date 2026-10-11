"""G3 recovery and consistency (offline, fake transports): interrupted Batch state, bidirectional acceptance
consistency, robustness of historical validation, and Batch-level errors/usage."""

import json
from types import SimpleNamespace

import pytest

from fyp_evaluation.call_configs.compile import response_hash
from fyp_evaluation.call_configs.run import ConfigRunner, RunError
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from g3_support import BAD, CALL_IDS, SCHEMAS, Clock, good_responder, make_runner, shared_inputs


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def attempt_state(runner, call_id, attempt="A001"):
    return runner.store.read_attempt(call_id, attempt)["state"]


def provenance_path(runner, call_id):
    return runner.root / f"authoring/call_config_provenance/{call_id}.json"


def second_runner(runner, fake):
    """A brand-new runner over the same on-disk state (as after a restart)."""
    return ConfigRunner(runner.root, audit_root=runner.audit_root, schema_dir=SCHEMAS, transport=fake, clock=Clock(), inputs=shared_inputs())


def count(fake, name):
    return [call[0] for call in fake.calls].count(name)


# ======================================================================= interrupted Batch state
def test_attempts_left_prepared_after_batch_creation_are_reconciled_and_every_result_processed_once(tmp_path):
    runner, fake = make_runner(tmp_path)
    calls = CALL_IDS[:5]
    state = {"armed": False, "writes": 0}
    original_create = fake.create_batch

    def create(file_id, metadata):
        batch_id = original_create(file_id, metadata)
        state["armed"] = True  # from now on the per-attempt state updates are interrupted
        return batch_id
    fake.create_batch = create
    real_write = runner.store.write_attempt

    def flaky(attempt):
        if state["armed"]:
            state["writes"] += 1
            if state["writes"] == 2:
                raise OSError("simulated interruption while updating attempt states")
        return real_write(attempt)
    runner.store.write_attempt = flaky
    with pytest.raises(OSError, match="simulated interruption"):
        runner.submit(calls)
    sub = runner.store.read_submission("SUB_0001")
    assert sub["state"] == "SUBMITTED" and sub["batch_id"]
    assert [attempt_state(runner, c) for c in calls] == ["SUBMITTED", "PREPARED", "PREPARED", "PREPARED", "PREPARED"]
    assert not (runner.store.root / ".lock").exists()

    restarted = second_runner(runner, fake)
    fake.finish_batch(sub["batch_id"], shuffle=True)
    result = restarted.collect()
    assert result["collected"] == ["SUB_0001"] and result["anomalies"] == [] and not result["fatal"]
    assert [attempt_state(restarted, c) for c in calls] == ["ACCEPTED"] * 5
    assert restarted.store.read_submission("SUB_0001")["state"] == "COLLECTED"
    assert [restarted.store.read_attempt(c, "A001")["batch_id"] for c in calls] == [sub["batch_id"]] * 5
    # exactly one upload, one create and one result download: nothing was resubmitted or double-ingested
    assert (count(fake, "upload_batch_file"), count(fake, "create_batch"), count(fake, "download_file")) == (1, 1, 1)
    for call_id in calls:
        assert len(parse_json_text(provenance_path(restarted, call_id).read_text(encoding="utf-8"))["attempts"]) == 1
    again = restarted.collect()
    assert again["collected"] == [] and count(fake, "download_file") == 1 and count(fake, "retrieve_batch") == 1
    with pytest.raises(RunError, match="already accepted"):
        restarted.submit(calls)
    assert restarted.submit()["submitted"] == 45  # only the never-submitted calls
    assert restarted.validate() == []


def test_a_submission_is_not_marked_collected_while_its_attempts_are_unresolved(tmp_path):
    runner, fake = make_runner(tmp_path)
    calls = CALL_IDS[:3]
    report = runner.submit(calls)
    for call_id in calls[1:]:  # simulate attempts that never advanced
        attempt = runner.store.read_attempt(call_id, "A001")
        attempt["state"] = "PREPARED"
        runner.store.write_attempt(attempt)
    fake.finish_batch(report["batch_id"])
    runner._reconcile_submissions = lambda: None  # prove the guard independently of the reconciliation
    result = runner.collect()
    assert any("still unresolved" in a for a in result["anomalies"])
    assert runner.store.read_submission("SUB_0001")["state"] == "SUBMITTED"
    assert [attempt_state(runner, c) for c in calls] == ["ACCEPTED", "PREPARED", "PREPARED"]
    del runner._reconcile_submissions  # back to the real reconciliation
    result = runner.collect()
    assert result["anomalies"] == [] and runner.store.read_submission("SUB_0001")["state"] == "COLLECTED"
    assert [attempt_state(runner, c) for c in calls] == ["ACCEPTED"] * 3
    assert count(fake, "upload_batch_file") == 1


def test_a_crash_before_the_upload_is_released_for_retry_and_a_crash_after_it_stays_protected(tmp_path):
    runner, fake = make_runner(tmp_path)

    def die(*_):
        raise KeyboardInterrupt  # a hard stop that no except-Exception handler can intercept
    original_upload = fake.upload_batch_file
    fake.upload_batch_file = die
    with pytest.raises(KeyboardInterrupt):
        runner.submit(CALL_IDS[:2])
    assert runner.store.read_submission("SUB_0001")["state"] == "INTENT_RECORDED"
    fake.upload_batch_file = original_upload
    restarted = second_runner(runner, fake)
    assert restarted.submit(CALL_IDS[:2])["submission_id"] == "SUB_0002"  # nothing could exist at the provider: safe to retry
    assert restarted.store.read_submission("SUB_0001")["state"] == "NOT_SUBMITTED"
    assert attempt_state(restarted, "CALL_001", "A001") == "TRANSPORT_FAILED" and attempt_state(restarted, "CALL_001", "A002") == "SUBMITTED"

    fake.create_batch = die
    with pytest.raises(KeyboardInterrupt):
        restarted.submit(CALL_IDS[2:4])
    assert restarted.store.read_submission("SUB_0003")["state"] == "FILE_UPLOADED"
    calls_before = len(fake.calls)
    again = second_runner(runner, fake)
    with pytest.raises(RunError, match="attempt A001 is PREPARED"):
        again.submit(CALL_IDS[2:4])  # never blindly resubmitted
    assert again.store.read_submission("SUB_0003")["state"] == "SUBMISSION_UNKNOWN" and len(fake.calls) == calls_before
    again.mark_not_submitted("SUB_0003", acknowledge_duplicate_risk=True)
    assert again.eligibility("CALL_003")[0]


# ================================================================ bidirectional acceptance consistency
def test_an_artifact_without_accepted_provenance_blocks_paid_selection_and_reopen(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    path = provenance_path(runner, "CALL_001")
    prov = parse_json_text(path.read_text(encoding="utf-8"))
    prov["accepted"], prov["accepted_history"] = None, []
    path.write_text(serialize_json(prov), encoding="utf-8")
    ok, reason = runner.eligibility("CALL_001", retry_rejected=True)
    assert not ok and reason.startswith("INCONSISTENT: authoring/call_configs/CALL_001.json exists but provenance records no accepted attempt")
    ok, reason = runner.eligibility("CALL_001", reopen=True)
    assert not ok and "INCONSISTENT" in reason and "--reopen" in reason
    for attempt in (lambda: runner.run_sync("CALL_001"), lambda: runner.run_sync("CALL_001", reopen_reason="x"), lambda: runner.submit(["CALL_001"])):
        with pytest.raises(RunError, match="INCONSISTENT"):
            attempt()
    assert runner.status()["calls"]["CALL_001"]["status"] == "INCONSISTENT" and fake.calls == [("create_response", "CALL_001")]
    assert any("disagree about acceptance" in e for e in runner.validate())
    # an artifact with no provenance file at all is the same inconsistency
    (runner.root / "authoring/call_configs/CALL_002.json").write_bytes((runner.root / "authoring/call_configs/CALL_001.json").read_bytes())
    assert "INCONSISTENT" in runner.eligibility("CALL_002")[1]
    assert runner.run_sync("CALL_003")["state"] == "ACCEPTED"  # ordinary work on other calls is unaffected


def test_technical_and_rejected_retries_are_unaffected_by_the_consistency_guard(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={"CALL_001": BAD})
    assert runner.run_sync("CALL_001")["state"] == "REJECTED"  # provenance exists with accepted = null and no artifact: consistent
    assert runner.eligibility("CALL_001", retry_rejected=True)[0] and runner.status()["calls"]["CALL_001"]["status"] == "REJECTED"
    fake.responder = good_responder(runner)
    assert runner.run_sync("CALL_001", retry_rejected=True)["state"] == "ACCEPTED"


# ============================================================ robustness of historical validation
def accepted_twice(tmp_path):
    runner, _ = make_runner(tmp_path)
    runner.run_sync("CALL_002")
    runner.run_sync("CALL_002", reopen_reason="second persona")
    runner.run_sync("CALL_005")
    return runner


@pytest.mark.parametrize("where", ["history", "accepted"])
@pytest.mark.parametrize("field", ["model", "artifact_sha256", "attempt_id", "parsed_response", "parsed_response_sha256",
                                   "prompt_template_sha256", "response_schema_sha256"])
def test_missing_required_provenance_fields_are_validation_errors_not_crashes(tmp_path, where, field):
    runner = accepted_twice(tmp_path)
    path = provenance_path(runner, "CALL_002")
    prov = parse_json_text(path.read_text(encoding="utf-8"))
    entry = prov["accepted_history"][0] if where == "history" else prov["accepted"]
    del entry[field]
    path.write_text(serialize_json(prov), encoding="utf-8")
    other = runner.root / "authoring/call_configs/CALL_005.json"
    other.write_bytes(other.read_bytes().replace(b"Alex Tan", b"Alex Teo"))  # another call keeps being validated
    errors = runner.validate()
    assert any(e.startswith("CALL_002: provenance invalid") and field in e for e in errors), errors
    assert any(e.startswith("CALL_005:") and "artifact hash differs" in e for e in errors)
    assert any(item.startswith("CALL_002:") for item in runner.check()["unreproducible"])
    assert runner.status()["calls"]["CALL_002"]["status"] == "INCONSISTENT"
    assert "INCONSISTENT" in runner.eligibility("CALL_002")[1] and runner.eligibility("CALL_003")[0]


def test_unusable_provenance_shapes_never_raise(tmp_path):
    runner = accepted_twice(tmp_path)
    path = provenance_path(runner, "CALL_002")
    good = path.read_text(encoding="utf-8")
    prov = parse_json_text(good)
    for broken in ("{not json", "[]", "null", json.dumps({k: v for k, v in prov.items() if k != "accepted_history"}),
                   json.dumps({k: v for k, v in prov.items() if k != "attempts"})):
        path.write_text(broken, encoding="utf-8")
        assert any(e.startswith("CALL_002: provenance") for e in runner.validate())
        assert runner.check()["unreproducible"] and runner.status()["calls"]["CALL_002"]["status"] == "INCONSISTENT"
    prov["accepted_history"][0]["parsed_response"] = {}
    prov["accepted_history"][0]["parsed_response_sha256"] = response_hash({}, runner.builder.response_schema)
    path.write_text(serialize_json(prov), encoding="utf-8")
    assert any("history[0] artifact_sha256 cannot be reproduced" in e or "cannot be rebuilt" in e for e in runner.validate())
    path.write_text(good, encoding="utf-8")
    assert runner.validate() == []


# ================================================================ Batch-level errors and usage
def test_a_batch_rejected_before_any_result_file_keeps_its_errors_and_usage(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = runner.submit(CALL_IDS[:2])
    errors = {"object": "list", "data": [{"code": "invalid_request", "line": 1, "message": "model not found", "param": "model"}]}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    fake.finish_batch(report["batch_id"], status="failed", drop={f"{c}-G3-A001" for c in CALL_IDS[:2]}, batch_errors=errors, usage=usage)
    result = runner.collect()
    assert result["fatal"] is True
    sub = runner.store.read_submission("SUB_0001")
    assert sub["state"] == "COLLECTED" and sub["batch"]["errors"] == errors and sub["batch"]["usage"] == usage
    assert sub["batch"]["status"] == "failed" and sub["batch"]["output_file_id"] is None and sub["batch"]["error_file_id"] is None
    assert not runner.store.result_file_path("SUB_0001", "output").exists()
    assert json.loads((runner.store.root / "submissions/SUB_0001.json").read_text(encoding="utf-8"))["batch"]["errors"] == errors


def test_a_completed_batch_snapshot_keeps_usage(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = runner.submit(CALL_IDS[:1])
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "input_tokens_details": {"cached_tokens": 0}}
    fake.finish_batch(report["batch_id"], usage=usage)
    runner.collect()
    assert runner.store.read_submission("SUB_0001")["batch"]["usage"] == usage


def test_openai_transport_returns_json_safe_batch_errors_and_usage():
    from fyp_evaluation.llm.openai_transport import OpenAITransport

    class Model:  # stands in for an SDK pydantic model
        def __init__(self, data):
            self.data = data

        def model_dump(self, mode):
            assert mode == "json"
            return self.data

    batch = SimpleNamespace(id="batch_1", status="failed", endpoint="/v1/responses", metadata={"submission_id": "SUB_0001"},
                            input_file_id="file_1", output_file_id=None, error_file_id=None,
                            request_counts=SimpleNamespace(total=2, completed=0, failed=2),
                            errors=Model({"object": "list", "data": [{"code": "x", "line": 1, "message": "m", "param": None}]}),
                            usage=Model({"input_tokens": 0}))

    class Client:
        batches = SimpleNamespace(retrieve=lambda batch_id: batch)

        def with_options(self, **_):
            return self

    result = OpenAITransport(Client()).retrieve_batch("batch_1")
    assert json.loads(json.dumps(result)) == result and result["errors"]["data"][0]["code"] == "x" and result["usage"] == {"input_tokens": 0}
    batch.errors = batch.usage = None
    assert OpenAITransport(Client()).retrieve_batch("batch_1")["errors"] is None
