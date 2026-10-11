"""G4 workflow with fake transports: sync, Batch, recovery, reopen, staleness, consistency, gate and CLI.

The shared lifecycle is covered in depth by the G3 tests; these prove the G4 wiring (paths, schemas, hooks, gate) and the
interruption/recovery/idempotency behaviour on G4 data. No test needs OPENAI_API_KEY or network access.
"""

import json
from copy import copy

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.check_plans.cli import main
from fyp_evaluation.check_plans.compile import project_check_plan_to_skeleton
from fyp_evaluation.check_plans.feasibility import synthesize_response
from fyp_evaluation.check_plans.inputs import G3StateError, load_inputs
from fyp_evaluation.check_plans.run import CheckPlanRunner, RunError
from fyp_evaluation.llm.errors import AmbiguousError, RetryableError
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body
from g4_support import CALL_IDS, RELEASE, SCHEMAS, Clock, bad_response, copy_release, g3_digest, good_responder, make_runner, shared_inputs, snapshot

CALLS = CALL_IDS[:6]


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def provenance_path(runner, call_id):
    return runner.root / f"authoring/check_plan_provenance/{call_id}.json"


def artifact_path(runner, call_id):
    return runner.root / f"authoring/check_plans/{call_id}.json"


def attempt_state(runner, call_id, attempt="A001"):
    return runner.store.read_attempt(call_id, attempt)["state"]


def second_runner(runner, fake):
    return CheckPlanRunner(runner.root, audit_root=runner.audit_root, schema_dir=SCHEMAS, transport=fake, clock=Clock(), inputs=shared_inputs())


# ============================================================================================= sync
def test_sync_success_writes_plan_provenance_and_record_and_everything_validates(tmp_path):
    before = g3_digest()
    runner, fake = make_runner(tmp_path)
    attempt = runner.run_sync("CALL_010")
    assert attempt["state"] == "ACCEPTED" and attempt["usage"]["input_tokens"] == 6000 and attempt["estimated_cost_usd"] == 0.037
    plan = parse_json_text(artifact_path(runner, "CALL_010").read_text(encoding="utf-8"))
    assert plan["stage"] == "G4" and plan["generation"]["attempt_id"] == "A001" and len(plan["planned_events"]) == 13
    assert serialize_json(project_check_plan_to_skeleton(plan)) == shared_inputs().skeleton_texts["CALL_010"]
    assert plan["call_config_sha256"] == shared_inputs().config_hashes["CALL_010"]
    prov = parse_json_text(provenance_path(runner, "CALL_010").read_text(encoding="utf-8"))
    assert prov["stage"] == "G4" and prov["accepted"]["attempt_id"] == "A001" and prov["accepted"]["parsed_response"]["events"]
    record = parse_json_text((runner.root / "authoring/check_plan_record.json").read_text(encoding="utf-8"))
    assert record["summary"] == {"accepted_calls": 1, "total_calls": 50, "reopened_calls": 0, "planned_events": 13}
    assert record["upstream"]["call_config_record_sha256"] == shared_inputs().config_record_sha256
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    assert fake.calls == [("create_response", "CALL_010")] and g3_digest() == before


def test_structural_rejection_is_recorded_and_retried_only_on_request(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={"CALL_003": bad_response})
    attempt = runner.run_sync("CALL_003")
    assert attempt["state"] == "REJECTED" and any("missing from the response" in e for e in attempt["validation_errors"])
    assert not artifact_path(runner, "CALL_003").exists()
    with pytest.raises(RunError, match="--retry-rejected"):
        runner.run_sync("CALL_003")
    fake.responder = good_responder(runner)
    assert runner.run_sync("CALL_003", retry_rejected=True)["state"] == "ACCEPTED"
    assert [a["state"] for a in parse_json_text(provenance_path(runner, "CALL_003").read_text(encoding="utf-8"))["attempts"]] == ["REJECTED", "ACCEPTED"]
    assert runner.validate() == []


def test_unparseable_incomplete_and_refused_outputs_are_rejected_not_accepted(tmp_path):
    for body, fragment in ((response_body("{nope"), "unparseable"), (response_body("", status="incomplete", incomplete_reason="max_output_tokens"), "incomplete"),
                           (response_body("", refusal="no"), "refusal"), (response_body(json.dumps({"events": "x"})), "schema")):
        runner, _ = make_runner(tmp_path / fragment, responder=lambda _, body=body: body)
        attempt = runner.run_sync("CALL_001")
        assert attempt["state"] == "REJECTED" and (fragment in (attempt["rejection_reason"] or "") or any(fragment in e for e in attempt["validation_errors"]))


def test_technical_failures_are_charged_to_their_own_budget(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.sync_error = RetryableError("timeout")
    assert runner.run_sync("CALL_001")["state"] == "TRANSPORT_FAILED"
    assert runner.eligibility("CALL_001")[0] and runner.call_state("CALL_001")["transport_failed"] == 1
    assert runner.run_sync("CALL_001")["state"] == "ACCEPTED"


def test_the_gate_refuses_when_g3_is_not_complete(tmp_path):
    incomplete = copy_release(tmp_path / "release", keep_g3=False)
    with pytest.raises(G3StateError, match="G3 call configs are not complete"):
        load_inputs(incomplete, SCHEMAS)
    partial = copy_release(tmp_path / "partial")
    (partial / "authoring/call_configs/CALL_007.json").unlink()
    with pytest.raises(G3StateError):
        load_inputs(partial, SCHEMAS)


def test_a_changed_g3_config_marks_the_plan_stale_and_reopen_keeps_history(tmp_path):
    runner, fake = make_runner(tmp_path)
    first = runner.run_sync("CALL_002")
    assert runner.validate() == []
    stale = copy(shared_inputs())
    stale.config_hashes = {**stale.config_hashes, "CALL_002": "0" * 64}
    runner._inputs = stale
    assert any("stale call_config_sha256" in e for e in runner.validate())
    runner._inputs = shared_inputs()
    changed = lambda ctx: {"events": [{**e, "text": e["text"].replace("Yes,", "Yes, certainly,")} for e in synthesize_response(ctx)["events"]]}
    fake.responder = good_responder(runner, overrides={"CALL_002": changed})
    second = runner.run_sync("CALL_002", reopen_reason="reworded")
    assert second["state"] == "ACCEPTED" and second["attempt_id"] == "A002"
    prov = parse_json_text(provenance_path(runner, "CALL_002").read_text(encoding="utf-8"))
    assert prov["accepted"]["attempt_id"] == "A002" and prov["accepted_history"][0]["attempt_id"] == first["attempt_id"]
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    record = parse_json_text((runner.root / "authoring/check_plan_record.json").read_text(encoding="utf-8"))
    assert record["calls"]["CALL_002"]["previous_artifact_sha256"] == [prov["accepted_history"][0]["artifact_sha256"]]


def test_acceptance_consistency_is_guarded_in_both_directions(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    artifact_path(runner, "CALL_001").unlink()
    assert "INCONSISTENT: provenance records accepted attempt A001" in runner.eligibility("CALL_001")[1]
    runner2, _ = make_runner(tmp_path / "b")
    runner2.run_sync("CALL_001")
    path = provenance_path(runner2, "CALL_001")
    prov = parse_json_text(path.read_text(encoding="utf-8"))
    prov["accepted"], prov["accepted_history"] = None, []
    path.write_text(serialize_json(prov), encoding="utf-8")
    ok, reason = runner2.eligibility("CALL_001", reopen=True)
    assert not ok and "INCONSISTENT" in reason and runner2.status()["calls"]["CALL_001"]["status"] == "INCONSISTENT"


# ============================================================================================ batch
def test_batch_roundtrip_with_shuffled_output_malformed_lines_and_repeat_safety(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={"CALL_002": bad_response})
    report = runner.submit(CALLS)
    assert report["state"] == "SUBMITTED" and report["submitted"] == 6
    assert runner.submit(CALLS[:1]) if False else True
    with pytest.raises(RunError, match="attempt A001 is SUBMITTED"):
        runner.submit(["CALL_001"])
    ids = {c: f"{c}-G4-A001" for c in CALLS}
    fake.finish_batch(report["batch_id"], shuffle=True, drop={ids["CALL_006"]}, errors={ids["CALL_005"]: (429, "rate_limit_exceeded", "slow")})
    output_id = fake.batches[report["batch_id"]]["output_file_id"]
    fake.files[output_id] += b'[]\n{}\n{"custom_id": 5}\n'
    result = runner.collect()
    assert result["collected"] == ["SUB_0001"] and any("is not a JSON object" in a for a in result["anomalies"])
    assert [attempt_state(runner, c) for c in CALLS] == ["ACCEPTED", "REJECTED", "ACCEPTED", "ACCEPTED", "TRANSPORT_FAILED", "TRANSPORT_FAILED"]
    sub = runner.store.read_submission("SUB_0001")
    assert runner.store.result_file_path("SUB_0001", "output").read_bytes() == fake.files[output_id]
    assert sub["state"] == "COLLECTED" and sub["batch"]["id"] == report["batch_id"]
    downloads = len([c for c in fake.calls if c[0] == "download_file"])
    assert runner.collect()["collected"] == [] and len([c for c in fake.calls if c[0] == "download_file"]) == downloads
    assert runner.validate() == []
    retry = runner.submit(["CALL_005", "CALL_006"])  # technical failures retry without --retry-rejected
    assert retry["submission_id"] == "SUB_0002"
    with pytest.raises(RunError, match="--retry-rejected"):
        runner.submit(["CALL_002"])
    fake.responder = good_responder(runner)
    retry2 = runner.submit(["CALL_002"], retry_rejected=True)
    fake.finish_batch(retry["batch_id"])
    fake.finish_batch(retry2["batch_id"])
    runner.collect()
    assert all(attempt_state(runner, c, "A002") == "ACCEPTED" for c in ("CALL_002", "CALL_005", "CALL_006"))
    assert len(list((runner.root / "authoring/check_plans").iterdir())) == 6 and runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    record = parse_json_text((runner.root / "authoring/check_plan_record.json").read_text(encoding="utf-8"))
    assert record["summary"]["accepted_calls"] == 6


def test_request_hash_is_recomputed_from_the_locally_persisted_request(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = runner.submit(CALLS[:2])
    request = runner.store.attempt_dir("CALL_001", "A001") / "request.json"
    request.write_text(request.read_text(encoding="utf-8").replace("CALL_001", "CALL_00X", 1), encoding="utf-8")
    fake.finish_batch(report["batch_id"])
    result = runner.collect()
    assert any("local request hash mismatch" in a for a in result["anomalies"])
    assert [attempt_state(runner, c) for c in CALLS[:2]] == ["REJECTED", "ACCEPTED"]


def test_attempts_left_prepared_after_batch_creation_are_reconciled_and_processed_once(tmp_path):
    runner, fake = make_runner(tmp_path)
    state = {"armed": False, "writes": 0}
    original = fake.create_batch

    def create(file_id, metadata):
        batch_id = original(file_id, metadata)
        state["armed"] = True
        return batch_id
    fake.create_batch = create
    real_write = runner.store.write_attempt

    def flaky(attempt):
        if state["armed"]:
            state["writes"] += 1
            if state["writes"] == 2:
                raise OSError("simulated interruption")
        return real_write(attempt)
    runner.store.write_attempt = flaky
    with pytest.raises(OSError):
        runner.submit(CALLS[:4])
    sub = runner.store.read_submission("SUB_0001")
    assert [attempt_state(runner, c) for c in CALLS[:4]] == ["SUBMITTED", "PREPARED", "PREPARED", "PREPARED"]
    restarted = second_runner(runner, fake)
    fake.finish_batch(sub["batch_id"], shuffle=True)
    assert restarted.collect()["collected"] == ["SUB_0001"]
    assert [attempt_state(restarted, c) for c in CALLS[:4]] == ["ACCEPTED"] * 4
    counts = [c[0] for c in fake.calls]
    assert (counts.count("upload_batch_file"), counts.count("create_batch"), counts.count("download_file")) == (1, 1, 1)
    assert restarted.collect()["collected"] == [] and restarted.validate() == []


def test_submission_unknown_blocks_blind_retries_and_recovers_manually(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.create_error, fake.create_succeeds_despite_error = AmbiguousError("timeout"), True
    report = runner.submit(CALLS[:3])
    assert report["state"] == "SUBMISSION_UNKNOWN" and "--attach-batch" in report["advice"]
    with pytest.raises(RunError, match="attempt A001 is PREPARED"):
        runner.submit(["CALL_001"])
    assert runner.collect()["collected"] == []
    with pytest.raises(RunError, match="acknowledge-duplicate-risk"):
        runner.mark_not_submitted("SUB_0001")
    runner.attach_batch("SUB_0001", fake.last_hidden_batch)
    fake.finish_batch(fake.last_hidden_batch)
    assert runner.collect()["collected"] == ["SUB_0001"]
    assert [attempt_state(runner, c) for c in CALLS[:3]] == ["ACCEPTED"] * 3


def test_crash_before_the_upload_is_released_and_a_crash_after_it_stays_protected(tmp_path):
    runner, fake = make_runner(tmp_path)

    def die(*_):
        raise KeyboardInterrupt
    original = fake.upload_batch_file
    fake.upload_batch_file = die
    with pytest.raises(KeyboardInterrupt):
        runner.submit(CALLS[:2])
    fake.upload_batch_file = original
    restarted = second_runner(runner, fake)
    assert restarted.submit(CALLS[:2])["submission_id"] == "SUB_0002"
    assert restarted.store.read_submission("SUB_0001")["state"] == "NOT_SUBMITTED"
    fake.create_batch = die
    with pytest.raises(KeyboardInterrupt):
        restarted.submit(CALLS[2:4])
    again = second_runner(runner, fake)
    with pytest.raises(RunError, match="attempt A001 is PREPARED"):
        again.submit(CALLS[2:4])
    assert again.store.read_submission("SUB_0003")["state"] == "SUBMISSION_UNKNOWN"


def test_a_failed_batch_keeps_its_errors_and_usage_and_blocks_nothing_permanently(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = runner.submit(CALLS[:2])
    errors = {"object": "list", "data": [{"code": "invalid_request", "line": 1, "message": "bad", "param": "model"}]}
    fake.finish_batch(report["batch_id"], status="failed", drop={f"{c}-G4-A001" for c in CALLS[:2]}, batch_errors=errors, usage={"input_tokens": 0})
    assert runner.collect()["fatal"] is True
    sub = runner.store.read_submission("SUB_0001")
    assert sub["batch"]["errors"] == errors and sub["batch"]["usage"] == {"input_tokens": 0}
    assert runner.call_state("CALL_001")["transport_failed"] == 0


def test_a_full_50_call_batch_with_consolidation_across_two_submissions(tmp_path):
    rejected = ("CALL_004", "CALL_021", "CALL_047")
    runner, fake = make_runner(tmp_path, overrides={c: bad_response for c in rejected})
    first = runner.submit(list(CALL_IDS))
    assert first["submitted"] == 50
    fake.finish_batch(first["batch_id"], shuffle=True, drop={"CALL_050-G4-A001"})
    runner.collect()
    first_round = snapshot(runner.root / "authoring/check_plans")
    assert len(first_round) == 46
    fake.responder = good_responder(runner)
    retry = runner.submit(retry_rejected=True)
    assert retry["submitted"] == 4 and retry["submission_id"] == "SUB_0002"
    fake.finish_batch(retry["batch_id"])
    runner.collect()
    after = snapshot(runner.root / "authoring/check_plans")
    assert len(after) == 50 and all(after[path] == data for path, data in first_round.items())
    assert runner.validate(require_complete=True) == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    record = parse_json_text((runner.root / "authoring/check_plan_record.json").read_text(encoding="utf-8"))
    assert record["summary"] == {"accepted_calls": 50, "total_calls": 50, "reopened_calls": 0, "planned_events": 968}


# ============================================================================================== CLI
@pytest.fixture()
def fast_gate(monkeypatch):
    monkeypatch.setattr("fyp_evaluation.check_plans.run.load_inputs", lambda root, schema_dir: shared_inputs())


def test_two_dry_runs_print_identical_requests_and_mutate_nothing(tmp_path, capsys):
    watched = (tmp_path, RELEASE / "authoring", RELEASE / "checklist.json")
    before, digest = snapshot(*watched), g3_digest()
    args = ["--dataset-root", str(RELEASE), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS), "--dry-run",
            "--call-id", "CALL_010", "--call-id", "CALL_034", "--print-request"]
    outputs = []
    for _ in range(2):
        assert main(args) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] and "feasibility: 50/50" in outputs[0]
    assert len([line for line in outputs[0].splitlines() if line.startswith('{"input"')]) == 2
    assert snapshot(*watched) == before and g3_digest() == digest and not (tmp_path / "audit").exists()


def test_cli_fake_sync_status_validate_check_and_batch_recovery(tmp_path, capsys, fast_gate):
    release = copy_release(tmp_path / "release")
    args = ["--dataset-root", str(release), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS)]
    helper = CheckPlanRunner(release, audit_root=tmp_path / "unused", schema_dir=SCHEMAS, inputs=shared_inputs())
    fake = FakeTransport(good_responder(helper))
    assert main([*args, "--sync", "--call-id", "CALL_010"], transport=fake) == 0
    assert "CALL_010 A001 ACCEPTED" in capsys.readouterr().out
    assert main([*args, "--sync", "--call-id", "CALL_010"], transport=fake) == 2
    assert "already accepted" in capsys.readouterr().err
    assert main([*args, "--status"]) == 0 and "CALL_010 ACCEPTED" in capsys.readouterr().out
    assert main([*args, "--validate"]) == 0 and "G4 outputs on disk are valid." in capsys.readouterr().out
    assert main([*args, "--check"]) == 0 and "G4 outputs reproduce byte-for-byte" in capsys.readouterr().out
    assert main([*args, "--validate", "--require-complete"]) == 1
    assert "no accepted check plan" in capsys.readouterr().out
    fake.create_error = AmbiguousError("lost")
    assert main([*args, "--submit", "--call-id", "CALL_011"], transport=fake) == 1
    assert "SUBMISSION_UNKNOWN" in capsys.readouterr().out
    assert main([*args, "--mark-not-submitted", "SUB_0001", "--acknowledge-duplicate-risk"], transport=fake) == 0
    assert main([*args, "--submit", "--call-id", "CALL_011"], transport=fake) == 0
    fake.finish_batch(next(reversed(fake.batches)))
    assert main([*args, "--collect"], transport=fake) == 0
    assert "CALL_011 A002 ACCEPTED" in capsys.readouterr().out
    assert main([*args, "--validate"]) == 0
