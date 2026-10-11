"""G3 execution workflow (offline, fake transports): malformed Batch output, historical integrity, interrupted
acceptance, raw provider-file retention, sync-pilot isolation and multi-Batch consolidation."""

import json

import pytest

from fyp_evaluation.call_configs.cli import main
from fyp_evaluation.call_configs.run import ConfigRunner, RunError
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.llm.fake_transport import FakeTransport
from g3_support import BAD, CALL_IDS, RELEASE, SCHEMAS, copy_release, good_responder, make_runner, shared_inputs, snapshot


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def cid(call_id, attempt="A001"):
    return f"{call_id}-G3-{attempt}"


def attempt_state(runner, call_id, attempt="A001"):
    return runner.store.read_attempt(call_id, attempt)["state"]


def provenance_path(runner, call_id):
    return runner.root / f"authoring/call_config_provenance/{call_id}.json"


# ------------------------------------------------------------------------------- A. malformed output
def test_malformed_batch_lines_are_anomalies_and_valid_results_are_still_ingested(tmp_path):
    runner, fake = make_runner(tmp_path)
    calls = CALL_IDS[:6]
    report = runner.submit(calls)
    batch_id = report["batch_id"]
    fake.finish_batch(batch_id, duplicate={cid("CALL_003")}, drop={cid("CALL_004"), cid("CALL_005"), cid("CALL_006")})
    output_id = fake.batches[batch_id]["output_file_id"]
    malformed = ["{}", "[]", '"text"', "7", '{"response": {"status_code": 200}}', '{"custom_id": 7, "response": {}}',
                 '{"custom_id": ""}', "not json",
                 json.dumps({"custom_id": cid("CALL_004"), "response": [1, 2], "error": None}),
                 json.dumps({"custom_id": cid("CALL_005"), "response": {"status_code": 200, "body": "oops"}, "error": None}),
                 json.dumps({"custom_id": cid("CALL_006"), "response": {"status_code": 500, "body": {"error": ["x"]}}, "error": {"code": 5}})]
    fake.files[output_id] = fake.files[output_id] + ("\n".join(malformed) + "\n").encode("utf-8")
    result = runner.collect()
    assert result["collected"] == ["SUB_0001"] and not result["fatal"]
    assert [attempt_state(runner, c) for c in calls] == ["ACCEPTED", "ACCEPTED", "TRANSPORT_FAILED", "TRANSPORT_FAILED",
                                                         "TRANSPORT_FAILED", "TRANSPORT_FAILED"]
    anomalies = "\n".join(result["anomalies"])
    assert anomalies.count("is not a JSON object") == 3 and anomalies.count("has no string custom_id") == 4
    assert "is not valid JSON" in anomalies and "duplicate result lines for CALL_003" in anomalies
    assert "CALL_004-G3-A001: malformed result line" in anomalies and "CALL_005-G3-A001: result line has no response body" in anomalies
    assert "transient" not in anomalies and runner.store.read_attempt("CALL_006", "A001")["fatal"] is False
    assert runner.store.result_file_path("SUB_0001", "output").read_bytes() == fake.files[output_id]  # exact bytes retained
    assert runner.validate() == []


# --------------------------------------------------------------------------- B. historical integrity
def test_validate_rebuilds_historical_accepted_artifacts_and_checks_their_recorded_hashes(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_002")
    runner.run_sync("CALL_002", reopen_reason="second persona")
    assert runner.validate() == []
    path = provenance_path(runner, "CALL_002")
    original = parse_json_text(path.read_text(encoding="utf-8"))
    for mutate in (lambda h: h.update(artifact_sha256="f" * 64), lambda h: h.update(attempt_id="A009"),
                   lambda h: h.update(model="some-other-model")):
        prov = json.loads(json.dumps(original))
        mutate(prov["accepted_history"][0])
        path.write_text(serialize_json(prov), encoding="utf-8")
        assert any("history[0] artifact_sha256 cannot be reproduced" in e for e in runner.validate())
    path.write_text(serialize_json(original), encoding="utf-8")
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


# ------------------------------------------------------------------------ C. interrupted acceptance
def test_accepted_in_provenance_but_missing_artifact_blocks_paid_attempts(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    (runner.root / "authoring/call_configs/CALL_001.json").unlink()  # audit attempt is ACCEPTED: not auto-repairable
    ok, reason = runner.eligibility("CALL_001", retry_rejected=True)
    assert not ok and reason.startswith("INCONSISTENT: provenance records accepted attempt A001")
    for attempt in (lambda: runner.run_sync("CALL_001"), lambda: runner.run_sync("CALL_001", retry_rejected=True),
                    lambda: runner.submit(["CALL_001"]), lambda: runner.run_sync("CALL_001", reopen_reason="x")):
        with pytest.raises(RunError, match="INCONSISTENT"):
            attempt()
    assert runner.status()["calls"]["CALL_001"]["status"] == "INCONSISTENT"
    assert any("disagree about acceptance" in e for e in runner.validate())
    assert fake.calls == [("create_response", "CALL_001")]  # no second paid request was ever made
    # other calls (including technical and rejected retries) are unaffected
    report = runner.submit()
    assert report["submitted"] == 49 and "INCONSISTENT" in report["skipped"]["CALL_001"]


def test_an_interrupted_acceptance_with_a_stored_response_is_completed_by_collect_not_repaid(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    (runner.root / "authoring/call_configs/CALL_001.json").unlink()
    attempt = runner.store.read_attempt("CALL_001", "A001")
    attempt["state"] = "RESPONDED"  # crash after provenance, before the artifact was written
    runner.store.write_attempt(attempt)
    assert "is RESPONDED" in runner.eligibility("CALL_001")[1]
    runner.collect()
    assert runner.status()["calls"]["CALL_001"]["status"] == "ACCEPTED" and runner.validate() == []
    assert fake.calls == [("create_response", "CALL_001")]


# ---------------------------------------------------------------- strict initial production Batch
def test_the_initial_batch_naming_all_50_calls_refuses_to_start_if_any_is_not_eligible(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run_sync("CALL_001")
    with pytest.raises(RunError, match="CALL_001: already accepted"):
        runner.submit(list(CALL_IDS))
    assert runner.store.submission_ids() == [] and fake.calls == [("create_response", "CALL_001")]


# ------------------------------------------------------------------------- pilot isolation (library)
def test_a_sync_pilot_in_its_own_root_and_audit_dir_never_affects_the_production_batch(tmp_path):
    (tmp_path / "g3_sync_pilot").mkdir()
    (tmp_path / "production").mkdir()
    pilot, pilot_fake = make_runner(tmp_path / "g3_sync_pilot")
    assert pilot.run_sync("CALL_010")["state"] == "ACCEPTED"
    production, fake = make_runner(tmp_path / "production")
    report = production.submit(list(CALL_IDS))
    assert report["submitted"] == 50 and report["state"] == "SUBMITTED" and report["skipped"] == {}
    assert not (production.root / "authoring/call_configs").exists()
    assert (pilot.root / "authoring/call_configs/CALL_010.json").is_file()


def test_cli_pilot_on_an_isolated_copy_leaves_the_real_release_and_production_audit_untouched(tmp_path, capsys):
    pilot_root = copy_release(tmp_path / "g3_sync_pilot" / "dataset")
    pilot_audit = tmp_path / "g3_sync_pilot" / "audit"
    production_audit = tmp_path / "production_audit"
    helper = ConfigRunner(pilot_root, audit_root=tmp_path / "unused", schema_dir=SCHEMAS, inputs=shared_inputs())
    fake = FakeTransport(good_responder(helper))
    before = snapshot(RELEASE / "authoring")
    assert main(["--dataset-root", str(pilot_root), "--audit-dir", str(pilot_audit), "--schema-dir", str(SCHEMAS),
                 "--sync", "--call-id", "CALL_010"], transport=fake) == 0
    assert "CALL_010 A001 ACCEPTED" in capsys.readouterr().out
    assert (pilot_root / "authoring/call_configs/CALL_010.json").is_file()
    assert (pilot_audit / "advisory_execution_pilot_v1" / "G3" / "attempts" / "CALL_010" / "A001" / "raw_response.json").is_file()
    assert snapshot(RELEASE / "authoring") == before  # the real release (with its accepted outputs) was not touched
    production_like = copy_release(tmp_path / "production_like")  # frozen inputs only, as before any model output existed
    assert main(["--dataset-root", str(production_like), "--audit-dir", str(production_audit), "--schema-dir", str(SCHEMAS), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count(" eligible\n") == 50 and "NOT eligible" not in out and not production_audit.exists()


# ------------------------------------------------------- multi-Batch consolidation and raw retention
def test_initial_50_call_batch_plus_one_retry_batch_consolidate_into_one_canonical_dataset(tmp_path):
    rejected = ("CALL_002", "CALL_017", "CALL_033")
    runner, fake = make_runner(tmp_path, overrides={c: BAD for c in rejected})
    first = runner.submit(list(CALL_IDS))
    assert first["submission_id"] == "SUB_0001" and first["submitted"] == 50
    fake.finish_batch(first["batch_id"], shuffle=True, drop={cid("CALL_050")},
                      errors={cid("CALL_005"): (500, "server_error", "x"), cid("CALL_040"): (None, "batch_expired", "x")})
    result = runner.collect()
    assert result["collected"] == ["SUB_0001"] and not result["fatal"]
    accepted_first = snapshot(runner.root / "authoring/call_configs")
    assert len(accepted_first) == 44

    # transport failures are eligible by default; rejected outputs only with --retry-rejected
    default = runner.dry_run()
    assert sorted(r["call_id"] for r in default["calls"] if r["eligible"]) == ["CALL_005", "CALL_040", "CALL_050"]
    fake.responder = good_responder(runner)
    retry = runner.submit(retry_rejected=True)
    assert retry["submission_id"] == "SUB_0002" and retry["submitted"] == 6 and retry["batch_id"] != first["batch_id"]
    assert len(runner.submit(retry_rejected=True)["skipped"]) == 50  # nothing left to resubmit while the retry is in flight
    fake.finish_batch(retry["batch_id"])
    runner.collect()

    assert runner.validate(require_complete=True) == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    after = snapshot(runner.root / "authoring/call_configs")
    assert len(after) == 50 and all(after[path] == data for path, data in accepted_first.items())  # first-round outputs kept
    record = parse_json_text((runner.root / "authoring/call_config_record.json").read_text(encoding="utf-8"))
    attempts = {c: entry["accepted_attempt_id"] for c, entry in record["calls"].items()}
    assert record["summary"]["accepted_calls"] == 50 and sorted(c for c, a in attempts.items() if a == "A002") == sorted(
        ["CALL_005", "CALL_040", "CALL_050", *rejected])

    # every submission keeps its own exact input, output and error files
    for submission_id in ("SUB_0001", "SUB_0002"):
        sub = runner.store.read_submission(submission_id)
        assert sub["state"] == "COLLECTED" and sub["batch"]["id"] == sub["batch_id"]
        assert (runner.store.root / "submissions" / f"{submission_id}.input.jsonl").read_bytes() == fake.files[sub["file_id"]]
        assert runner.store.result_file_path(submission_id, "output").read_bytes() == fake.files[sub["batch"]["output_file_id"]]
    sub1 = runner.store.read_submission("SUB_0001")
    assert runner.store.result_file_path("SUB_0001", "error").read_bytes() == fake.files[sub1["batch"]["error_file_id"]]
    assert not runner.store.result_file_path("SUB_0002", "error").exists()
    for call_id in rejected:
        prov = parse_json_text(provenance_path(runner, call_id).read_text(encoding="utf-8"))
        assert [(a["attempt_id"], a["state"], a["submission_id"]) for a in prov["attempts"]] == [
            ("A001", "REJECTED", "SUB_0001"), ("A002", "ACCEPTED", "SUB_0002")]
