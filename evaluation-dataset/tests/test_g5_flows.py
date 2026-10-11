"""G5 runner and CLI flows with a fake Gemini transport: acceptance of every verdict, budgets, capacity, ambiguity, fatal
echoes, reopen/history, and recovery BEFORE eligibility through the real CLI path. Offline; outputs go to tmp only."""

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.llm.errors import AmbiguousError, CapacityError, CredentialsError, FatalRequestError
from fyp_evaluation.llm.fake_gemini_transport import FakeGeminiTransport
from fyp_evaluation.plan_verifications.cli import main
from fyp_evaluation.plan_verifications.run import PlanVerificationRunner, RunError, synthesize_response
from g5_support import SCHEMAS, Clock, good_responder, make_runner, mismatch_report, report_body, runner_factory, shared_inputs, snapshot

CALL, OTHER = "CALL_021", "CALL_041"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def artifact(runner, call_id=CALL):
    return parse_json_text((runner.root / runner.spec.artifact_path(call_id)).read_text(encoding="utf-8"))


def prov(runner, call_id=CALL):
    return parse_json_text((runner.root / runner.spec.provenance_path(call_id)).read_text(encoding="utf-8"))


def leak(ctx):
    return mismatch_report(ctx, "C05", "EV_001", ctx.planned_text("EV_001")[:30])


# ============================================================================== completed verifications
def test_valid_report_is_accepted_reproducible_and_g6_eligible(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = runner.run([CALL])
    (attempt,) = report["attempts"]
    assert attempt["state"] == "ACCEPTED" and fake.calls == [("create_response", CALL)]
    assert attempt["send_count"] == 1
    result = artifact(runner)
    assert result["verdict"] == "VALID" and result["recommended_repair"] is None
    assert result["generation"]["params"] == {"thinking_level": "medium", "service_tier": "flex", "max_output_tokens": 32768}
    assert result["check_plan_sha256"] == shared_inputs().plan_hashes[CALL]
    summary = prov(runner)["attempts"][0]
    assert summary["service_tier"] == "flex" and summary["params"]["service_tier"] == "flex"
    assert summary["upstream_sha256"]["check_plan"] == shared_inputs().plan_hashes[CALL]
    assert summary["estimated_cost_usd"] > 0 and summary["price_table_version"] == "2026-10-11"
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    assert runner.g6_eligibility([CALL])[CALL] == {"eligible": True, "reason": "eligible", "verdict": "VALID"}
    status = runner.status()
    assert status["calls"][CALL]["status"] == "ACCEPTED" and status["verdicts"] == {"VALID": 1, "INVALID": 0, "UNCLEAR": 0}


def test_invalid_report_is_a_completed_outcome_and_is_never_retried(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={CALL: leak})
    assert runner.run([CALL])["attempts"][0]["state"] == "ACCEPTED"
    result = artifact(runner)
    assert result["verdict"] == "INVALID" and result["recommended_repair"] == "G4_ONLY"
    assert result["issues"][0]["evidence"][0]["quote"] == shared_inputs().plans[CALL]["planned_events"][0]["authoring"]["planned_text"][:30].strip()
    with pytest.raises(RunError, match="already accepted"):
        runner.run([CALL])
    assert runner.eligibility(CALL) == (False, "already accepted (use --reopen --reason to supersede it deliberately)")
    assert runner.g6_eligibility([CALL])[CALL]["reason"] == "verdict INVALID"
    assert len(fake.calls) == 1


def test_unclear_report_blocks_g6_and_recommends_review(tmp_path):
    def unclear(ctx):
        return mismatch_report(ctx, "T05", "EV_015", ctx.planned_text("EV_015")[:20], code="PLAN_COHERENCE", scope="REVIEW",
                               assessment="UNCLEAR")
    runner, _ = make_runner(tmp_path, overrides={CALL: unclear})
    runner.run([CALL])
    assert (artifact(runner)["verdict"], artifact(runner)["recommended_repair"]) == ("UNCLEAR", "REVIEW")
    assert runner.g6_eligibility([CALL])[CALL]["eligible"] is False


def test_reopen_reverifies_explicitly_and_keeps_history(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run([CALL])
    first = (runner.root / runner.spec.artifact_path(CALL)).read_text(encoding="utf-8")
    fake.responder = good_responder(runner, {CALL: leak})
    v2 = runner.run([CALL], reopen_reason="re-verify after review")["attempts"][0]
    assert v2["state"] == "ACCEPTED" and v2["attempt_id"] == "A002" and artifact(runner)["verdict"] == "INVALID"
    (history,) = prov(runner)["accepted_history"]
    assert history["reason"] == "re-verify after review" and history["params"]["thinking_level"] == "medium"
    assert runner._rebuild_text(CALL, history) == first.replace("\r\n", "\n")
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


# ============================================================================== rejected reports and budgets
def test_malformed_reports_are_rejected_within_the_response_budget(tmp_path):
    def reversed_order(ctx):
        report = synthesize_response(ctx)
        report["checks"].reverse()
        return report
    runner, fake = make_runner(tmp_path, overrides={CALL: reversed_order})
    assert runner.run([CALL])["attempts"][0]["state"] == "REJECTED"
    with pytest.raises(RunError, match="latest attempt was REJECTED"):
        runner.run([CALL])
    runner.run([CALL], retry_rejected=True)
    runner.run([CALL], retry_rejected=True)
    assert runner.eligibility(CALL, retry_rejected=True)[1].startswith("AWAITING_HUMAN")
    assert runner.status()["calls"][CALL]["status"] == "AWAITING_HUMAN" and len(fake.calls) == 3


def test_truncated_response_is_rejected(tmp_path):
    runner, _ = make_runner(tmp_path, status="incomplete")
    attempt = runner.run([CALL])["attempts"][0]
    assert attempt["state"] == "REJECTED" and "truncated" in attempt["rejection_reason"]


# ============================================================================== capacity, ambiguity, fatal echoes
def test_capacity_rejections_are_resent_with_identical_bytes_inside_one_attempt(tmp_path):
    runner, fake = make_runner(tmp_path, errors=[CapacityError("HTTP 503"), CapacityError("HTTP 429")])
    attempt = runner.run([CALL])["attempts"][0]
    assert attempt["state"] == "ACCEPTED" and attempt["attempt_id"] == "A001" and len(fake.calls) == 3
    assert attempt["send_count"] == 3 == len(fake.calls) and prov(runner)["attempts"][0]["send_count"] == 3  # sends != attempts
    assert [s["outcome"].split(":")[0] for s in prov(runner)["attempts"][0]["sends"]] == ["capacity rejection"] * 2
    slept = runner.slept["slept"]
    assert len(slept) == 2 and 24 <= slept[0] <= 36 and 72 <= slept[1] <= 108


def test_persistent_capacity_stops_the_run(tmp_path):
    runner, fake = make_runner(tmp_path, errors=[CapacityError("HTTP 503")] * 4)
    report = runner.run([CALL, OTHER])
    (attempt,) = report["attempts"]
    assert attempt["state"] == "TRANSPORT_FAILED" and not attempt["possibly_billed"] and "capacity" in report["stopped"]
    assert fake.calls == [("create_response", CALL)] * 4  # OTHER was never sent


def test_ambiguous_outcome_blocks_the_call_until_acknowledged(tmp_path):
    runner, fake = make_runner(tmp_path, errors=[AmbiguousError("APITimeoutError: timed out")])
    report = runner.run([CALL, OTHER])
    attempt = report["attempts"][0]
    assert attempt["state"] == "TRANSPORT_FAILED" and attempt["ambiguous"] and attempt["possibly_billed"]
    assert attempt["send_count"] == 1
    assert len(fake.calls) == 1 and "acknowledge-possible-duplicate" in report["stopped"]
    assert runner.status()["calls"][CALL]["status"] == "AWAITING_ACKNOWLEDGEMENT"
    with pytest.raises(RunError, match="AWAITING_ACKNOWLEDGEMENT"):
        runner.run([CALL])
    again = runner.run([CALL], acknowledge_duplicate=True)
    assert again["acknowledged"] == "A001" and again["attempts"][0]["state"] == "ACCEPTED" and again["attempts"][0]["attempt_id"] == "A002"
    assert prov(runner)["attempts"][0]["ambiguous_acknowledged_at"]


def test_explicit_non_flex_echo_is_fatal_and_stops_before_another_send(tmp_path):
    runner, fake = make_runner(tmp_path, service_tier="standard")
    with pytest.raises(FatalRequestError, match="TIER_MISMATCH"):
        runner.run([CALL, OTHER])
    summary = prov(runner)["attempts"][0]
    assert summary["state"] == "TRANSPORT_FAILED" and summary["fatal"] and summary["possibly_billed"] and summary["service_tier"] == "standard"
    assert len(fake.calls) == 1 and not (runner.root / runner.spec.artifact_path(CALL)).exists()


def test_model_echo_mismatch_is_fatal(tmp_path):
    runner, fake = make_runner(tmp_path, model="gemini-3.6-flash")
    with pytest.raises(FatalRequestError, match="MODEL_MISMATCH"):
        runner.run([CALL, OTHER])
    assert len(fake.calls) == 1


def test_missing_tier_echo_is_accepted_and_recorded_as_not_echoed(tmp_path):
    runner, _ = make_runner(tmp_path, service_tier=None)
    assert runner.run([CALL])["attempts"][0]["state"] == "ACCEPTED"
    summary = prov(runner)["attempts"][0]
    assert summary["service_tier"] is None and summary["params"]["service_tier"] == "flex"
    assert runner.status()["tier_not_echoed"] == 1
    record = parse_json_text((runner.root / runner.spec.record_path).read_text(encoding="utf-8"))
    assert record["summary"]["tier_not_echoed"] == 1 and record["calls"][CALL]["observed_service_tier"] is None


def test_permanent_request_error_stops_the_run(tmp_path):
    runner, fake = make_runner(tmp_path, errors=[FatalRequestError("HTTP 400 unsupported thinking_level")])
    with pytest.raises(FatalRequestError):
        runner.run([CALL, OTHER])
    assert prov(runner)["attempts"][0]["fatal"] and len(fake.calls) == 1


# ============================================================================== recovery before eligibility (CLI path)
def seed(tmp_path, *, raw=None, responded=None):
    """Create one interrupted attempt for CALL in tmp: PREPARED, PREPARED with a stored raw response, or RESPONDED."""
    root, audit = tmp_path / "release", tmp_path / "audit"
    root.mkdir(parents=True, exist_ok=True)
    seeder = PlanVerificationRunner(root, audit_root=audit, schema_dir=SCHEMAS, transport=FakeGeminiTransport(), clock=Clock(),
                                    inputs=shared_inputs())
    with seeder.store.lock():
        attempt, _ = seeder._new_attempt(CALL, "sync")
    ctx = seeder.builder.context(CALL)
    if raw is not None:
        seeder.store.write_raw_response(CALL, attempt["attempt_id"], {"meta": {"transport": "sync"}, "response": raw(ctx)})
    if responded is not None:
        seeder._store_response(attempt, responded(ctx), {"transport": "sync"})
    return root, audit


def cli(root, audit, *extra, transport=None, factory=None):
    argv = ["--dataset-root", str(root), "--audit-dir", str(audit), "--schema-dir", str(SCHEMAS), *extra]
    return main(argv, transport=transport, transport_factory=factory, clock=Clock(), runner_class=runner_factory())


def valid_body(ctx):
    return report_body(synthesize_response(ctx))


def test_prepared_with_raw_response_is_recovered_without_another_send(tmp_path, capsys):
    root, audit = seed(tmp_path, raw=valid_body)
    fake = FakeGeminiTransport()
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 0
    assert fake.calls == [] and f"recovered {CALL} A001 ACCEPTED" in capsys.readouterr().out
    assert (root / "authoring/plan_verifications" / f"{CALL}.json").is_file()


def test_prepared_without_response_becomes_ambiguous_and_blocks_until_acknowledged(tmp_path, capsys):
    root, audit = seed(tmp_path)
    fake = FakeGeminiTransport(lambda body: valid_body(shared_ctx()))
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 0
    assert fake.calls == [] and f"recovered {CALL} A001 TRANSPORT_FAILED" in capsys.readouterr().out
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 2
    assert "AWAITING_ACKNOWLEDGEMENT" in capsys.readouterr().err and fake.calls == []
    assert cli(root, audit, "--run", "--call-id", CALL, "--acknowledge-possible-duplicate", transport=fake) == 0
    assert fake.calls == [("create_response", CALL)] and f"{CALL} A002 ACCEPTED" in capsys.readouterr().out


def shared_ctx():
    from fyp_evaluation.plan_verifications.request import VerificationContext
    return VerificationContext(shared_inputs(), CALL)


def test_responded_attempt_is_adjudicated_without_another_send(tmp_path, capsys):
    root, audit = seed(tmp_path, responded=valid_body)
    fake = FakeGeminiTransport()
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 0
    assert fake.calls == [] and f"recovered {CALL} A001 ACCEPTED" in capsys.readouterr().out


@pytest.mark.parametrize("target", [CALL, OTHER])
def test_stored_fatal_response_stops_before_credentials_and_any_send(tmp_path, capsys, target):
    root, audit = seed(tmp_path, responded=lambda ctx: report_body(synthesize_response(ctx), service_tier="standard"))
    called = []

    def factory():
        called.append(True)
        raise CredentialsError("must not be needed")
    assert cli(root, audit, "--run", "--call-id", target, factory=factory) == 2
    assert called == [] and "TIER_MISMATCH" in capsys.readouterr().err
    record = parse_json_text((root / "authoring/plan_verification_provenance" / f"{CALL}.json").read_text(encoding="utf-8"))
    assert record["attempts"][0]["state"] == "TRANSPORT_FAILED" and record["attempts"][0]["fatal"]


def test_read_only_commands_never_mutate_interrupted_state(tmp_path):
    root, audit = seed(tmp_path, raw=valid_body)
    before = snapshot(root, audit)
    fake = FakeGeminiTransport()
    assert cli(root, audit, "--status", transport=fake) == 0
    assert cli(root, audit, "--dry-run", "--call-id", CALL, transport=fake) == 0
    cli(root, audit, "--validate", transport=fake)
    cli(root, audit, "--check", transport=fake)
    assert snapshot(root, audit) == before and fake.calls == []


def test_recovery_runs_without_credentials_when_nothing_else_is_selected(tmp_path):
    root, audit = seed(tmp_path, responded=valid_body)

    def factory():
        raise CredentialsError("GEMINI_API_KEY is not set")
    assert cli(root, audit, "--run", "--call-id", CALL, factory=factory) == 0
