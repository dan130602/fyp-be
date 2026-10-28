"""G7 runner and CLI flows with a fake Gemini transport: acceptance of every verdict, budgets, capacity, ambiguity, fatal echoes,
the G7A preflight, acceptance ordering under concurrent change, reopen/history against the exact G6 dialogue, recovery BEFORE
eligibility through the real CLI path, and the isolated controls. Offline; outputs go to tmp only, every mutation to a copy."""

import hashlib
import shutil

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.dialogue_verifications import controls
from fyp_evaluation.dialogue_verifications.cli import main
from fyp_evaluation.dialogue_verifications.run import DialogueVerificationRunner, RunError, synthesize_response
from fyp_evaluation.dialogues.config import DIALOGUE_UNREADABLE
from fyp_evaluation.dialogues.inputs import read_dialogue
from fyp_evaluation.llm.errors import AmbiguousError, CapacityError, CredentialsError, FatalRequestError
from fyp_evaluation.llm.fake_gemini_transport import FakeGeminiTransport
from fyp_evaluation.llm.store import StageWriteLockBusy
from g7_support import (
    RELEASE, SCHEMAS, Clock, fresh_runner, good_responder, load_inputs, make_runner, make_world, mismatch_report, report_body,
    runner_factory, shared_inputs, snapshot,
)

CALL, OTHER = "CALL_016", "CALL_041"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def artifact(runner, call_id=CALL):
    return parse_json_text((runner.root / runner.spec.artifact_path(call_id)).read_text(encoding="utf-8"))


def prov(runner, call_id=CALL):
    return parse_json_text((runner.root / runner.spec.provenance_path(call_id)).read_text(encoding="utf-8"))


def state(runner, call_id=CALL, attempt="A001"):
    return runner.store.read_attempt(call_id, attempt)["state"]


def rewrite(path, mutate):
    data = parse_json_text(path.read_bytes().decode("utf-8"))
    mutate(data)
    path.write_bytes(serialize_json(data).encode("utf-8"))


def digest(*relative):
    """One digest over the committed production G5/G6 outputs (they must never change while G7 tests run)."""
    h = hashlib.sha256()
    for directory in relative:
        for path in sorted((RELEASE / directory).rglob("*")):
            if path.is_file():
                h.update(path.as_posix().encode() + path.read_bytes())
    return h.hexdigest()


PRODUCTION = ("authoring/dialogues", "authoring/dialogue_provenance", "authoring/plan_verifications", "authoring/plan_verification_provenance")


def leak(ctx):  # C04 must FAIL: a fee statement in the closing turn would be an omission leak
    return mismatch_report(ctx, "C04", {"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}, code="OMISSION_LEAKAGE")


def wrong_final_instruction(ctx):
    return mismatch_report(ctx, "T05", {"turn_id": ctx.turn_of_event["EV_016"], "quote": ctx.planned_text("EV_016")[:20]})


def serious_quality(ctx):
    report = synthesize_response(ctx)
    report["quality_findings"] = [{"quality_id": "Q1", "severity": "SERIOUS", "code": "ARTIFICIAL_EXTENSION",
                                  "evidence": [{"turn_id": 62, "quote": ctx.turns[62]["text"][:40]}], "explanation": "Administrative filler."}]
    return report


# ============================================================================== completed verifications
def test_valid_report_is_accepted_reproducible_and_g8_eligible_and_production_is_untouched(tmp_path):
    before = digest(*PRODUCTION)
    runner, fake = make_runner(tmp_path)
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "ACCEPTED" and fake.calls == [("create_response", CALL)] and attempt["send_count"] == 1
    result = artifact(runner)
    assert result["verdict"] == "VALID" and result["recommended_repair"] is None and result["quality_review"] == "CLEAR"
    assert result["generation"]["params"] == {"thinking_level": "high", "service_tier": "flex", "max_output_tokens": 65536}
    inputs = shared_inputs()
    assert result["dialogue_sha256"] == inputs.dialogue_hashes[CALL] and result["plan_verification_sha256"] == inputs.report_hashes[CALL]
    summary = prov(runner)["attempts"][0]
    assert summary["service_tier"] == "flex" and summary["params"]["thinking_level"] == "high" and summary["transport"] == "sync"
    assert summary["upstream_sha256"]["dialogue"] == inputs.dialogue_hashes[CALL]
    assert summary["upstream_sha256"]["plan_verification"] == inputs.report_hashes[CALL]
    assert summary["estimated_cost_usd"] > 0 and summary["price_table_version"] == "2026-10-11"
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    assert runner.g8_eligibility([CALL])[CALL] == {"eligible": True, "reason": "eligible", "verdict": "VALID", "quality_review": "CLEAR"}
    status = runner.status()
    assert status["calls"][CALL]["status"] == "ACCEPTED" and status["verdicts"] == {"VALID": 1, "INVALID": 0, "UNCLEAR": 0}
    assert status["g8_eligible"] == 1 and status["quality_reviews"] == {"CLEAR": 1, "REVIEW_REQUIRED": 0}
    record = parse_json_text((runner.root / runner.spec.record_path).read_text(encoding="utf-8"))
    assert record["calls"][CALL]["verdict"] == "VALID" and record["upstream"]["dialogue_record_sha256"] == inputs.g6_record_sha256
    assert digest(*PRODUCTION) == before  # G5/G6 outputs are byte-identical


def test_invalid_report_is_a_completed_outcome_and_is_never_retried(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={CALL: leak})
    assert runner.run([CALL])["attempts"][0]["state"] == "ACCEPTED"
    result = artifact(runner)
    assert result["verdict"] == "INVALID" and result["recommended_repair"] == "G6_ONLY"
    assert result["issues"][0]["evidence"][0]["turn_id"] == 66 and result["issues"][0]["evidence"][0]["speaker"] == "EMPLOYEE"
    with pytest.raises(RunError, match="already accepted"):
        runner.run([CALL])
    assert runner.eligibility(CALL) == (False, "already accepted (use --reopen --reason to supersede it deliberately)")
    row = runner.g8_eligibility([CALL])[CALL]
    assert row["eligible"] is False and row["reason"] == "verdict INVALID" and row["verdict"] == "INVALID"
    assert len(fake.calls) == 1  # an INVALID report is never retried to obtain a different verdict


def test_unclear_report_is_accepted_without_retry_and_recommends_review(tmp_path):
    def unclear(ctx):
        return mismatch_report(ctx, "T05", None, code="MISSING_SUPPORT", scope="REVIEW", assessment="UNCLEAR", evidence_on_issue=False)
    runner, fake = make_runner(tmp_path, overrides={CALL: unclear})
    runner.run([CALL])
    assert (artifact(runner)["verdict"], artifact(runner)["recommended_repair"]) == ("UNCLEAR", "REVIEW")
    assert runner.g8_eligibility([CALL])[CALL]["eligible"] is False and len(fake.calls) == 1


def test_an_edge_case_mismatch_alone_makes_the_report_invalid(tmp_path):
    def edge(ctx):
        report = synthesize_response(ctx)
        report["edge_cases"][1].update(assessment="MISMATCH", issue_ids=["I1"], evidence=[])
        report["issues"] = [{"issue_id": "I1", "assessment": "MISMATCH", "code": "MISSING_SUPPORT", "question_ids": [], "assignment_ids": ["EA_028"],
                             "evidence": [], "explanation": "Never spoken.", "repair_scope": "G6_ONLY"}]
        return report
    runner, _ = make_runner(tmp_path, overrides={CALL: edge})
    runner.run([CALL])
    assert artifact(runner)["verdict"] == "INVALID" and runner.g8_eligibility([CALL])[CALL]["eligible"] is False


def test_serious_quality_findings_stay_prominent_but_never_block_g8(tmp_path, capsys):
    runner, _ = make_runner(tmp_path, overrides={CALL: serious_quality})
    runner.run([CALL])
    assert artifact(runner)["verdict"] == "VALID" and artifact(runner)["quality_review"] == "REVIEW_REQUIRED"
    assert runner.g8_eligibility([CALL])[CALL] == {"eligible": True, "reason": "eligible", "verdict": "VALID", "quality_review": "REVIEW_REQUIRED"}
    status = runner.status()
    assert status["quality_reviews"]["REVIEW_REQUIRED"] == 1 and status["g8_eligible"] == 1
    record = parse_json_text((runner.root / runner.spec.record_path).read_text(encoding="utf-8"))
    assert record["calls"][CALL]["quality_review"] == "REVIEW_REQUIRED" and record["summary"]["quality_reviews"] == {"CLEAR": 0, "REVIEW_REQUIRED": 1}
    assert main(["--dataset-root", str(runner.root), "--audit-dir", str(runner.audit_root), "--schema-dir", str(SCHEMAS), "--status"],
                clock=Clock(), runner_class=runner_factory()) == 0
    out = capsys.readouterr().out
    assert f"{CALL} ACCEPTED verdict=VALID quality_review=REVIEW_REQUIRED g8_eligible=True" in out and "QUALITY_REVIEW_REQUIRED" in out


def test_accepted_reports_are_skipped_on_resume_and_reopen_keeps_history(tmp_path):
    runner, fake = make_runner(tmp_path)
    runner.run([CALL, OTHER])
    assert len(fake.calls) == 2
    again = runner.run()  # a resume of everything skips both accepted reports
    assert CALL not in [a["call_id"] for a in again["attempts"]] and OTHER not in [a["call_id"] for a in again["attempts"]]
    assert len(fake.calls) == 2 + len(again["attempts"])
    first = (runner.root / runner.spec.artifact_path(CALL)).read_text(encoding="utf-8")
    fake.responder = good_responder(runner, {CALL: leak})
    v2 = runner.run([CALL], reopen_reason="re-verify after review")["attempts"][0]
    assert v2["state"] == "ACCEPTED" and v2["attempt_id"] == "A002" and artifact(runner)["verdict"] == "INVALID"
    (history,) = prov(runner)["accepted_history"]
    assert history["reason"] == "re-verify after review" and history["params"]["thinking_level"] == "high"
    assert runner._rebuild_text(CALL, history) == first.replace("\r\n", "\n")
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    with pytest.raises(RunError, match="--reopen needs --call-id"):
        runner.run(reopen_reason="x")


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


def test_an_ungrounded_quote_is_a_rejection_with_the_reason_recorded(tmp_path):
    def invented(ctx):
        return mismatch_report(ctx, "C04", {"turn_id": 66, "quote": "this was never said in the call"}, code="OMISSION_LEAKAGE")
    runner, _ = make_runner(tmp_path, overrides={CALL: invented})
    attempt = runner.run([CALL])["attempts"][0]
    assert attempt["state"] == "REJECTED" and any("not a substring of the cited turn" in e for e in attempt["validation_errors"])
    assert not (runner.root / runner.spec.artifact_path(CALL)).exists()


def test_invalid_json_and_incomplete_provider_output_are_rejected(tmp_path):
    from fyp_evaluation.llm.fake_gemini_transport import interaction_body
    runner, _ = make_runner(tmp_path, responder=lambda body: interaction_body("not json"))
    assert runner.run([CALL])["attempts"][0]["state"] == "REJECTED"
    runner, _ = make_runner(tmp_path / "b", status="incomplete")
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


def test_persistent_capacity_stops_the_run_after_at_most_four_sends(tmp_path):
    runner, fake = make_runner(tmp_path, errors=[CapacityError("HTTP 503")] * 4)
    report = runner.run([CALL, OTHER])
    (attempt,) = report["attempts"]
    assert attempt["state"] == "TRANSPORT_FAILED" and not attempt["possibly_billed"] and "capacity" in report["stopped"]
    assert fake.calls == [("create_response", CALL)] * 4  # OTHER was never sent


def test_ambiguous_outcome_blocks_the_call_until_acknowledged(tmp_path):
    runner, fake = make_runner(tmp_path, errors=[AmbiguousError("APITimeoutError: timed out")])
    report = runner.run([CALL, OTHER])
    attempt = report["attempts"][0]
    assert attempt["state"] == "TRANSPORT_FAILED" and attempt["ambiguous"] and attempt["possibly_billed"] and attempt["send_count"] == 1
    assert len(fake.calls) == 1 and "acknowledge-possible-duplicate" in report["stopped"]
    assert runner.status()["calls"][CALL]["status"] == "AWAITING_ACKNOWLEDGEMENT"
    assert runner.usage_report()["totals"]["unpriced_possibly_billed"] == 1  # potentially billed, reported separately
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


def test_model_echo_mismatch_and_permanent_request_errors_are_fatal(tmp_path):
    runner, fake = make_runner(tmp_path / "a", model="gemini-3.6-flash")
    with pytest.raises(FatalRequestError, match="MODEL_MISMATCH"):
        runner.run([CALL, OTHER])
    assert len(fake.calls) == 1
    runner, fake = make_runner(tmp_path / "b", errors=[FatalRequestError("HTTP 400 unsupported thinking_level")])
    with pytest.raises(FatalRequestError):
        runner.run([CALL, OTHER])
    assert prov(runner)["attempts"][0]["fatal"] and len(fake.calls) == 1


def test_missing_tier_echo_is_accepted_and_recorded_as_not_echoed(tmp_path):
    runner, _ = make_runner(tmp_path, service_tier=None)
    assert runner.run([CALL])["attempts"][0]["state"] == "ACCEPTED"
    assert prov(runner)["attempts"][0]["service_tier"] is None and runner.status()["tier_not_echoed"] == 1


# ============================================================================== recovery before eligibility (CLI path)
def seed(tmp_path, *, raw=None, responded=None):
    """Create one interrupted attempt for CALL in tmp: PREPARED, PREPARED with a stored raw response, or RESPONDED."""
    root, audit = tmp_path / "release", tmp_path / "audit"
    root.mkdir(parents=True, exist_ok=True)
    seeder = DialogueVerificationRunner(root, audit_root=audit, schema_dir=SCHEMAS, transport=FakeGeminiTransport(), clock=Clock(),
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
    assert (root / "authoring/dialogue_verifications" / f"{CALL}.json").is_file()


def test_prepared_without_response_becomes_ambiguous_and_blocks_until_acknowledged(tmp_path, capsys):
    root, audit = seed(tmp_path)
    ctx = DialogueVerificationRunner(root, audit_root=audit, schema_dir=SCHEMAS, inputs=shared_inputs()).builder.context(CALL)
    fake = FakeGeminiTransport(lambda body: valid_body(ctx))
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 0
    assert fake.calls == [] and f"recovered {CALL} A001 TRANSPORT_FAILED" in capsys.readouterr().out
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 2
    assert "AWAITING_ACKNOWLEDGEMENT" in capsys.readouterr().err and fake.calls == []
    assert cli(root, audit, "--run", "--call-id", CALL, "--acknowledge-possible-duplicate", transport=fake) == 0
    assert fake.calls == [("create_response", CALL)] and f"{CALL} A002 ACCEPTED" in capsys.readouterr().out


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
    record = parse_json_text((root / "authoring/dialogue_verification_provenance" / f"{CALL}.json").read_text(encoding="utf-8"))
    assert record["attempts"][0]["state"] == "TRANSPORT_FAILED" and record["attempts"][0]["fatal"]


def test_recovery_runs_without_credentials_when_nothing_else_is_selected(tmp_path):
    root, audit = seed(tmp_path, responded=valid_body)

    def factory():
        raise CredentialsError("GEMINI_API_KEY is not set")
    assert cli(root, audit, "--run", "--call-id", CALL, factory=factory) == 0


def test_read_only_commands_never_mutate_state_or_need_credentials(tmp_path, capsys):
    root, audit = seed(tmp_path, raw=valid_body)
    before = snapshot(root, audit)
    fake = FakeGeminiTransport()

    def factory():
        raise CredentialsError("offline modes never ask for credentials")
    for mode in (["--status"], ["--dry-run", "--call-id", CALL], ["--dry-run", "--print-request", "--call-id", CALL], ["--validate"], ["--check"]):
        cli(root, audit, *mode, transport=fake, factory=factory)
    out = capsys.readouterr().out
    assert snapshot(root, audit) == before and fake.calls == []
    assert "UTF-8 bytes" in out and "(approximate)" in out and "feasibility: 50/50" in out and "not a promise of cost" in out


def test_cli_exit_codes_and_usage_errors(tmp_path, capsys):
    root, audit = tmp_path / "release", tmp_path / "audit"
    root.mkdir()
    fake = FakeGeminiTransport(lambda body: report_body(leak(shared_ctx())))
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 0  # a completed INVALID verdict is a successful outcome
    assert cli(root, audit, "--run", "--call-id", CALL, transport=fake) == 2  # already accepted
    assert "already accepted" in capsys.readouterr().err
    for bad in (["--run", "--reopen"], ["--run", "--reason", "x"], ["--status", "--limit", "1"], ["--dry-run", "--acknowledge-possible-duplicate"],
                ["--validate", "--print-request"], ["--check", "--require-complete"]):
        with pytest.raises(SystemExit):
            cli(root, audit, *bad)


def shared_ctx():
    from fyp_evaluation.dialogue_verifications.request import DialogueVerificationContext
    return DialogueVerificationContext(shared_inputs(), CALL)


# ============================================================================== the G7A preflight (before any paid request)
def test_a_missing_dialogue_blocks_only_that_call_and_no_request_is_built_for_it(tmp_path):
    runner, fake = make_world(tmp_path)
    (runner.root / "authoring/dialogues" / f"{CALL}.json").unlink()
    ok, reason = runner.eligibility(CALL)
    assert not ok and "no accepted G6 dialogue" in reason
    assert runner.eligibility(OTHER)[0] is True
    assert runner._ensure_fresh_inputs(runner.inputs.call_ids) is None  # one call's dialogue never blocks the whole run
    assert CALL not in runner.inputs.dialogues and runner.inputs.dialogue_hashes[CALL] is None
    report = runner.dry_run()
    row = next(r for r in report["calls"] if r["call_id"] == CALL)
    assert row["eligible"] is False and row["request_sha256"] is None and "no accepted G6 dialogue" in row["reason"]
    assert report["estimate"]["eligible_calls"] == 49 and report["feasibility"]["checked"] == 49
    with pytest.raises(RunError, match="no accepted G6 dialogue"):
        runner.run([CALL])
    assert fake.calls == []
    chosen = runner.run(limit=1)  # the unrestricted run skips the ineligible call and sends one request for another
    assert [a["call_id"] for a in chosen["attempts"]] == ["CALL_001"] and CALL in chosen["skipped"] and len(fake.calls) == 1


def test_both_g6_files_absent_is_not_acceptance_even_though_per_call_validation_is_silent(tmp_path):
    runner, _ = make_world(tmp_path)
    (runner.root / "authoring/dialogues" / f"{CALL}.json").unlink()
    (runner.root / "authoring/dialogue_provenance" / f"{CALL}.json").unlink()
    g6 = runner._g6()
    assert g6._validate_call(CALL, g6._schema_validators()) == []  # the shared validation returns no errors for "nothing"...
    row = g6.g7_eligibility([CALL])[CALL]
    assert row["eligible"] is False and row["reason"] == "no accepted G6 dialogue"  # ...but that is not a verified dialogue


def test_an_orphan_dialogue_without_accepted_provenance_is_not_eligible(tmp_path):
    runner, _ = make_world(tmp_path)
    (runner.root / "authoring/dialogue_provenance" / f"{CALL}.json").unlink()
    ok, reason = runner.eligibility(CALL)
    assert not ok and "integrity check failed" in reason and "disagree about acceptance" in reason
    prov_only, _ = make_world(tmp_path / "b")
    (prov_only.root / "authoring/dialogues" / f"{CALL}.json").unlink()
    assert "no accepted G6 dialogue" in prov_only.eligibility(CALL)[1]


def test_a_tampered_dialogue_or_tampered_g6_provenance_is_not_eligible(tmp_path):
    runner, _ = make_world(tmp_path / "a")
    path = runner.root / "authoring/dialogues" / f"{CALL}.json"
    path.write_bytes(path.read_bytes() + b"\n")
    ok, reason = runner.eligibility(CALL)
    assert not ok and "artifact hash differs from provenance" in reason
    runner, _ = make_world(tmp_path / "b")
    rewrite(runner.root / "authoring/dialogue_provenance" / f"{CALL}.json",
            lambda d: d["accepted"]["parsed_response"]["turns"][3].update(text="Edited after acceptance."))
    ok, reason = runner.eligibility(CALL)
    assert not ok and "integrity check failed" in reason
    assert runner.eligibility(OTHER)[0] is True


def test_stale_or_invalid_g5_approval_and_changed_g3_g4_make_the_dialogue_ineligible(tmp_path):
    runner, _ = make_world(tmp_path / "a")
    rewrite(runner.root / "authoring/plan_verification_provenance" / f"{CALL}.json", lambda d: d["accepted"].update(artifact_sha256="0" * 64))
    ok, reason = runner.eligibility(CALL)
    assert not ok and "not G7-eligible per G6" in reason and "G5" in reason
    runner, _ = make_world(tmp_path / "b")
    plan = runner.root / "authoring/check_plans" / f"{CALL}.json"
    plan.write_bytes(plan.read_bytes() + b"\n")
    ok, reason = runner.eligibility(CALL)
    assert not ok and ("stale" in reason or "changed" in reason or "invalid" in reason)


def test_an_unreadable_dialogue_is_never_classified_as_deleted(tmp_path, monkeypatch):
    runner, _ = make_world(tmp_path)
    target = runner.root / "authoring/dialogues" / f"{CALL}.json"
    original = type(target).read_bytes

    def read_bytes(self):
        if self == target:
            raise PermissionError(13, "Permission denied", str(self))
        return original(self)
    monkeypatch.setattr(type(target), "read_bytes", read_bytes)
    assert read_dialogue(runner.root, CALL) == (DIALOGUE_UNREADABLE, None)  # not None: a confirmed absence is only FileNotFoundError
    ok, reason = runner.eligibility(CALL)
    assert not ok and "upstream_invalid" in reason and "Permission denied" in reason  # fail closed with the real cause
    assert "no accepted" not in reason and "absent" not in reason
    assert runner._g6().g7_eligibility([CALL])[CALL]["eligible"] is False
    assert runner.eligibility(OTHER)[0] is True


# ============================================================================== acceptance ordering (under the lock)
def during_inference(runner, fake, action):
    """Replace the fake provider's responder by one that first performs `action` (a change made while the request is in flight)."""
    inner = good_responder(runner)

    def respond(body):
        action()
        return inner(body)
    fake.responder = respond


def assert_one_send(fake):
    assert [c[0] for c in fake.calls] == ["create_response"]


def test_a_g6_dialogue_changed_during_inference_is_terminal_stale_and_keeps_the_response(tmp_path):
    runner, fake = make_world(tmp_path)
    path = runner.root / "authoring/dialogues" / f"{CALL}.json"
    during_inference(runner, fake, lambda: path.write_bytes(path.read_bytes() + b"\n"))
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "STALE_UPSTREAM" and "dialogue" in attempt["stale_upstream"]["changed"]
    assert runner.store.read_raw_response(CALL, "A001") is not None and runner.call_state(CALL)["rejected"] == 0  # no structural budget used
    assert_one_send(fake) and not (runner.root / runner.spec.artifact_path(CALL)).exists()


@pytest.mark.parametrize("what", ["dialogue", "plan_verification", "check_plan", "call_config"])
def test_a_deleted_or_changed_upstream_during_inference_is_terminal_stale(tmp_path, what):
    runner, fake = make_world(tmp_path)
    path = {"dialogue": runner.root / "authoring/dialogues" / f"{CALL}.json",
            "plan_verification": runner.root / "authoring/plan_verifications" / f"{CALL}.json",
            "check_plan": runner.root / "authoring/check_plans" / f"{CALL}.json",
            "call_config": runner.root / "authoring/call_configs" / f"{CALL}.json"}[what]
    during_inference(runner, fake, (path.unlink if what in ("dialogue", "plan_verification") else lambda: path.write_bytes(path.read_bytes() + b"\n")))
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "STALE_UPSTREAM" and what in attempt["stale_upstream"]["changed"]
    assert runner.store.read_raw_response(CALL, "A001") is not None and runner.call_state(CALL)["rejected"] == 0
    assert_one_send(fake)


def test_an_unreadable_dialogue_during_acceptance_is_a_resumable_hold(tmp_path, monkeypatch):
    runner, fake = make_world(tmp_path)
    import fyp_evaluation.dialogue_verifications.inputs as g7_inputs
    original = g7_inputs.read_dialogue
    during_inference(runner, fake, lambda: monkeypatch.setattr(
        g7_inputs, "read_dialogue", lambda root, call_id: (DIALOGUE_UNREADABLE, None) if call_id == CALL else original(root, call_id)))
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "RESPONDED" and runner.store.read_attempt(CALL, "A001")["blocked"]["kind"] == "UPSTREAM_INVALID"
    monkeypatch.undo()
    again = runner.run([CALL])
    assert again["recovered"] == [(CALL, "A001", "ACCEPTED")] and state(runner) == "ACCEPTED" and runner.validate() == []
    assert_one_send(fake)  # the stored response was adjudicated again, never resent


def test_g6_provenance_corruption_with_unchanged_dialogue_bytes_is_a_resumable_hold(tmp_path):
    runner, fake = make_world(tmp_path)
    path = runner.root / "authoring/dialogue_provenance" / f"{CALL}.json"
    original = path.read_bytes()
    during_inference(runner, fake, lambda: rewrite(path, lambda d: d["accepted"].update(artifact_sha256="0" * 64)))
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "RESPONDED" and runner.store.read_attempt(CALL, "A001")["blocked"]["kind"] == "UPSTREAM_INVALID"
    path.write_bytes(original)
    runner.run([CALL])
    assert state(runner) == "ACCEPTED" and runner.validate() == []
    assert_one_send(fake)


def test_write_lock_contention_defers_acceptance_without_losing_the_response(tmp_path, monkeypatch):
    runner, fake = make_world(tmp_path)

    class Busy:
        def __enter__(self):
            raise StageWriteLockBusy("the dataset write lock is busy")

        def __exit__(self, *exc):
            return False
    monkeypatch.setattr("fyp_evaluation.llm.stage_runner.dataset_write_lock", lambda *a, **k: Busy())
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "RESPONDED" and runner.store.read_attempt(CALL, "A001")["blocked"]["kind"] == "WRITE_LOCK_BUSY"
    assert runner.status()["calls"][CALL]["status"] == "DEFERRED_WRITE_LOCK"
    monkeypatch.undo()
    runner.run([CALL])
    assert state(runner) == "ACCEPTED"
    assert_one_send(fake)


def private_prompt_and_schema_runner(tmp_path):
    """A runner whose prompt directory and schema directory are private copies, so a test can change them."""
    prompts, schemas = tmp_path / "prompts_stage7", tmp_path / "schemas"
    shutil.copytree(SCHEMAS.parent / "prompts" / "stage7", prompts)
    shutil.copytree(SCHEMAS, schemas)
    root = tmp_path / "release"
    root.mkdir()
    holder = {}
    fake = FakeGeminiTransport(lambda body: holder["responder"](body))
    runner = DialogueVerificationRunner(root, audit_root=tmp_path / "audit", schema_dir=schemas, prompt_dir=prompts, transport=fake,
                                        clock=Clock(), inputs=shared_inputs())
    holder["responder"] = good_responder(runner)
    return runner, fake, prompts, schemas


@pytest.mark.parametrize("which, key", [("prompt", "prompt_template_sha256"), ("schema", "response_schema_sha256")])
def test_a_prompt_or_schema_changed_while_the_process_is_alive_makes_the_response_stale(tmp_path, which, key):
    runner, fake, prompts, schemas = private_prompt_and_schema_runner(tmp_path)
    path = prompts / "dialogue_verification.v1.md" if which == "prompt" else schemas / "stage7" / "dialogue_verification_response.schema.json"
    during_inference(runner, fake, lambda: path.write_bytes(path.read_bytes() + b"\n "))
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "STALE_UPSTREAM" and attempt["stale_upstream"]["changed"] == [key]
    assert "settings changed" in attempt["rejection_reason"] and runner.store.read_raw_response(CALL, "A001") is not None
    assert_one_send(fake)


def test_an_unreadable_prompt_is_a_resumable_hold_and_unchanged_files_still_accept(tmp_path):
    runner, fake, prompts, _ = private_prompt_and_schema_runner(tmp_path)
    path = prompts / "dialogue_verification.v1.md"
    original = path.read_bytes()
    during_inference(runner, fake, path.unlink)
    (attempt,) = runner.run([CALL])["attempts"]
    assert attempt["state"] == "RESPONDED" and runner.store.read_attempt(CALL, "A001")["blocked"]["kind"] == "UPSTREAM_INVALID"
    path.write_bytes(original)
    runner.run([CALL])
    assert state(runner) == "ACCEPTED" and runner.validate() == []
    assert_one_send(fake)


# ============================================================================== history against the exact G6 dialogue
def accepted_world_then_g6_reopen(tmp_path):
    runner, fake = make_world(tmp_path)
    runner.run([CALL])
    first_text = (runner.root / runner.spec.artifact_path(CALL)).read_text(encoding="utf-8").replace("\r\n", "\n")
    first_dialogue = (runner.root / "authoring/dialogues" / f"{CALL}.json").read_bytes()
    controls.apply_mutation(runner.root, "B_fee_leak", schema_dir=SCHEMAS, audit_root=tmp_path / "g6_audit")
    return runner, fake, first_text, first_dialogue


def test_a_g6_reopen_makes_the_report_stale_and_the_old_report_rebuilds_against_the_old_dialogue(tmp_path):
    runner, fake, first_text, first_dialogue = accepted_world_then_g6_reopen(tmp_path)
    assert (runner.root / "authoring/dialogues" / f"{CALL}.json").read_bytes() != first_dialogue
    later = fresh_runner(runner, transport=fake)
    status = later.status()["calls"][CALL]
    assert status["status"] == "ACCEPTED_STALE" and status["dependency"] == "stale" and status["g8_eligible"] is False
    assert "dialogue" in later.g8_eligibility([CALL])[CALL]["reason"]
    assert later.eligibility(CALL)[1].startswith("already accepted")
    assert later.validate() != []  # a stale accepted report is reported, never silently trusted
    fake.responder = good_responder(later, {CALL: leak})
    v2 = later.run([CALL], reopen_reason="re-verify the repaired dialogue")["attempts"][0]
    assert v2["state"] == "ACCEPTED" and v2["attempt_id"] == "A002"
    (history,) = prov(later)["accepted_history"]
    assert history["artifact_sha256"] and later._rebuild_text(CALL, history) == first_text  # against the OLD dialogue and versions
    assert artifact(later)["dialogue_sha256"] != parse_json_text(first_text)["dialogue_sha256"]
    assert later.validate() == [] and later.check() == {"missing": [], "changed": [], "extra": []}
    other = fresh_runner(later)
    assert other.validate() == [] and other.g8_eligibility([CALL])[CALL]["verdict"] == "INVALID"


def test_cached_inputs_that_lag_behind_a_g6_reopen_are_reloaded_before_the_request_is_built(tmp_path):
    runner, fake = make_world(tmp_path)
    stale_hash = runner.inputs.dialogue_hashes[CALL]
    controls.apply_mutation(runner.root, "B_fee_leak", schema_dir=SCHEMAS, audit_root=tmp_path / "g6_audit")  # after this runner loaded
    (attempt,) = runner.run([CALL])["attempts"]
    new_hash = read_dialogue(runner.root, CALL)[0]
    assert new_hash != stale_hash and attempt["state"] == "ACCEPTED"
    assert attempt["upstream_sha256"]["dialogue"] == new_hash == artifact(runner)["dialogue_sha256"]
    sent = parse_json_text(runner.store.read_request(CALL, "A001"))
    assert controls.FEE_SENTENCE in sent["input"]  # the request carries the CURRENT dialogue, not the cached one
    assert runner.validate() == []


def test_missing_or_corrupted_g6_history_fails_explicitly_and_never_uses_the_current_dialogue(tmp_path):
    runner, fake, first_text, _ = accepted_world_then_g6_reopen(tmp_path)
    later = fresh_runner(runner, transport=fake)
    fake.responder = good_responder(later)
    later.run([CALL], reopen_reason="re-verify")
    assert fresh_runner(runner).validate() == []
    g6_prov = runner.root / "authoring/dialogue_provenance" / f"{CALL}.json"
    original = g6_prov.read_bytes()

    rewrite(g6_prov, lambda d: d.update(accepted_history=[]))  # the dialogue the old report verified is gone from G6's history
    errs = fresh_runner(runner).validate()
    assert any("not in G6's accepted history" in e for e in errs), errs
    g6_prov.write_bytes(original)

    rewrite(g6_prov, lambda d: d["accepted_history"][0]["parsed_response"]["turns"][3].update(text="Edited after acceptance."))
    errs = fresh_runner(runner).validate()
    assert any("G6" in e for e in errs), errs  # a corrupted historical entry with an unchanged recorded hash
    g6_prov.write_bytes(original)
    assert fresh_runner(runner).validate() == []


def test_a_g6_entry_built_from_different_g3_g4_g5_versions_than_the_report_recorded_is_rejected(tmp_path):
    runner, fake, _, _ = accepted_world_then_g6_reopen(tmp_path)
    later = fresh_runner(runner, transport=fake)
    fake.responder = good_responder(later)
    later.run([CALL], reopen_reason="re-verify")
    g7_prov = runner.root / "authoring/dialogue_verification_provenance" / f"{CALL}.json"

    def other_plan(data):
        entry = data["accepted_history"][0]
        next(a for a in data["attempts"] if a["attempt_id"] == entry["attempt_id"])["upstream_sha256"]["check_plan"] = "f" * 64
    rewrite(g7_prov, other_plan)
    errs = fresh_runner(runner).validate()
    assert errs and any("G6" in e or "G3/G4" in e or "check plan" in e.lower() or "reconstructed" in e for e in errs), errs


# ============================================================================== the isolated controls
def test_controls_are_built_through_g6_preserve_every_event_and_never_touch_production(tmp_path):
    before = digest(*PRODUCTION)
    built = {name: controls.build_control(RELEASE, tmp_path / name, name, schema_dir=SCHEMAS, audit_root=tmp_path / f"g6_audit_{name}")
             for name in controls.MUTATIONS}
    original = parse_json_text((RELEASE / "authoring/dialogues" / f"{CALL}.json").read_text(encoding="utf-8"))
    for name, root in built.items():
        runner = DialogueVerificationRunner(root, audit_root=tmp_path / f"audit_{name}", schema_dir=SCHEMAS, clock=Clock(),
                                            transport=FakeGeminiTransport(), inputs=load_inputs(root, SCHEMAS))
        g6 = runner._g6()
        assert g6.validate(require_complete=True) == [] and g6.check() == {"missing": [], "changed": [], "extra": []}, name  # G6 passes
        assert runner.eligibility(CALL) == (True, "eligible") and runner.dry_run([CALL])["feasibility"]["infeasible"] == {}  # G7 preflight
        dialogue = parse_json_text((root / "authoring/dialogues" / f"{CALL}.json").read_text(encoding="utf-8"))
        assert [(a["event_id"], a["turn_id"]) for a in dialogue["event_alignment"]] == [(a["event_id"], a["turn_id"]) for a in original["event_alignment"]]
        assert len(dialogue["turns"]) == len(original["turns"]) == 68 and max(a["turn_id"] for a in dialogue["event_alignment"]) == 61
        assert [t["speaker"] for t in dialogue["turns"]] == [t["speaker"] for t in original["turns"]]  # alternation intact
        lo, hi = dialogue["length"]["accepted_word_range"]
        assert lo <= dialogue["length"]["words"] <= hi
        changed = [i for i, (a, b) in enumerate(zip(dialogue["turns"], original["turns"])) if a != b]
        assert changed == ({"A_unchanged": [], "B_fee_leak": [66], "C_buy_after_confirmation": [67]}[name])
    b = parse_json_text((built["B_fee_leak"] / "authoring/dialogues" / f"{CALL}.json").read_text(encoding="utf-8"))
    c = parse_json_text((built["C_buy_after_confirmation"] / "authoring/dialogues" / f"{CALL}.json").read_text(encoding="utf-8"))
    assert controls.FEE_SENTENCE in b["turns"][66]["text"] and b["turns"][66]["speaker"] == "EMPLOYEE"
    assert c["turns"][67]["text"].startswith(controls.BUY_SENTENCE) and c["turns"][67]["speaker"] == "CLIENT"
    assert digest(*PRODUCTION) == before


def test_a_detection_of_each_planted_defect_is_accepted_with_exact_actual_turn_evidence(tmp_path):
    for name, turn_id, quote, question, code in [
            ("B_fee_leak", 66, controls.FEE_SENTENCE, "C04", "OMISSION_LEAKAGE"),
            ("C_buy_after_confirmation", 67, "My final instruction for that same holding is to buy instead.", "T05", "EXTRACTION_STATE")]:
        root = controls.build_control(RELEASE, tmp_path / name, name, schema_dir=SCHEMAS, audit_root=tmp_path / f"g6_{name}")
        holder = {}
        runner = DialogueVerificationRunner(root, audit_root=tmp_path / f"audit_{name}", schema_dir=SCHEMAS, clock=Clock(),
                                            transport=FakeGeminiTransport(lambda body: holder["r"](body)),
                                            inputs=load_inputs(root, SCHEMAS))
        holder["r"] = good_responder(runner, {CALL: lambda ctx, q=question, c=code, t=turn_id, s=quote: mismatch_report(
            ctx, q, {"turn_id": t, "quote": s}, code=c)})
        assert runner.run([CALL])["attempts"][0]["state"] == "ACCEPTED"
        report = artifact(runner)
        assert report["verdict"] == "INVALID"
        (item,) = report["issues"][0]["evidence"]
        assert item["turn_id"] == turn_id and item["quote"] == quote and item["event_ids"] == []  # supporting speech outside any event
        start, end = item["spans"][0]
        dialogue = parse_json_text((root / "authoring/dialogues" / f"{CALL}.json").read_text(encoding="utf-8"))
        assert dialogue["turns"][turn_id]["text"][start:end] == quote
        assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
