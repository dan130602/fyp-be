"""Claude-assisted review ingestion for G5 and G7 (multilingual releases): the reviewer reads the stage's own brief and writes the
stage's own response; Python validates, derives the verdict and accepts it through the normal lifecycle. Offline; no provider is
contacted and no credential is read."""

import json
import shutil

import pytest

import multilingual_support as M
from fyp_evaluation import register_guide
from fyp_evaluation.dialogue_verifications.cli import main as g7_main
from fyp_evaluation.dialogue_verifications.run import synthesize_response as g7_valid
from fyp_evaluation.llm import exposure
from fyp_evaluation.llm.errors import AmbiguousError
from fyp_evaluation.llm.stage_runner import RunError
from fyp_evaluation.plan_verifications.cli import main as g5_main
from fyp_evaluation.plan_verifications.run import synthesize_response as g5_valid

SINGLISH, MANDARIN = M.first_call("SINGLISH"), M.first_call("MANDARIN")
REVIEWER = {"kind": "claude-code-harness", "model": "claude-sonnet-5-5", "note": "test review"}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """G3 and G4 accepted (fake) for two calls; G5 and later are ingested by the tests themselves."""
    tmp = tmp_path_factory.mktemp("review")
    chain = M.chain(tmp, [SINGLISH, MANDARIN])
    assert set(chain.run_g3().values()) == set(chain.run_g4().values()) == {"ACCEPTED"}
    return chain


def fresh(base, tmp_path):
    return M.copy_chain(base.root, tmp_path / "dataset", call_ids=base.call_ids)


def packet_file(runner, call_id, response, tmp_path, name="review.json", reopen=False, retry_rejected=False, **overrides):
    packet = runner.prepare_review(call_id, reopen=reopen, retry_rejected=retry_rejected)
    envelope = {"format": "claude-review/1", "stage": runner.spec.stage, "call_id": call_id, "reviewer": dict(REVIEWER),
                "request_sha256": packet["request_sha256"], "response": response}
    envelope.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
    return path


def accept_g5(chain, tmp_path, call_ids=(SINGLISH, MANDARIN)):
    runner = chain.open_g5()
    for call_id in call_ids:
        path = packet_file(runner, call_id, g5_valid(runner.builder.context(call_id)), tmp_path, f"g5_{call_id}.json")
        assert runner.ingest_review(call_id, path)["state"] == "ACCEPTED"
    return runner


def artifact(chain, folder, call_id):
    return json.loads((chain.root / "authoring" / folder / f"{call_id}.json").read_text(encoding="utf-8"))


def provenance(chain, folder, call_id):
    return json.loads((chain.root / "authoring" / folder / f"{call_id}.json").read_text(encoding="utf-8"))


def mismatch_g5(ctx, assessment="MISMATCH"):
    qid = next(q for q in ctx.order if ctx.checks[q]["evidence_mode"] == "PRESENCE" and ctx.checks[q]["evidence_event_ids"])
    event = ctx.checks[qid]["evidence_event_ids"][0]
    report = g5_valid(ctx)
    check = next(c for c in report["checks"] if c["question_id"] == qid)
    quote = ctx.planned_text(event)[:30]
    check.update(assessment=assessment, issue_ids=["I1"], rationale="The planned wording does not establish the expected act.",
                 evidence=[{"source": "G4", "ref": event, "quote": quote}])
    report["issues"] = [{"issue_id": "I1", "assessment": assessment, "code": "COMPLIANCE_MEANING", "question_ids": [qid], "assignment_ids": [],
                         "evidence": [{"source": "G4", "ref": event, "quote": quote}], "explanation": "The event conveys something else.",
                         "repair_scope": "G4_ONLY"}]
    return report


def mismatch_g7(ctx, assessment="MISMATCH"):
    qid = next(q for q in ctx.order if ctx.checks[q]["evidence_mode"] == "PRESENCE" and ctx.checks[q]["evidence_event_ids"])
    event = ctx.checks[qid]["evidence_event_ids"][0]
    report = g7_valid(ctx)
    check = next(c for c in report["checks"] if c["question_id"] == qid)
    evidence = [{"turn_id": ctx.turn_of_event[event], "quote": ctx.planned_text(event)[:30]}]
    check.update(assessment=assessment, issue_ids=["I1"], rationale="The dialogue does not establish the expected act.", evidence=evidence)
    report["issues"] = [{"issue_id": "I1", "assessment": assessment, "code": "COMPLIANCE_MEANING", "question_ids": [qid], "assignment_ids": [],
                         "language_row_ids": [], "evidence": evidence if assessment == "MISMATCH" else [], "explanation": "It conveys something else.",
                         "repair_scope": "G6_ONLY"}]
    return report


# ============================================================================================ valid ingestion, reconstruction, honest metadata
def test_a_valid_g5_review_is_accepted_reconstructs_and_records_the_reviewer_honestly(base, tmp_path):
    chain = fresh(base, tmp_path)
    runner = chain.open_g5()
    path = packet_file(runner, SINGLISH, g5_valid(runner.builder.context(SINGLISH)), tmp_path)
    attempt = runner.ingest_review(SINGLISH, path)
    assert attempt["state"] == "ACCEPTED" and attempt["transport"] == "manual" and attempt["possibly_billed"] is False
    assert attempt["model"] == "claude-code-harness:claude-sonnet-5-5" and attempt["estimated_cost_usd"] == 0.0
    assert attempt["params"] == {"method": "manual_review", "reviewer": "claude-code-harness", "reviewer_model": "claude-sonnet-5-5",
                                 "request_sha256": attempt["params"]["request_sha256"]}
    report = artifact(chain, "plan_verifications", SINGLISH)
    assert report["verdict"] == "VALID" and report["generation"]["model"].startswith("claude-code-harness:")
    assert report["generation"]["params"]["method"] == "manual_review" and "gemini" not in json.dumps(report["generation"]).lower()
    assert report["source"]["register_guide_sha256"] == register_guide.guide_sha256()
    prov = provenance(chain, "plan_verification_provenance", SINGLISH)
    assert prov["attempts"][0]["transport"] == "manual" and prov["accepted"]["model"].startswith("claude-code-harness:")
    assert prov["attempts"][0]["exposure"]["run_kind"] == "manual"
    assert runner.transport is not None and runner.transport().calls == [], "nothing was sent anywhere"
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}
    ledger = exposure.build(chain.root, chain.audit_root, runner.inputs.dataset_id)
    assert ledger["live_attempts"] == [] and ledger["summary"]["exposed_calls"] == 0


def test_the_stage_reports_the_reviewer_not_the_provider_in_its_status(base, tmp_path):
    chain = fresh(base, tmp_path)
    runner = accept_g5(chain, tmp_path, [MANDARIN])
    row = runner.status()["calls"][MANDARIN]
    assert row["verdict"] == "VALID" and row["g6_eligible"] is True and row["provider_sends"] == 0 and row["estimated_cost_usd"] == 0.0


# ============================================================================================ verdict derivation and downstream eligibility (G5)
def test_g5_verdicts_are_derived_by_python_and_only_valid_permits_g6(base, tmp_path):
    chain = fresh(base, tmp_path)
    runner = chain.open_g5()
    ctx = runner.builder.context(MANDARIN)
    steps = [("INVALID", mismatch_g5(ctx), False), ("UNCLEAR", mismatch_g5(ctx, "UNCLEAR"), False), ("VALID", g5_valid(ctx), True)]
    for index, (verdict, response, eligible) in enumerate(steps):
        path = packet_file(runner, MANDARIN, response, tmp_path, f"g5_{index}.json", reopen=bool(index))
        attempt = runner.ingest_review(MANDARIN, path, reopen_reason=f"step {index}" if index else None)
        assert attempt["state"] == "ACCEPTED"
        assert artifact(chain, "plan_verifications", MANDARIN)["verdict"] == verdict
        assert runner.g6_eligibility([MANDARIN])[MANDARIN]["eligible"] is eligible, verdict
    history = provenance(chain, "plan_verification_provenance", MANDARIN)["accepted_history"]
    assert len(history) == 2 and runner.validate() == [], "earlier reviews stay as superseded history"


# ============================================================================================ rejection of unsound reviews
def reject(base, tmp_path, make, expected, stage="G5", call_id=MANDARIN):
    """Ingest a deliberately unsound review into a fresh copy: it must be REJECTED for the stated reason and accept nothing."""
    chain = fresh(base, tmp_path)
    runner = chain.open_g5() if stage == "G5" else None
    if stage == "G7":
        accept_g5(chain, tmp_path, [call_id])
        assert chain.run_g6(call_ids=[call_id]) == {call_id: "ACCEPTED"}
        runner = chain.open_g7()
    response = make(runner.builder.context(call_id))
    path = packet_file(runner, call_id, response, tmp_path, "unsound.json")
    attempt = runner.ingest_review(call_id, path)
    assert attempt["state"] == "REJECTED", attempt["state"]
    assert any(expected in error for error in attempt["validation_errors"]), attempt["validation_errors"][:4]
    assert not (runner.root / runner.spec.artifact_path(call_id)).exists()
    assert runner.validate() == []
    return chain, runner


def _drop_last_check(ctx):
    report = g5_valid(ctx)
    report["checks"] = report["checks"][:-1]
    return report


def _swap_first_checks(ctx):
    report = g5_valid(ctx)
    report["checks"][0], report["checks"][1] = report["checks"][1], report["checks"][0]
    return report


def _drop_rationale(ctx):
    report = g5_valid(ctx)
    del report["checks"][0]["rationale"]
    return report


def _fabricated_g5(ctx):
    report = g5_valid(ctx)
    next(c for c in report["checks"] if c["evidence"])["evidence"][0]["quote"] = "this sentence was never planned"
    return report


def _wrong_event_g5(ctx):
    report = g5_valid(ctx)
    check = next(c for c in report["checks"] if c["evidence"])
    check["evidence"][0]["ref"] = next(e["event_id"] for e in ctx.events if e["event_id"] != check["evidence"][0]["ref"])  # a real quote under another event
    return report


def _unknown_issue(ctx):
    report = g5_valid(ctx)
    report["checks"][0]["issue_ids"] = ["I9"]
    return report


def _unlinked_mismatch(ctx):
    report = mismatch_g5(ctx)
    report["checks"] = [{**c, "issue_ids": []} if c["issue_ids"] else c for c in report["checks"]]
    return report


@pytest.mark.parametrize("make,expected", [
    (_drop_last_check, "schema:"), (_swap_first_checks, "checks must list exactly the 20"), (_drop_rationale, "schema:"),
    (_fabricated_g5, "not a substring"), (_wrong_event_g5, "not a substring"), (_unknown_issue, "unknown issue"),
    (_unlinked_mismatch, "needs at least one linked issue")])
def test_g5_reviews_that_do_not_hold_together_are_rejected_and_accept_nothing(base, tmp_path, make, expected):
    reject(base, tmp_path, make, expected)


def test_a_rejected_review_can_be_retried_only_deliberately_and_three_rejections_stop_the_call(base, tmp_path):
    chain, runner = reject(base, tmp_path, _fabricated_g5, "not a substring")
    good = packet_file(runner, MANDARIN, g5_valid(runner.builder.context(MANDARIN)), tmp_path, "good.json", retry_rejected=True)
    with pytest.raises(RunError, match="REJECTED"):
        runner.ingest_review(MANDARIN, good)  # the rejection must be looked at first
    assert runner.ingest_review(MANDARIN, good, retry_rejected=True)["state"] == "ACCEPTED"


# ============================================================================================ the envelope, the call, the reviewer
def test_a_review_for_another_call_or_stage_or_with_a_borrowed_identity_is_refused_before_any_attempt(base, tmp_path):
    chain = fresh(base, tmp_path)
    runner = chain.open_g5()
    good = g5_valid(runner.builder.context(MANDARIN))
    path = packet_file(runner, MANDARIN, good, tmp_path)
    with pytest.raises(RunError, match="is for 'CALL_"):
        runner.ingest_review(SINGLISH, path)
    with pytest.raises(RunError, match="not G7"):
        chain.open_g5()  # a G5 file offered to the G7 loader
        from fyp_evaluation.llm.gemini_runner import load_review

        load_review(path, "G7", MANDARIN)
    for reviewer in ({"kind": "gemini-flex", "model": "gemini-3.8-flash"}, {"kind": "claude-code-harness", "model": "gemini-3.8-flash"},
                     {"kind": "claude-code-harness", "model": ""}, {"kind": "claude-code-harness"}):
        bad = packet_file(runner, MANDARIN, good, tmp_path, "bad.json", reviewer=reviewer)
        with pytest.raises(RunError, match="reviewer"):
            runner.ingest_review(MANDARIN, bad)
    extra = packet_file(runner, MANDARIN, good, tmp_path, "extra.json", extra_key=True)
    with pytest.raises(RunError, match="exactly the keys"):
        runner.ingest_review(MANDARIN, extra)
    with pytest.raises(RunError, match="cannot read"):
        runner.ingest_review(MANDARIN, tmp_path / "missing.json")
    assert runner.store.attempt_ids(MANDARIN) == [], "no refused review left an attempt behind"


# ============================================================================================ changed inputs
def test_a_review_written_against_a_different_brief_is_refused(base, tmp_path, monkeypatch):
    chain = fresh(base, tmp_path)
    runner = chain.open_g5()
    good = g5_valid(runner.builder.context(SINGLISH))
    path = packet_file(runner, SINGLISH, good, tmp_path)
    tampered = packet_file(runner, SINGLISH, good, tmp_path, "tampered.json", request_sha256="0" * 64)
    with pytest.raises(RunError, match="different brief"):
        runner.ingest_review(SINGLISH, tampered)
    directory = tmp_path / "guides"  # the register guide changes between preparation and acceptance
    directory.mkdir()
    for guide in register_guide.GUIDE_DIR.glob("register_singlish.*.md"):
        shutil.copyfile(guide, directory / guide.name)
    monkeypatch.setattr(register_guide, "GUIDE_DIR", directory)
    (directory / "register_singlish.v1.md").write_text(register_guide.guide_text() + "\nChanged after preparation.\n", encoding="utf-8")
    with pytest.raises(RunError, match="different brief"):
        runner.ingest_review(SINGLISH, path)
    assert runner.store.attempt_ids(SINGLISH) == [] and not (chain.root / "authoring/plan_verifications" / f"{SINGLISH}.json").exists()


def test_a_changed_upstream_plan_between_preparation_and_acceptance_is_refused(base, tmp_path):
    chain = fresh(base, tmp_path)
    runner = chain.open_g5()
    path = packet_file(runner, MANDARIN, g5_valid(runner.builder.context(MANDARIN)), tmp_path)
    plan_path = chain.root / "authoring/check_plans" / f"{MANDARIN}.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["planned_events"][0]["authoring"]["planned_text"] += "!"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017 - refused as stale, unreadable or a different brief, whichever the gate reports first
        runner.ingest_review(MANDARIN, path)
    assert not (chain.root / "authoring/plan_verifications" / f"{MANDARIN}.json").exists()


# ============================================================================================ re-import and reopening
def test_importing_the_same_review_twice_does_not_accept_twice_and_reopening_preserves_history(base, tmp_path):
    chain = fresh(base, tmp_path)
    runner = chain.open_g5()
    path = packet_file(runner, MANDARIN, g5_valid(runner.builder.context(MANDARIN)), tmp_path)
    assert runner.ingest_review(MANDARIN, path)["attempt_id"] == "A001"
    before = (chain.root / "authoring/plan_verifications" / f"{MANDARIN}.json").read_bytes()
    with pytest.raises(RunError, match="already accepted"):
        runner.ingest_review(MANDARIN, path)
    assert runner.store.attempt_ids(MANDARIN) == ["A001"] and (chain.root / "authoring/plan_verifications" / f"{MANDARIN}.json").read_bytes() == before
    second = runner.ingest_review(MANDARIN, path, reopen_reason="re-reviewed")
    assert second["attempt_id"] == "A002" and second["state"] == "ACCEPTED"
    prov = provenance(chain, "plan_verification_provenance", MANDARIN)
    assert [h["attempt_id"] for h in prov["accepted_history"]] == ["A001"] and prov["accepted"]["attempt_id"] == "A002"
    assert runner.validate() == []


# ============================================================================================ G7: coverage, verdicts, eligibility, language rows
@pytest.fixture(scope="module")
def reviewed(base, tmp_path_factory):
    """G5 accepted by review for both calls and G6 accepted (fake Batch) for the Mandarin call."""
    tmp = tmp_path_factory.mktemp("g7")
    chain = fresh(base, tmp)
    accept_g5(chain, tmp)
    assert chain.run_g6(call_ids=[MANDARIN]) == {MANDARIN: "ACCEPTED"}
    return chain


def test_g7_verdicts_are_derived_and_only_valid_permits_g8(reviewed, tmp_path):
    chain = fresh(reviewed, tmp_path)
    runner = chain.open_g7()
    ctx = runner.builder.context(MANDARIN)
    steps = [("INVALID", mismatch_g7(ctx), False), ("UNCLEAR", mismatch_g7(ctx, "UNCLEAR"), False), ("VALID", g7_valid(ctx), True)]
    for index, (verdict, response, eligible) in enumerate(steps):
        path = packet_file(runner, MANDARIN, response, tmp_path, f"g7_{index}.json", reopen=bool(index))
        attempt = runner.ingest_review(MANDARIN, path, reopen_reason=f"step {index}" if index else None)
        assert attempt["state"] == "ACCEPTED"
        report = artifact(chain, "dialogue_verifications", MANDARIN)
        assert report["verdict"] == verdict and report["generation"]["model"].startswith("claude-code-harness:")
        row = runner.g8_eligibility([MANDARIN])[MANDARIN]
        assert row["eligible"] is eligible, (verdict, row)
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


def test_a_serious_naturalness_finding_is_advisory_and_never_blocks(reviewed, tmp_path):
    chain = fresh(reviewed, tmp_path)
    runner = chain.open_g7()
    ctx = runner.builder.context(MANDARIN)
    response = g7_valid(ctx)
    turn = next(t for t in ctx.turns if t["speaker"] == "CLIENT")
    response["quality_findings"] = [{"quality_id": "Q1", "severity": "SERIOUS", "code": "LANGUAGE_NATURALNESS",
                                     "evidence": [{"turn_id": turn["turn_id"], "quote": turn["text"][:12]}], "explanation": "Stilted."}]
    assert runner.ingest_review(MANDARIN, packet_file(runner, MANDARIN, response, tmp_path))["state"] == "ACCEPTED"
    report = artifact(chain, "dialogue_verifications", MANDARIN)
    assert report["verdict"] == "VALID" and report["quality_review"] == "REVIEW_REQUIRED"
    assert runner.g8_eligibility([MANDARIN])[MANDARIN]["eligible"] is True


def reject_g7(reviewed, tmp_path, make, expected):
    chain = fresh(reviewed, tmp_path)
    runner = chain.open_g7()
    path = packet_file(runner, MANDARIN, make(runner.builder.context(MANDARIN)), tmp_path, "unsound.json")
    attempt = runner.ingest_review(MANDARIN, path)
    assert attempt["state"] == "REJECTED" and any(expected in error for error in attempt["validation_errors"]), attempt["validation_errors"][:4]
    assert not (chain.root / "authoring/dialogue_verifications" / f"{MANDARIN}.json").exists() and runner.validate() == []


def _drop_edge_row(ctx):
    report = g7_valid(ctx)
    report["edge_cases"] = report["edge_cases"][:-1]
    return report


def _quote_in_wrong_turn(ctx):
    report = g7_valid(ctx)
    check = next(c for c in report["checks"] if c["evidence"])
    check["evidence"][0]["turn_id"] = next(t["turn_id"] for t in ctx.turns if check["evidence"][0]["quote"] not in t["text"])  # real speech, wrong turn
    return report


def _fabricated_g7(ctx):
    report = g7_valid(ctx)
    next(c for c in report["checks"] if c["evidence"])["evidence"][0]["quote"] = "the client said something else entirely"
    return report


def _issue_not_linked_back(ctx):
    report = mismatch_g7(ctx)
    next(c for c in report["checks"] if c["issue_ids"])["issue_ids"] = []
    return report


def _match_citing_an_issue(ctx):
    report = g7_valid(ctx)
    report["checks"][0]["issue_ids"] = ["I1"]
    return report


@pytest.mark.parametrize("make,expected", [
    (_drop_edge_row, "edge_cases must list exactly"), (_quote_in_wrong_turn, "not a substring"), (_fabricated_g7, "not a substring"),
    (_issue_not_linked_back, "needs at least one linked issue"), (_match_citing_an_issue, "unknown issue")])
def test_g7_reviews_with_missing_rows_or_misplaced_quotes_are_rejected(reviewed, tmp_path, make, expected):
    reject_g7(reviewed, tmp_path, make, expected)


def test_a_missing_dialogue_or_an_unverified_g5_blocks_the_review(reviewed, tmp_path):
    chain = fresh(reviewed, tmp_path)
    runner = chain.open_g7()
    path = packet_file(runner, MANDARIN, g7_valid(runner.builder.context(MANDARIN)), tmp_path)
    (chain.root / "authoring/dialogues" / f"{MANDARIN}.json").unlink()
    with pytest.raises(RunError):
        runner.ingest_review(MANDARIN, path)
    assert runner.store.attempt_ids(MANDARIN) == []
    with pytest.raises(RunError):  # the Singlish call has no G6 dialogue at all
        runner.prepare_review(SINGLISH)


# ============================================================================================ the provider path is untouched
def test_an_ambiguous_provider_attempt_blocks_the_provider_but_not_a_review(reviewed, tmp_path):
    chain = fresh(reviewed, tmp_path)
    runner = chain.open_g7(errors=[AmbiguousError("deadline expired")])
    first = runner.run(call_ids=[MANDARIN])
    assert first["attempts"][0]["state"] == "TRANSPORT_FAILED" and first["attempts"][0]["ambiguous"] is True
    with pytest.raises(RunError, match="AWAITING_ACKNOWLEDGEMENT"):
        runner.run(call_ids=[MANDARIN])  # the provider path is still gated
    path = packet_file(runner, MANDARIN, g7_valid(runner.builder.context(MANDARIN)), tmp_path)
    attempt = runner.ingest_review(MANDARIN, path)
    assert attempt["state"] == "ACCEPTED" and attempt["attempt_id"] == "A002"
    prov = provenance(chain, "dialogue_verification_provenance", MANDARIN)
    assert [a["transport"] for a in prov["attempts"]] == ["sync", "manual"] and prov["attempts"][0]["possibly_billed"] is True
    assert runner.validate() == []


def test_a_provider_verification_in_the_same_release_still_records_its_provider_parameters(reviewed, tmp_path):
    chain = fresh(reviewed, tmp_path)
    runner = chain.open_g7()
    attempt = runner.run(call_ids=[MANDARIN])["attempts"][0]
    assert attempt["state"] == "ACCEPTED" and attempt["transport"] == "sync" and attempt["params"]["service_tier"] == "flex"
    assert attempt["model"] == "gemini-3.8-flash"


def test_the_historical_release_refuses_review_ingestion(tmp_path):
    import g5_support

    runner, _ = g5_support.make_runner(tmp_path)
    with pytest.raises(RunError, match="only available for multilingual"):
        runner.prepare_review("CALL_001")
    with pytest.raises(RunError, match="only available for multilingual"):
        runner.ingest_review("CALL_001", tmp_path / "x.json")


# ============================================================================================ the command lines
def test_the_commands_prepare_and_ingest_a_review_for_g5_and_g7(reviewed, tmp_path, capsys):
    chain = fresh(reviewed, tmp_path)
    base_args = ["--dataset-root", str(chain.root), "--audit-dir", str(chain.audit_root), "--schema-dir", str(M.SCHEMAS)]
    packet = tmp_path / "packet.json"
    assert g7_main([*base_args, "--prepare-review", "--call-id", MANDARIN, "--output", str(packet)]) == 0
    prepared = json.loads(packet.read_text(encoding="utf-8"))
    assert prepared["stage"] == "G7" and prepared["request"]["system_instruction"] and prepared["upstream_sha256"]["dialogue"]
    ctx = chain.open_g7().builder.context(MANDARIN)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"format": "claude-review/1", "stage": "G7", "call_id": MANDARIN, "reviewer": REVIEWER,
                                  "request_sha256": prepared["request_sha256"], "response": g7_valid(ctx)}, ensure_ascii=False), encoding="utf-8")
    capsys.readouterr()
    assert g7_main([*base_args, "--ingest-review", str(review), "--call-id", MANDARIN]) == 0 and "ACCEPTED" in capsys.readouterr().out
    assert g7_main([*base_args, "--ingest-review", str(review), "--call-id", MANDARIN]) == 2  # already accepted: refused, exit 2
    for argv in (["--prepare-review", "--call-id", MANDARIN, "--call-id", SINGLISH], ["--prepare-review"], ["--run", "--output", "x"]):
        with pytest.raises(SystemExit):
            g5_main([*base_args, *argv])
    assert g5_main([*base_args, "--prepare-review", "--call-id", SINGLISH]) == 2  # its G5 review is already accepted
    assert g5_main([*base_args, "--prepare-review", "--call-id", SINGLISH, "--reopen", "--reason", "again", "--output", str(tmp_path / "g5.json")]) == 0
