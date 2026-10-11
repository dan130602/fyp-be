"""G6 eligibility integrity: a call is eligible only for a CURRENT, INTACT, accepted G5 VALID report. Offline; tmp outputs only.

Before the fix `g6_eligibility()` trusted `dependency_status()` (which only reads the artifact file) and the artifact's own
`verdict` field, so an orphan or tampered report could be eligible. These tests pin the fail-closed behaviour."""

import json

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.llm.errors import AmbiguousError, CapacityError
from fyp_evaluation.plan_verifications.cli import main
from g5_support import SCHEMAS, Clock, make_runner, mismatch_report, runner_factory, snapshot

CALL = "CALL_021"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def accepted(tmp_path, overrides=None):
    runner, fake = make_runner(tmp_path, overrides=overrides)
    assert runner.run([CALL])["attempts"][0]["state"] == "ACCEPTED"
    return runner, fake


def artifact_path(runner):
    return runner.root / runner.spec.artifact_path(CALL)


def provenance_path(runner):
    return runner.root / runner.spec.provenance_path(CALL)


def rewrite(path, mutate):
    data = parse_json_text(path.read_bytes().decode("utf-8"))
    mutate(data)
    path.write_bytes(serialize_json(data).encode("utf-8"))


def verdict_of(runner):
    return runner.g6_eligibility([CALL])[CALL]


def leak(ctx):
    return mismatch_report(ctx, "C05", "EV_001", ctx.planned_text("EV_001")[:30])


def unclear(ctx):
    return mismatch_report(ctx, "T05", "EV_015", ctx.planned_text("EV_015")[:20], code="PLAN_COHERENCE", scope="REVIEW",
                           assessment="UNCLEAR")


# ---- 4. the one eligible case ------------------------------------------------------------------------------------
def test_current_intact_valid_report_is_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    assert verdict_of(runner) == {"eligible": True, "reason": "eligible", "verdict": "VALID"}
    row = runner.status()["calls"][CALL]
    assert row["g6_eligible"] is True and row["verdict"] == "VALID"


# ---- 1. orphan VALID report (no accepted provenance) -------------------------------------------------------------
def test_orphan_valid_report_without_provenance_file_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    provenance_path(runner).unlink()
    assert runner.call_state(CALL)["inconsistent"]  # the shared lifecycle already calls this INCONSISTENT
    result = verdict_of(runner)
    assert result["eligible"] is False and result["verdict"] is None
    assert result["reason"].startswith("integrity check failed") and "disagree" in result["reason"]


def test_orphan_valid_report_with_accepted_cleared_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    rewrite(provenance_path(runner), lambda d: d.update(accepted=None))
    assert runner.call_state(CALL)["inconsistent"]
    result = verdict_of(runner)
    assert result["eligible"] is False and result["reason"].startswith("integrity check failed")


# ---- 2. invalid or inconsistent accepted provenance --------------------------------------------------------------
def test_schema_invalid_provenance_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    rewrite(provenance_path(runner), lambda d: d.pop("attempts"))
    result = verdict_of(runner)
    assert result["eligible"] is False and result["verdict"] is None and "provenance" in result["reason"]


def test_unreadable_provenance_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    provenance_path(runner).write_bytes(b"{not json")
    assert verdict_of(runner)["eligible"] is False


def test_accepted_hash_that_disagrees_with_the_artifact_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    rewrite(provenance_path(runner), lambda d: d["accepted"].update(artifact_sha256="0" * 64))
    result = verdict_of(runner)
    assert result["eligible"] is False and result["reason"].startswith("integrity check failed")


def test_accepted_response_that_no_longer_matches_its_hash_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    rewrite(provenance_path(runner), lambda d: d["accepted"]["parsed_response"]["checks"][0].update(rationale="Edited after acceptance."))
    result = verdict_of(runner)
    assert result["eligible"] is False and result["reason"].startswith("integrity check failed")


# ---- 3. tampered G5 artifact -------------------------------------------------------------------------------------
def test_artifact_edited_from_invalid_to_valid_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path, overrides={CALL: leak})
    assert verdict_of(runner)["reason"] == "verdict INVALID"
    rewrite(artifact_path(runner), lambda d: d.update(verdict="VALID", recommended_repair=None))
    result = verdict_of(runner)
    assert result["eligible"] is False and result["verdict"] is None
    assert "artifact hash differs from provenance" in result["reason"]
    assert runner.status()["calls"][CALL]["g6_eligible"] is False


def test_valid_artifact_with_any_edit_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    rewrite(artifact_path(runner), lambda d: d["checks"][0].update(rationale="Changed after acceptance."))
    assert verdict_of(runner)["eligible"] is False


def test_artifact_replaced_by_unparseable_bytes_is_not_eligible(tmp_path):
    runner, _ = accepted(tmp_path)
    artifact_path(runner).write_bytes(b"not json")
    result = verdict_of(runner)
    assert result["eligible"] is False and result["verdict"] is None


# ---- 5. completed INVALID / UNCLEAR outcomes ---------------------------------------------------------------------
@pytest.mark.parametrize("override, verdict", [(leak, "INVALID"), (unclear, "UNCLEAR")])
def test_current_accepted_invalid_or_unclear_is_not_eligible(tmp_path, override, verdict):
    runner, _ = accepted(tmp_path, overrides={CALL: override})
    assert verdict_of(runner) == {"eligible": False, "reason": f"verdict {verdict}", "verdict": verdict}


# ---- 6. previously VALID, now stale ------------------------------------------------------------------------------
def test_previously_valid_report_with_stale_dependencies_is_not_eligible(tmp_path, monkeypatch):
    runner, _ = accepted(tmp_path)
    assert verdict_of(runner)["eligible"] is True
    original = runner._current_upstream_hashes

    def changed_plan(call_id):
        return {**original(call_id), "check_plan": "f" * 64}
    monkeypatch.setattr(runner, "_current_upstream_hashes", changed_plan)
    assert runner.dependency_status([CALL])[CALL]["status"] == "stale"
    result = verdict_of(runner)
    assert result["eligible"] is False and result["reason"] == "G5 report is stale (check_plan changed)"


def test_missing_report_is_not_eligible(tmp_path):
    runner, _ = make_runner(tmp_path)
    assert verdict_of(runner) == {"eligible": False, "reason": "no accepted G5 report", "verdict": None}


# ---- the check is read-only and does not touch completed results -------------------------------------------------
def test_eligibility_never_writes_and_leaves_accepted_results_unchanged(tmp_path):
    runner, fake = accepted(tmp_path)
    before = snapshot(runner.root, tmp_path / "audit")
    sends = len(fake.calls)
    for _ in range(2):
        assert verdict_of(runner)["eligible"] is True
        runner.status()
    assert snapshot(runner.root, tmp_path / "audit") == before and len(fake.calls) == sends
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


# ---- observational usage and cost reporting (no cap, no state) ---------------------------------------------------
def test_status_reports_sends_capacity_retries_tokens_cost_and_outstanding_calls(tmp_path):
    runner, fake = make_runner(tmp_path, errors=[CapacityError("HTTP 503"), CapacityError("HTTP 429")])
    runner.run([CALL])
    usage = runner.status()["usage"]
    assert usage["attempts"] == 1 and usage["provider_sends"] == len(fake.calls) == 3 and usage["capacity_retries"] == 2
    assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0 and usage["estimated_cost_usd"] > 0
    assert usage["completed_calls"] == 1 and usage["outstanding_calls"] == 49 and usage["unpriced_possibly_billed"] == 0
    row = runner.status()["calls"][CALL]
    assert row["provider_sends"] == 3 and row["estimated_cost_usd"] == usage["estimated_cost_usd"]
    assert runner.status()["calls"]["CALL_041"]["provider_sends"] == 0


def test_usage_counts_an_ambiguous_attempt_as_unpriced_but_possibly_billed(tmp_path):
    runner, _ = make_runner(tmp_path, errors=[AmbiguousError("APITimeoutError: timed out")])
    runner.run([CALL])
    usage = runner.status()["usage"]
    assert usage["provider_sends"] == 1 and usage["estimated_cost_usd"] == 0 and usage["unpriced_possibly_billed"] == 1
    assert usage["completed_calls"] == 0 and usage["outstanding_calls"] == 50


def test_cli_status_prints_usage_without_sending_anything(tmp_path, capsys):
    runner, fake = make_runner(tmp_path)
    runner.run([CALL])
    sends = len(fake.calls)
    argv = ["--dataset-root", str(runner.root), "--audit-dir", str(tmp_path / "audit"), "--schema-dir", str(SCHEMAS), "--status"]
    assert main(argv, clock=Clock(), runner_class=runner_factory()) == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith("usage"))
    assert json.loads(line.split(": ", 1)[1])["completed_calls"] == 1 and "not a provider balance" in line
    assert f"{CALL} ACCEPTED" in out and "sends=1" in out and len(fake.calls) == sends
