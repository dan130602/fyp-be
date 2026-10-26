"""Phase 3B: G7 v2 on the multilingual release: language rows (blocking), advisory language naturalness, the register guide as an
upstream input, and negative fixtures for the dialogue-level checks (fake Gemini transport only; nothing is sent anywhere)."""

import json
import shutil

import pytest
from jsonschema import Draft202012Validator

import g7_support
import multilingual_support as M
from fyp_evaluation import register_guide
from fyp_evaluation.dialogue_verifications.compile import derive_verdict
from fyp_evaluation.dialogue_verifications.language_rows import (
    MIXED_BOTH, MIXED_SPREAD, MIXED_SWITCH, SINGLISH_CLIENT, SINGLISH_EMPLOYEE, rows_for,
)
from fyp_evaluation.dialogue_verifications.request import RequestBuilder
from fyp_evaluation.dialogue_verifications.run import synthesize_response
from fyp_evaluation.dialogue_verifications.validate_response import expand, expand_all

CALLS = {profile: M.first_call(profile) for profile in M.PROFILES}


@pytest.fixture(scope="module")
def world(verified250_root, tmp_path_factory):
    """A writable copy of the accepted G3-G7 state with a G7 runner over it."""
    root, _ = verified250_root
    chain = M.copy_chain(root, tmp_path_factory.mktemp("g7world") / "dataset")
    chain.open_g7()
    return chain


def report_of(chain, call_id):
    return json.loads((chain.root / "authoring/dialogue_verifications" / f"{call_id}.json").read_text(encoding="utf-8"))


def ctx_of(chain, call_id):
    return chain.g7.builder.context(call_id)


# ============================================================================================ acceptance and artifacts
def test_all_250_reports_are_valid_v2_artifacts_with_the_rows_their_profile_requires(world):
    schema = world.g7.builder.artifact_schema
    specs = world.g7.inputs.specs
    for call_id in world.call_ids:
        report = report_of(world, call_id)
        assert report["schema_version"] == "2.0" and report["verdict"] == "VALID" and not list(Draft202012Validator(schema).iter_errors(report)), call_id
        rows = [r["row_id"] for r in report["language_checks"]]
        assert rows == [r["row_id"] for r in rows_for(specs[call_id])], call_id
        profile = specs[call_id]["metadata"]["language_profile"]
        assert {"EN_ZH_MIXED": [MIXED_BOTH, MIXED_SWITCH, MIXED_SPREAD], "SINGLISH": [SINGLISH_CLIENT, SINGLISH_EMPLOYEE]}.get(profile, []) == [
            r for r in rows if not r.startswith("VARIANT_")], call_id
        assert report["summary"]["language_checks"]["match"] == len(rows)
        assert ("register_guide_sha256" in report["source"]) is (profile == "SINGLISH")


def test_the_stage_validates_checks_and_the_record_is_v2(world):
    assert world.g7.validate() == [] and world.g7.check() == {"missing": [], "changed": [], "extra": []}
    record = json.loads((world.root / "authoring/dialogue_verification_record.json").read_text(encoding="utf-8"))
    schema = json.loads((M.SCHEMAS / "stage7/dialogue_verification_record.v2.schema.json").read_text(encoding="utf-8"))
    assert not list(Draft202012Validator(schema).iter_errors(record)) and record["summary"]["verdicts"]["VALID"] == 250
    assert all(row["eligible"] for row in world.g7.g8_eligibility().values()) if hasattr(world.g7, "g8_eligibility") else True


@pytest.mark.parametrize("profile", M.PROFILES)
def test_the_g7_brief_lists_the_language_rows_and_carries_the_guide_only_for_singlish(world, profile):
    builder = world.g7.builder
    body = builder.build(CALLS[profile], "A001")
    brief = json.loads(body["input"])
    assert brief["language_rows"] == rows_for(world.g7.inputs.specs[CALLS[profile]])
    assert brief["call"]["language"]["language_profile"] == profile and ("register_guide" in brief) is (profile == "SINGLISH")
    assert "THE LANGUAGE OF THE CALL AND THE LANGUAGE ROWS" in body["system_instruction"] and "LANGUAGE_REQUIREMENT_NOT_MET" in json.dumps(body["response_format"])
    assert all(turn["text"] for turn in brief["dialogue"])


def test_the_pilot_g7_request_is_unchanged_in_shape():
    builder = RequestBuilder(g7_support.shared_inputs(), M.SCHEMAS)
    body = builder.build("CALL_001", "A001")
    assert "language_rows" not in json.loads(body["input"]) and "THE LANGUAGE OF THE CALL" not in body["system_instruction"]
    assert "LANGUAGE_REQUIREMENT_NOT_MET" not in json.dumps(body["response_format"])


# ============================================================================================ language rows are blocking
def mixed_call(world):
    return next(c for c in world.call_ids if world.g7.inputs.specs[c]["metadata"]["language_profile"] == "EN_ZH_MIXED")


def singlish_call(world):
    return next(c for c in world.call_ids if world.g7.inputs.specs[c]["metadata"]["language_profile"] == "SINGLISH")


def mismatch(report, row_id, evidence, *, code="LANGUAGE_REQUIREMENT_NOT_MET", assessment="MISMATCH", scope="G6_ONLY"):
    row = next(r for r in report["language_checks"] if r["row_id"] == row_id)
    row.update(assessment=assessment, issue_ids=["I1"], rationale="The requirement is not met.", evidence=evidence)
    report["issues"] = [{"issue_id": "I1", "assessment": assessment, "code": code, "question_ids": [], "assignment_ids": [],
                         "language_row_ids": [row_id], "evidence": evidence, "explanation": "The requirement is not met.", "repair_scope": scope}]
    return report


def with_language_ids(report):
    for issue in report["issues"]:
        issue.setdefault("language_row_ids", [])
    return report


def synth(world, call_id):
    ctx = ctx_of(world, call_id)
    return ctx, with_language_ids(synthesize_response(ctx))


def test_a_failed_language_row_makes_the_verdict_invalid_and_blocks_g8(world):
    call_id = singlish_call(world)
    ctx, report = synth(world, call_id)
    employee = next(t for t in ctx.turns if t["speaker"] == "EMPLOYEE")
    evidence = [{"turn_id": employee["turn_id"], "quote": employee["text"][:30]}]
    report = mismatch(report, SINGLISH_EMPLOYEE, evidence)
    assert world.g7.validator.errors(report, ctx) == []
    checks, edges, issues, quality, language, errors = expand_all(report, ctx)
    assert not errors and derive_verdict(checks, edges, issues, language) == "INVALID"
    assert derive_verdict(checks, edges, [], [r for r in language if r["assessment"] == "MATCH"]) == "VALID"


def test_an_unclear_language_row_yields_unclear(world):
    call_id = mixed_call(world)
    ctx, report = synth(world, call_id)
    report = mismatch(report, MIXED_SPREAD, [], assessment="UNCLEAR")
    assert world.g7.validator.errors(report, ctx) == []
    checks, edges, issues, quality, language, _ = expand_all(report, ctx)
    assert derive_verdict(checks, edges, issues, language) == "UNCLEAR"


def test_language_rows_must_match_the_briefs_rows_exactly(world):
    call_id = singlish_call(world)
    ctx, report = synth(world, call_id)
    report["language_checks"] = report["language_checks"][:1]
    assert any("language_checks must list exactly" in e for e in world.g7.validator.errors(report, ctx))
    ctx, report = synth(world, call_id)
    report["language_checks"].reverse()
    assert any("language_checks must list exactly" in e for e in world.g7.validator.errors(report, ctx))
    ctx, report = synth(world, M.first_call("ENGLISH") if False else next(c for c in world.call_ids if not rows_for(world.g7.inputs.specs[c])))
    report["language_checks"] = [{"row_id": "X", "evidence": [], "rationale": "x", "assessment": "MATCH", "issue_ids": []}]
    assert any("language_checks must list exactly" in e for e in world.g7.validator.errors(report, ctx))


def test_the_mechanical_citation_rules_for_a_language_match(world):
    call_id = mixed_call(world)
    ctx, report = synth(world, call_id)
    client = [t for t in ctx.turns if t["speaker"] == "CLIENT"]
    han_only = next(t for t in client if "好" in t["text"])

    def with_evidence(row_id, evidence):
        r = json.loads(json.dumps(report))
        next(x for x in r["language_checks"] if x["row_id"] == row_id)["evidence"] = evidence
        return world.g7.validator.errors(r, ctx)

    assert with_evidence(MIXED_BOTH, []) and "must quote client speech in Mandarin" in with_evidence(MIXED_BOTH, [])[0]
    assert any("both English and Mandarin" in e for e in with_evidence(MIXED_SWITCH, [{"turn_id": han_only["turn_id"], "quote": "好的"}]))
    assert any("far apart" in e for e in with_evidence(MIXED_SPREAD, [{"turn_id": client[0]["turn_id"], "quote": client[0]["text"][:6]}]))
    sing_ctx, sing = synth(world, singlish_call(world))
    sing["language_checks"][1]["evidence"] = []
    assert any("must quote employee speech" in e for e in world.g7.validator.errors(sing, sing_ctx))
    sing["language_checks"][1]["evidence"] = [{"turn_id": next(t for t in sing_ctx.turns if t["speaker"] == "CLIENT")["turn_id"], "quote": "Okay"}]
    assert any("must quote employee speech" in e for e in world.g7.validator.errors(sing, sing_ctx))


def test_issue_linkage_rules_for_language_rows(world):
    call_id = singlish_call(world)
    ctx, report = synth(world, call_id)
    # a language issue must name a row; a failed professional-English row must cite employee speech; unknown rows are rejected
    report = mismatch(report, SINGLISH_CLIENT, [])
    report["issues"][0]["language_row_ids"] = []
    assert any("must name the language row" in e for e in world.g7.validator.errors(report, ctx))
    ctx, report = synth(world, call_id)
    report = mismatch(report, SINGLISH_EMPLOYEE, [])
    assert any("must cite the employee speech" in e for e in world.g7.validator.errors(report, ctx))
    ctx, report = synth(world, call_id)
    report = mismatch(report, SINGLISH_CLIENT, [])
    report["issues"][0]["language_row_ids"] = ["NOPE"]
    assert any("not language rows" in e for e in world.g7.validator.errors(report, ctx))
    ctx, report = synth(world, call_id)
    report = mismatch(report, SINGLISH_CLIENT, [])  # an absent requirement may cite nothing
    assert world.g7.validator.errors(report, ctx) == []
    report["language_checks"][0]["issue_ids"] = []
    assert any("needs at least one linked issue" in e for e in world.g7.validator.errors(report, ctx))


def test_language_naturalness_is_advisory_and_never_changes_the_verdict(world):
    call_id = singlish_call(world)
    ctx, report = synth(world, call_id)
    client = next(t for t in ctx.turns if t["speaker"] == "CLIENT")
    report["quality_findings"] = [{"quality_id": "Q1", "severity": "SERIOUS", "code": "LANGUAGE_NATURALNESS",
                                   "evidence": [{"turn_id": client["turn_id"], "quote": client["text"][:20]}], "explanation": "Caricatured particles."}]
    assert world.g7.validator.errors(report, ctx) == []
    checks, edges, issues, quality, language, _ = expand_all(report, ctx)
    from fyp_evaluation.dialogue_verifications.compile import derive_quality_review
    assert derive_verdict(checks, edges, issues, language) == "VALID" and derive_quality_review(quality) == "REVIEW_REQUIRED"
    v1 = json.loads((M.SCHEMAS / "stage7/dialogue_verification_response.schema.json").read_text(encoding="utf-8"))
    assert list(Draft202012Validator(v1).iter_errors(report)), "the historical response schema has neither the code nor language_checks"


# ============================================================================================ negative fixtures at the dialogue level
def test_a_dialogue_whose_events_break_a_chronology_constraint_is_rejected_by_g6(world):
    """Chronology: events placed against an ORDER constraint never reach G7."""
    from fyp_evaluation.dialogues.feasibility import synthesize_dialogue

    runner = world.open_g6()
    for call_id in world.call_ids:
        ctx = runner.builder.context(call_id)
        order = next((c for c in ctx.constraints if c["type"] == "ORDER" and ctx.speakers[c["before"]] != ctx.speakers[c["after"]]), None)
        if order:
            break
    good = synthesize_dialogue(ctx)
    assert runner.validator.errors(good, ctx) == []
    where = lambda event: next(i for i, t in enumerate(good["turns"]) if f"[[{event}]]" in t["text"])  # noqa: E731
    i, j = where(order["before"]), where(order["after"])
    swapped = json.loads(json.dumps(good))
    swapped["turns"][i], swapped["turns"][j] = good["turns"][j], good["turns"][i]  # whole turns: alternation is preserved
    errors = runner.validator.errors(swapped, ctx)
    assert any(e.startswith("ORDER:") for e in errors), errors


def test_filler_that_adds_a_confirmation_fee_or_market_comment_reaches_g7_and_an_invalid_report_validates(world):
    """Semantics: no mechanical rule decides meaning, so the verifier must SEE the filler and its report must be expressible."""
    call_id = singlish_call(world)
    ctx = ctx_of(world, call_id)
    filler = next(t for t in ctx.turns if t["speaker"] == "EMPLOYEE" and not ctx.spans_by_turn.get(t["turn_id"]))
    brief = json.loads(world.g7.builder.brief_text(call_id))
    assert any(t["turn_id"] == filler["turn_id"] and t["text"] == filler["text"] for t in brief["dialogue"])
    report = synthesize_response(ctx)
    report = with_language_ids(report)
    check = next(c for c in report["checks"] if c["question_id"] == "C12")
    check.update(assessment="MISMATCH", issue_ids=["I1"], rationale="The filler comments on market performance and promises results.",
                 evidence=[{"turn_id": filler["turn_id"], "quote": filler["text"][:20]}])
    report["issues"] = [{"issue_id": "I1", "assessment": "MISMATCH", "code": "OMISSION_LEAKAGE", "question_ids": ["C12"], "assignment_ids": [],
                         "language_row_ids": [], "evidence": check["evidence"], "explanation": "Filler leaks a prohibited act.", "repair_scope": "G6_ONLY"}]
    checks, edges, issues, quality, language, errors = expand_all(report, ctx)
    assert not errors and derive_verdict(checks, edges, issues, language) == "INVALID"


def test_a_single_language_mixed_dialogue_fails_the_mechanical_rule_for_the_rows_it_cannot_cite(world):
    """Language: a dialogue with no client Mandarin cannot honestly MATCH the both-languages row (nothing real to cite)."""
    call_id = mixed_call(world)
    ctx, report = synth(world, call_id)
    english_only = next(t for t in ctx.turns if t["speaker"] == "CLIENT" and not any("一" <= ch <= "鿿" for ch in t["text"]))
    row = next(r for r in report["language_checks"] if r["row_id"] == MIXED_BOTH)
    row["evidence"] = [{"turn_id": english_only["turn_id"], "quote": english_only["text"][:20]}]
    assert any("Mandarin" in e for e in world.g7.validator.errors(report, ctx))


# ============================================================================================ the register guide
def test_a_guide_change_stales_only_singlish_reports_and_validation_stays_clean(world, tmp_path, monkeypatch):
    directory = tmp_path / "guides"
    directory.mkdir()
    for path in register_guide.GUIDE_DIR.glob("register_singlish.*.md"):
        shutil.copyfile(path, directory / path.name)
    recorded = register_guide.guide_sha256()
    monkeypatch.setattr(register_guide, "GUIDE_DIR", directory)
    (directory / "register_singlish.v1.md").write_text(register_guide.guide_text() + "\nA wording change.\n", encoding="utf-8")
    specs = world.g7.inputs.specs
    singlish = {c for c in world.call_ids if register_guide.applies(specs[c])}
    statuses = world.g7.dependency_status(world.call_ids)
    assert {c for c, s in statuses.items() if s["status"] != "current"} == singlish
    assert world.g7.validate() == []
    assert report_of(world, sorted(singlish)[0])["source"]["register_guide_sha256"] == recorded


# ============================================================================================ multilingual semantic controls
CONTROL_CASES = [("M_filler_confirmation", "MANDARIN"), ("M_filler_fee", "MANDARIN"), ("M_filler_market", "EN_ZH_MIXED"),
                 ("M_filler_fee", "ENGLISH"), ("S_employee_singlish", "SINGLISH")]


@pytest.mark.parametrize("name,profile", CONTROL_CASES)
def test_a_multilingual_control_passes_g6_changes_one_turn_and_reaches_the_g7_brief(world, tmp_path, name, profile):
    from fyp_evaluation.dialogue_verifications import controls
    from fyp_evaluation.dialogue_verifications.inputs import load_inputs

    call_id = M.first_call(profile)
    destination = controls.build_multilingual_control(world.root, tmp_path / "control", name, call_id, profile, schema_dir=M.SCHEMAS,
                                                      audit_root=tmp_path / "audit")
    before = json.loads((world.root / "authoring/dialogues" / f"{call_id}.json").read_text(encoding="utf-8"))["turns"]
    after = json.loads((destination / "authoring/dialogues" / f"{call_id}.json").read_text(encoding="utf-8"))["turns"]
    changed = [(b["text"], a["text"]) for b, a in zip(before, after) if b != a]
    assert len(before) == len(after) and len(changed) == 1, (name, profile)
    old, new = changed[0]
    assert new != old and old.rstrip() in new
    builder = RequestBuilder(load_inputs(destination, M.SCHEMAS), M.SCHEMAS)
    assert json.dumps(new, ensure_ascii=False)[1:-1] in builder.brief_text(call_id)


def test_a_control_must_apply_to_its_profile_and_name_a_known_control(world, tmp_path):
    from fyp_evaluation.dialogue_verifications import controls

    with pytest.raises(ValueError, match="does not apply"):
        controls.build_multilingual_control(world.root, tmp_path / "x1", "S_employee_singlish", M.first_call("MANDARIN"), "MANDARIN", schema_dir=M.SCHEMAS)
    with pytest.raises(ValueError, match="unknown multilingual control"):
        controls.build_multilingual_control(world.root, tmp_path / "x2", "nope", M.first_call("MANDARIN"), "MANDARIN", schema_dir=M.SCHEMAS)
