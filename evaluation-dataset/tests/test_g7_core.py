"""G7 pure logic: provider-schema compatibility, request determinism and content, quote grounding, response validation
(coverage, linkage, severity, the citation rules), verdict/quality derivation and byte-exact compilation. Offline."""

import copy
import json
import re

import pytest
from jsonschema import Draft202012Validator

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.dialogue_verifications import config
from fyp_evaluation.dialogue_verifications.compile import (
    compile_artifact, derive_quality_review, derive_repair, derive_verdict,
)
from fyp_evaluation.dialogue_verifications.controls import BUY_SENTENCE
from fyp_evaluation.dialogue_verifications.grounding import locate
from fyp_evaluation.dialogue_verifications.request import DialogueVerificationContext, RequestBuilder
from fyp_evaluation.dialogue_verifications.run import synthesize_response
from fyp_evaluation.dialogue_verifications.validate_response import DialogueVerificationValidator, expand
from fyp_evaluation.llm.fake_gemini_transport import call_id_of
from g7_support import CALL_IDS, SCHEMAS, mismatch_report, quote_of, shared_inputs

X, NO_EDGE = "CALL_016", "CALL_027"
PROMPTS = SCHEMAS.parent / "prompts" / "stage7"

_BUILDER = None


def builder():
    global _BUILDER
    if _BUILDER is None:
        _BUILDER = RequestBuilder(shared_inputs(), SCHEMAS, PROMPTS)
    return _BUILDER


def ctx_of(call_id=X):
    return builder().context(call_id)


def validator():
    return DialogueVerificationValidator(builder().response_schema)


def errors(report, ctx=None):
    return validator().errors(report, ctx or ctx_of())


def ctx_with_turn(call_id, turn_id, transform):
    """A context over a COPY of the dialogue in which one turn's text is changed (the shared inputs stay untouched)."""
    inputs = copy.copy(shared_inputs())
    dialogue = copy.deepcopy(inputs.dialogues[call_id])
    turn = dialogue["turns"][turn_id]
    turn["text"] = transform(turn["text"])
    inputs.dialogues = {**inputs.dialogues, call_id: dialogue}
    return DialogueVerificationContext(inputs, call_id)


def has(errs, text):
    return any(text in e for e in errs)


# ============================================================================================ provider compatibility
def test_the_response_schema_follows_the_gemini_compatible_pattern():
    schema = builder().response_schema
    Draft202012Validator.check_schema(schema)
    bounded, forbidden = [], {"pattern", "const", "oneOf", "uniqueItems", "$ref", "anyOf", "allOf"}

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, path
                assert sorted(node.get("required", [])) == sorted(node["properties"]), f"{path}: every key is required"
            for key in ("minItems", "maxItems"):
                if key in node:
                    bounded.append((path, key, node[key]))
            assert not forbidden & set(node), path
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}/{index}")
    walk(schema, "")
    # the only array bounds are the 20 checks; nested bounds make Gemini answer HTTP 400 without detail
    assert sorted(bounded) == [("/properties/checks", "maxItems", 20), ("/properties/checks", "minItems", 20)]
    assert set(schema["required"]) == {"checks", "edge_cases", "issues", "quality_findings"}


def test_all_fifty_calls_accept_a_synthetic_report_and_compile_a_schema_valid_artifact():
    inputs, schema = shared_inputs(), Draft202012Validator(builder().artifact_schema)
    assert len(inputs.dialogues) == 50 and set(inputs.dialogues) == set(CALL_IDS)
    for call_id in CALL_IDS:
        ctx = ctx_of(call_id)
        report = synthesize_response(ctx)
        assert errors(report, ctx) == [], call_id
        assert [e["assignment_id"] for e in report["edge_cases"]] == ctx.assignment_ids
        artifact = compile_artifact(inputs, ctx, report, generation(call_id), builder().response_schema)
        assert artifact["verdict"] == "VALID" and artifact["quality_review"] == "CLEAR"
        assert list(schema.iter_errors(artifact)) == [], call_id
    assert sum(1 for c in CALL_IDS if not ctx_of(c).assignment_ids) == 10  # the ten calls without assignments


def generation(call_id):
    return {"attempt_id": "A001", "model": config.MODEL, "prompt_template_sha256": builder().prompt_template_sha256,
            "response_schema_sha256": builder().response_schema_sha256,
            "params": {"thinking_level": "high", "service_tier": "flex", "max_output_tokens": config.MAX_OUTPUT_TOKENS}}


# ============================================================================================ the request
def test_the_request_is_deterministic_high_thinking_flex_and_carries_the_whole_dialogue():
    first, second = builder().build(X, "A001"), builder().build(X, "A009")
    assert first == second  # no attempt id anywhere in the request
    assert first["generation_config"] == {"thinking_level": "high", "max_output_tokens": 65536}
    assert first["service_tier"] == "flex" and first["model"] == "gemini-3.8-flash" and first["store"] is False
    assert call_id_of(first) == X
    brief = json.loads(first["input"])
    ctx = ctx_of()
    assert [c["question_id"] for c in brief["checks"]] == ctx.order and len(brief["checks"]) == 20
    assert brief["dialogue"] == [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"]} for t in ctx.turns]
    assert [t["turn_id"] for t in brief["dialogue"]] == list(range(len(ctx.turns))) and len(ctx.turns) == 68
    assert {e["event_id"]: e["turn_id"] for e in brief["events"]} == ctx.turn_of_event
    assert [e["assignment_id"] for e in brief["edge_cases"]] == ctx.assignment_ids == ["EA_027", "EA_028"]
    assert all(c["expected"] for c in brief["checks"]) and "constraints" in brief and "g3" in brief
    # Python-owned material stays out of the model-facing text
    text = first["input"]
    assert not re.search(r"[0-9a-f]{64}", text) and "event_alignment" not in text and "review_notes" not in text
    assert "plan_verification" not in text and '"spans"' not in text
    system = first["system_instruction"]
    assert "{{" not in system and "synthetic_nric_fin" in system and "EDGE_CASE_ONLY" in system  # normalization is present
    assert '"C01"' in system and "cross_check_rules" in system


def test_the_request_hashes_are_stable_for_all_fifty_calls():
    hashes = {c: json.dumps(builder().build(c, "A001"), sort_keys=True) for c in CALL_IDS}
    again = {c: json.dumps(builder().build(c, "A002"), sort_keys=True) for c in CALL_IDS}
    assert hashes == again and len(set(hashes.values())) == 50


def test_g5_request_bytes_and_public_helpers_survive_the_shared_gemini_extraction(tmp_path):
    from fyp_evaluation.llm.gemini_runner import GeminiFlexMixin
    from fyp_evaluation.plan_verifications import run as g5_run
    from g7_support import RELEASE

    assert issubclass(g5_run.PlanVerificationRunner, GeminiFlexMixin)
    for name in ("parse_interaction", "estimate_cost", "usage_summary", "synthesize_response", "describe"):
        assert callable(getattr(g5_run, name))
    assert g5_run.estimate_cost({"input_tokens": 1_000_000, "cached_tokens": 0, "output_tokens": 0, "thought_tokens": 0}) == 0.375
    assert g5_run.estimate_cost({"input_tokens": 1, "cached_tokens": 0, "output_tokens": 1, "thought_tokens": 0}, tier="standard") is None
    runner = g5_run.PlanVerificationRunner(RELEASE, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, inputs=shared_inputs().upstream.upstream)
    rows = {row["call_id"]: row["request_sha256"] for row in runner.dry_run()["calls"]}
    assert len(rows) == 50
    for call_id, sha in rows.items():  # the request hash each production G5 attempt recorded when it was really sent
        accepted = parse_json_text((RELEASE / "authoring/plan_verification_provenance" / f"{call_id}.json").read_text(encoding="utf-8"))
        recorded = next(a for a in accepted["attempts"] if a["attempt_id"] == accepted["accepted"]["attempt_id"])["request_sha256"]
        assert recorded == sha, call_id


# ============================================================================================ advisory/execution and FAIL semantics
def template():
    return (PROMPTS / "dialogue_verification.v1.md").read_text(encoding="utf-8")


def squashed(text):
    return re.sub(r"\s+", " ", text)


def test_the_prompt_follows_each_checks_target_instead_of_equating_every_fail_with_absence():
    text = squashed(template())
    for blanket in ("must NOT establish the condition anywhere", "Is each expected FAIL really absent", "if the gold says FAIL, the dialogue must NOT"):
        assert blanket not in text
    for required in ("follow each row's own target", "target.must", "evidence_mode", "requirement_type", "OBLIGATION (C01-C08)",
                     "PROHIBITION (C09-C12)", "AUTHORED VIOLATION", "A FAIL with no such violation in the dialogue is a MISMATCH",
                     "Do not treat every FAIL as an absence"):
        assert required in text, required
    system = builder().build(X, "A001")["system_instruction"]
    assert '"requirement_type":"PROHIBITION"' in system  # the checklist the model reads carries the obligation/prohibition split


def test_the_prompt_requires_actual_advisory_and_execution_realization_in_the_downstream_intent_order():
    text = squashed(template())
    assert "NOT a chronology" not in text and "label list" not in text  # the array order is now the intended segment order
    for required in ("ADVISORY AND EXECUTION REALIZATION", "call.call_intents", "the array order is the intended segment order",
                     "canonical vocabulary order fixed upstream", "it was not chosen per call", "authoring policy",
                     "it is not a per-event chronology", "RECOMMENDS the transaction or product",
                     "BEFORE the final transaction confirmation or commitment", "Preliminary client mentions of a product, amount or intention may come first",
                     "C07 FAIL: the advisory component is retained", "WITHOUT a concrete rationale",
                     "no recommendation at all (an execution-only call)", "MISSING_SUPPORT", "advisory_expectation",
                     "never from the events array"):
        assert required in text, required


@pytest.mark.parametrize("call_id", CALL_IDS)
def test_every_brief_states_the_ordered_intents_and_a_target_consistent_with_the_requirement_type(call_id):
    """The prompt tells the model to follow target.must; this pins that the data it reads matches the four FAIL/PASS quadrants."""
    brief = json.loads(builder().build(call_id, "A001")["input"])
    assert brief["call"]["call_intents"] == ["ADVISORY", "EXECUTION"]
    requirement = {c["id"]: c.get("requirement_type") for c in shared_inputs().documents["checklist"] if c["type"] == "COMPLIANCE"}
    for row in brief["checks"]:
        if row["kind"] != "COMPLIANCE":
            continue
        verdict, kind = row["expected"], requirement[row["question_id"]]
        must_establish = (verdict == "PASS") == (kind == "OBLIGATION")  # obligation PASS / prohibition FAIL establish; the others do not
        assert row["target"]["must"] == ("ESTABLISH" if must_establish else "NOT_ESTABLISH"), (call_id, row["question_id"])
        assert row["evidence_mode"] == ("PRESENCE" if must_establish else "ABSENCE")
        assert row["target"]["condition_ref"].endswith("fail_condition" if kind == "PROHIBITION" else "pass_condition") or kind == "OBLIGATION"


def test_all_four_pass_fail_quadrants_occur_so_the_mapping_is_not_vacuous():
    seen = set()
    requirement = {c["id"]: c.get("requirement_type") for c in shared_inputs().documents["checklist"] if c["type"] == "COMPLIANCE"}
    for call_id in CALL_IDS:
        for row in json.loads(builder().build(call_id, "A001")["input"])["checks"]:
            if row["kind"] == "COMPLIANCE":
                seen.add((requirement[row["question_id"]], row["expected"], row["target"]["must"]))
    assert seen == {("OBLIGATION", "PASS", "ESTABLISH"), ("OBLIGATION", "FAIL", "NOT_ESTABLISH"),
                    ("PROHIBITION", "PASS", "NOT_ESTABLISH"), ("PROHIBITION", "FAIL", "ESTABLISH")}


def test_a_prohibition_fail_marked_match_must_cite_the_authored_violation_but_a_prohibition_pass_needs_no_quote():
    ctx = ctx_of("CALL_049")  # C12 gold FAIL (the employee guarantees returns); C09-C11 gold PASS
    report = synthesize_response(ctx)
    c12 = next(c for c in report["checks"] if c["question_id"] == "C12")
    assert c12["evidence"] and ctx.checks["C12"]["evidence_mode"] == "PRESENCE" and ctx.gold["C12"]["expected_verdict"] == "FAIL"
    assert errors(report, ctx) == []
    c12["evidence"] = []
    assert has(errors(report, ctx), "check C12: a PRESENCE check marked MATCH must cite")  # no authored violation quoted
    c12["evidence"] = [quote_of(ctx, "EV_001")]  # speech of ANOTHER check's event is not the violation
    assert has(errors(report, ctx), "check C12: a PRESENCE check marked MATCH must cite")
    for q in ("C09", "C10", "C11"):
        assert ctx.checks[q]["evidence_mode"] == "ABSENCE" and next(c for c in synthesize_response(ctx)["checks"] if c["question_id"] == q)["evidence"] == []


def test_a_missing_recommendation_is_reportable_as_a_grounded_free_c07_mismatch_that_makes_the_report_invalid():
    ctx = ctx_of("CALL_001")  # C07 gold FAIL: the dialogue has no recommendation at all
    assert ctx.gold["C07"]["expected_verdict"] == "FAIL" and ctx.checks["C07"]["evidence_mode"] == "ABSENCE"
    report = mismatch_report(ctx, "C07", None, code="MISSING_SUPPORT", scope="G6_ONLY", evidence_on_issue=False)
    assert errors(report, ctx) == []  # the required act is ABSENT: no invented citation is demanded
    artifact = compiled(report, "CALL_001")
    assert artifact["verdict"] == "INVALID" and artifact["recommended_repair"] == "G6_ONLY"
    assert artifact["issues"][0]["code"] == "MISSING_SUPPORT" and artifact["issues"][0]["evidence"] == []
    # a recommendation placed after the confirmation, in contrast, must be quoted
    late = mismatch_report(ctx, "C07", None, code="COMPLIANCE_MEANING", evidence_on_issue=False)
    assert has(errors(late, ctx), "must cite the dialogue speech it rests on")


C07_FAIL_CALLS = ["CALL_001", "CALL_024", "CALL_031", "CALL_036", "CALL_042", "CALL_047", "CALL_049"]
ENDORSEMENTS = ["I'd go ahead with this.", "I think proceeding is the right step.", "I recommend going ahead with this."]


def first_confirmation_turn(ctx):
    return min(ctx.turn_of_event[e["event_id"]] for e in ctx.events if "C08" in e["question_ids"])


@pytest.mark.parametrize("sentence", ENDORSEMENTS)
@pytest.mark.parametrize("call_id", C07_FAIL_CALLS)
def test_g7_accepts_a_grounded_c07_match_citing_a_varied_endorsement_in_isolated_dialogues(call_id, sentence):
    """Isolated fixtures (never the committed dialogues): an endorsement written by the G6 model as supporting speech before the
    confirmation is quotable evidence of the advisory component; it carries no planned event and no rationale."""
    base = ctx_of(call_id)
    assert base.gold["C07"]["expected_verdict"] == "FAIL" and not any("C07" in e["question_ids"] for e in base.events)  # plan facts
    first = first_confirmation_turn(base)
    turn = max(t["turn_id"] for t in base.turns[:first] if t["speaker"] == "EMPLOYEE")
    ctx = ctx_with_turn(call_id, turn, lambda text: text.rstrip() + " " + sentence)
    report = synthesize_response(ctx)
    c07 = next(c for c in report["checks"] if c["question_id"] == "C07")
    c07["evidence"] = [{"turn_id": turn, "quote": sentence}]
    c07["rationale"] = "The employee endorses proceeding before the confirmation without giving any basis."
    assert errors(report, ctx) == []
    checks, _, _, _, errs = expand(report, ctx)
    (item,) = next(c for c in checks if c["question_id"] == "C07")["evidence"]
    assert errs == [] and item["turn_id"] == turn < first and item["speaker"] == "EMPLOYEE" and item["event_ids"] == []
    artifact = compile_artifact(shared_inputs(), ctx, report, generation(call_id), builder().response_schema)
    assert artifact["verdict"] == "VALID" and artifact["quality_review"] == "CLEAR"


def test_c07_pass_calls_keep_a_planned_recommendation_event_that_precedes_the_first_confirmation():
    """Plan structure plus the committed alignment of the 43 accepted C07 PASS dialogues: unchanged by the targeted repair."""
    inputs = shared_inputs()
    for call_id in CALL_IDS:
        ctx = ctx_of(call_id)
        carriers = [e["event_id"] for e in ctx.events if "C07" in e["question_ids"]]
        if call_id in C07_FAIL_CALLS:
            assert carriers == []
            continue
        (event_id,) = carriers
        confirmations = [inputs.dialogues[call_id]["event_alignment"][i] for i in range(len(inputs.dialogues[call_id]["event_alignment"]))
                         if "C08" in next(e for e in ctx.events if e["event_id"] == inputs.dialogues[call_id]["event_alignment"][i]["event_id"])["question_ids"]]
        span = next(s for s in inputs.dialogues[call_id]["event_alignment"] if s["event_id"] == event_id)
        assert all((span["turn_id"], span["start"]) < (c["turn_id"], c["start"]) for c in confirmations), call_id


# ============================================================================================ grounding
def test_locate_returns_the_exact_substring_and_every_occurrence():
    text = "Thank you. Thank you for your time — it's “great”."
    exact, spans, reason = locate("thank you", text)
    assert exact is None and reason  # case-sensitive
    exact, spans, reason = locate("Thank you", text)
    assert (exact, spans, reason) == ("Thank you", [[0, 9], [11, 20]], None)  # both occurrences retained, none arbitrary
    exact, spans, _ = locate('time - it\'s "great"', text)  # dashes and curly quotes are normalised, the stored text is exact
    assert exact == "time — it's “great”" and text[spans[0][0]:spans[0][1]] == exact
    assert locate("  ", text)[2] and locate("missing words", text)[2]
    assert locate("Thank   you", text)[0] == "Thank you"  # whitespace runs collapse


def test_locate_rejects_a_quote_matching_different_exact_substrings():
    exact, spans, reason = locate("a-b", "a-b and a—b")
    assert exact is None and "ambiguous" in reason


# ============================================================================================ response validation
def test_the_synthetic_report_is_valid_and_expanded_with_exact_spans_and_event_ids():
    ctx, report = ctx_of(), synthesize_response(ctx_of())
    checks, edges, issues, quality, errs = expand(report, ctx)
    assert errs == [] and issues == [] and quality == [] and len(checks) == 20
    c08 = next(c for c in checks if c["question_id"] == "T05")
    (item,) = c08["evidence"]
    turn = ctx.turns[item["turn_id"]]
    start, end = item["spans"][0]
    assert turn["text"][start:end] == item["quote"] and item["speaker"] == turn["speaker"] == "CLIENT"
    assert item["event_ids"] == ["EV_016"]  # the planned event whose G6 span overlaps the quote
    assert next(c for c in checks if c["question_id"] == "C04")["evidence"] == []  # an omission needs no citation


def test_checks_must_be_the_twenty_questions_in_order():
    ctx = ctx_of()
    for mutate, text in [(lambda r: r["checks"].pop(), "exactly the 20"),
                         (lambda r: r["checks"].reverse(), "exactly the 20"),
                         (lambda r: r["checks"].append(copy.deepcopy(r["checks"][0])), "exactly the 20"),
                         (lambda r: r["checks"][3].update(question_id="C03"), "exactly the 20")]:
        report = synthesize_response(ctx)
        mutate(report)
        errs = errors(report, ctx)
        assert errs and (has(errs, text) or has(errs, "schema")), errs


def test_edge_cases_must_match_the_frozen_assignments_exactly_and_in_order():
    ctx = ctx_of()
    assert ctx.assignment_ids == ["EA_027", "EA_028"]
    for mutate in (lambda r: r["edge_cases"].pop(), lambda r: r["edge_cases"].reverse(), lambda r: r["edge_cases"].clear(),
                   lambda r: r["edge_cases"].append(dict(copy.deepcopy(r["edge_cases"][0]), assignment_id="EA_999")),
                   lambda r: r["edge_cases"].append(copy.deepcopy(r["edge_cases"][0]))):
        report = synthesize_response(ctx)
        mutate(report)
        assert has(errors(report, ctx), "edge_cases must list exactly")
    quiet = ctx_of(NO_EDGE)
    assert quiet.assignment_ids == [] and errors(synthesize_response(quiet), quiet) == []  # an empty array is the correct answer
    extra = synthesize_response(quiet)
    extra["edge_cases"].append({"assignment_id": "EA_001", "evidence": [], "rationale": "x", "assessment": "MATCH", "issue_ids": []})
    assert has(errors(extra, quiet), "edge_cases must list exactly")


def test_an_edge_case_marked_match_must_cite_an_event_that_carries_it():
    ctx = ctx_of()
    carriers = ctx.events_of_assignment["EA_027"]
    assert carriers  # EA_027 (same-turn K01 correction) is carried by planned events
    report = synthesize_response(ctx)
    report["edge_cases"][0]["evidence"] = []
    assert has(errors(report, ctx), "edge case EA_027: marked MATCH")
    report["edge_cases"][0]["evidence"] = [{"turn_id": 0, "quote": ctx.turns[0]["text"][:20]}]  # real speech, but not a carrier
    assert has(errors(report, ctx), "edge case EA_027: marked MATCH")


def test_a_presence_check_marked_match_must_cite_one_of_its_own_events():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    t05 = next(c for c in report["checks"] if c["question_id"] == "T05")
    t05["evidence"] = []
    assert has(errors(report, ctx), "check T05: a PRESENCE check marked MATCH must cite")
    t05["evidence"] = [quote_of(ctx, "EV_012")]  # speech of ANOTHER question's event does not count
    assert has(errors(report, ctx), "check T05: a PRESENCE check marked MATCH must cite")


# ---- the grounding rules (Codex round 2): every MISMATCH issue except MISSING_SUPPORT is grounded in actual speech ---------
def test_an_unquoted_extraction_state_mismatch_alleging_sell_to_buy_is_rejected_and_the_quoted_one_is_accepted():
    ctx = ctx_with_turn(X, 67, lambda text: BUY_SENTENCE + " " + text)  # the control-C closing turn
    claim = mismatch_report(ctx, "T05", None, evidence_on_issue=False)
    errs = errors(claim, ctx)
    assert has(errs, "issue I1: a MISMATCH issue coded EXTRACTION_STATE must cite the dialogue speech it rests on"), errs
    cite = {"turn_id": 67, "quote": "My final instruction for that same holding is to buy instead."}
    grounded = mismatch_report(ctx, "T05", cite, scope="REVIEW")
    assert errors(grounded, ctx) == []
    checks, _, issues, _, _ = expand(grounded, ctx)
    (item,) = issues[0]["evidence"]
    assert item["quote"] == cite["quote"] and item["event_ids"] == [] and item["speaker"] == "CLIENT"  # supporting speech, no event
    start, end = item["spans"][0]
    assert ctx.turns[67]["text"][start:end] == cite["quote"]
    assert next(c for c in checks if c["question_id"] == "T05")["assessment"] == "MISMATCH"


@pytest.mark.parametrize("code", ["COMPLIANCE_MEANING", "OMISSION_LEAKAGE", "EXTRACTION_STATE", "SCENARIO_CONTRADICTION", "EDGE_CASE",
                                  "CROSS_CHECK_INTERFERENCE"])
def test_every_substantive_mismatch_code_must_be_grounded(code):
    ctx = ctx_of()
    assert has(errors(mismatch_report(ctx, "C04", None, code=code, evidence_on_issue=False), ctx), f"coded {code} must cite")
    assert errors(mismatch_report(ctx, "C04", {"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}, code=code), ctx) == []


def test_missing_support_and_unclear_findings_may_have_no_citation():
    ctx = ctx_of()
    assert errors(mismatch_report(ctx, "T04", None, code="MISSING_SUPPORT", evidence_on_issue=False), ctx) == []
    for code in config.ISSUE_CODES:
        report = mismatch_report(ctx, "C04", None, code=code, assessment="UNCLEAR", evidence_on_issue=False)
        assert errors(report, ctx) == [], code
    # a genuine omission/MATCH on an absence check needs no quotation either
    assert next(c for c in synthesize_response(ctx)["checks"] if c["question_id"] == "C04")["evidence"] == []


def test_linkage_and_severity_rules():
    ctx = ctx_of()
    cite = quote_of(ctx, "EV_012")

    def report(**changes):
        r = mismatch_report(ctx, "T01", cite)
        for path, value in changes.items():
            node = r
            *head, last = path.split(".")
            for part in head:
                node = node[int(part)] if part.isdigit() else node[part]
            node[int(last) if last.isdigit() else last] = value
        return r
    assert errors(report(), ctx) == []
    # a MATCH row may not link issues; a non-MATCH row needs one; severity must agree with the most severe linked issue
    t01 = 12 + 2  # index of T01 in the checks list
    assert has(errors(report(**{f"checks.{t01}.assessment": "MATCH"}), ctx), "a MATCH row must not reference issues")
    assert has(errors(report(**{f"checks.{t01}.issue_ids": []}), ctx), "needs at least one linked issue")
    assert has(errors(report(**{f"checks.{t01}.assessment": "UNCLEAR"}), ctx), "must equal the most severe linked issue")
    assert has(errors(report(**{"issues.0.assessment": "UNCLEAR"}), ctx), "must equal the most severe linked issue")
    assert has(errors(report(**{f"checks.{t01}.issue_ids": ["I9"]}), ctx), "unknown issue 'I9'")
    assert has(errors(report(**{"issues.0.question_ids": ["T02"]}), ctx), "do not include T01")
    r = report()
    r["issues"][0]["question_ids"] = ["T01", "T02"]  # an issue naming T02 must be referenced by the T02 check
    assert has(errors(r, ctx), "names T02 but check T02 does not reference it")
    r = report()
    r["issues"][0]["assignment_ids"] = ["EA_027"]
    assert has(errors(r, ctx), "names EA_027 but edge case EA_027 does not reference it")
    r["edge_cases"][0].update(assessment="MISMATCH", issue_ids=["I1"], evidence=[])
    assert errors(r, ctx) == []  # linked in both directions
    r = report()
    r["issues"][0]["assignment_ids"] = ["EA_999"]
    assert has(errors(r, ctx), "are not assignments of this call")
    r = report()
    r["issues"].append(copy.deepcopy(r["issues"][0]))
    assert has(errors(r, ctx), "issue_id values must be distinct")
    r = report()
    r["issues"][0]["issue_id"] = "X1"
    assert has(errors(r, ctx), "must look like I1")


def test_a_global_issue_with_no_question_or_assignment_is_allowed():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    report["issues"] = [{"issue_id": "I1", "assessment": "MISMATCH", "code": "SCENARIO_CONTRADICTION", "question_ids": [], "assignment_ids": [],
                         "evidence": [{"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}], "explanation": "The closing contradicts the scenario.",
                         "repair_scope": "UPSTREAM_REVIEW"}]
    assert errors(report, ctx) == []


def test_quality_findings_need_a_citation_and_unique_ids():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    report["quality_findings"] = [{"quality_id": "Q1", "severity": "SERIOUS", "code": "REPETITION", "evidence": [], "explanation": "Repeats."}]
    assert has(errors(report, ctx), "must cite the turn it concerns")
    report["quality_findings"][0]["evidence"] = [{"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}]
    assert errors(report, ctx) == []
    report["quality_findings"].append(copy.deepcopy(report["quality_findings"][0]))
    assert has(errors(report, ctx), "quality_id values must be distinct")
    report["quality_findings"][1]["quality_id"] = "bad"
    assert has(errors(report, ctx), "must look like Q1")


def test_citations_must_be_real_speech_of_the_cited_turn():
    ctx = ctx_of()

    def with_cite(cite):
        report = mismatch_report(ctx, "C04", cite)
        return errors(report, ctx)
    ok = {"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}
    assert with_cite(ok) == []
    assert has(with_cite({"turn_id": 999, "quote": "x"}), "is not a turn of this dialogue")
    assert has(with_cite({"turn_id": -1, "quote": "x"}), "is not a turn of this dialogue")
    assert has(with_cite({"turn_id": 66, "quote": "this sentence was never said"}), "not a substring of the cited turn")
    assert has(with_cite({"turn_id": 66, "quote": "   "}), "empty or whitespace-only")
    # a quote taken from a different turn is not accepted against this one
    assert has(with_cite({"turn_id": 65, "quote": ctx.turns[66]["text"][:40]}), "not a substring of the cited turn")
    # a quote spanning two turns is not a substring of either
    spanning = ctx.turns[65]["text"][-20:] + " " + ctx.turns[66]["text"][:20]
    assert has(with_cite({"turn_id": 65, "quote": spanning}), "not a substring") and has(with_cite({"turn_id": 66, "quote": spanning}), "not a substring")
    assert has(with_cite({"turn_id": 66, "quote": ctx.turns[66]["text"][:30] + "x" * 400}), "longer than 400")


def test_repeated_quotations_retain_every_occurrence_in_the_turn():
    ctx = ctx_with_turn(X, 66, lambda text: "Thank you. Thank you. " + text)
    report = mismatch_report(ctx, "C04", {"turn_id": 66, "quote": "Thank you"}, code="OMISSION_LEAKAGE")
    _, _, issues, _, errs = expand(report, ctx)
    expected = [[m.start(), m.end()] for m in re.finditer("Thank you", ctx.turns[66]["text"])]
    assert len(expected) >= 3 and expected[:2] == [[0, 9], [11, 20]]  # the turn already said it once more
    assert errs == [] and issues[0]["evidence"][0]["spans"] == expected  # every occurrence, never an arbitrary first match


def test_planned_text_dashes_are_grounded_to_the_exact_source():
    for call_id in CALL_IDS:
        ctx = ctx_of(call_id)
        dashed = [e["event_id"] for e in ctx.events if "—" in e["authoring"]["planned_text"]]
        if dashed:
            break
    assert dashed
    text = ctx.planned_text(dashed[0])
    ascii_quote = text[:60].replace("—", "-")
    report = mismatch_report(ctx, "C01", {"turn_id": ctx.turn_of_event[dashed[0]], "quote": ascii_quote}, code="OMISSION_LEAKAGE")
    _, _, issues, _, errs = expand(report, ctx)
    assert errs == [] and issues[0]["evidence"][0]["quote"] == text[:60] and dashed[0] in issues[0]["evidence"][0]["event_ids"]


def test_size_limits_are_enforced_by_python_not_the_provider_schema():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    report["checks"][0]["rationale"] = "x" * 701
    assert has(errors(report, ctx), "rationale longer than 700")
    report = synthesize_response(ctx)
    report["checks"][0]["rationale"] = "  "
    assert has(errors(report, ctx), "rationale is empty")
    report = mismatch_report(ctx, "C04", {"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}, code="OMISSION_LEAKAGE")
    report["issues"][0]["explanation"] = "y" * 1201
    assert has(errors(report, ctx), "explanation longer than 1200")
    report = synthesize_response(ctx)
    report["checks"][0]["evidence"] = [{"turn_id": 66, "quote": ctx.turns[66]["text"][:i + 3]} for i in range(7)]
    assert has(errors(report, ctx), "more than 6 evidence items")
    report = synthesize_response(ctx)
    report["issues"] = [{"issue_id": f"I{i}", "assessment": "UNCLEAR", "code": "MISSING_SUPPORT", "question_ids": [], "assignment_ids": [],
                         "evidence": [], "explanation": "e", "repair_scope": "REVIEW"} for i in range(1, 22)]
    assert has(errors(report, ctx), "more than 20 issues")
    report = synthesize_response(ctx)
    report["quality_findings"] = [{"quality_id": f"Q{i}", "severity": "NOTE", "code": "REPETITION",
                                  "evidence": [{"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}], "explanation": "e"} for i in range(1, 12)]
    assert has(errors(report, ctx), "more than 10 quality findings")


def test_unknown_keys_enums_and_missing_fields_are_schema_errors():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    report["verdict"] = "VALID"  # the model never states a verdict
    assert has(errors(report, ctx), "schema")
    report = synthesize_response(ctx)
    del report["quality_findings"]
    assert has(errors(report, ctx), "schema")
    report = synthesize_response(ctx)
    report["checks"][0]["assessment"] = "MAYBE"
    assert has(errors(report, ctx), "schema")
    report = synthesize_response(ctx)
    report["checks"][0]["evidence"] = [{"turn_id": "66", "quote": "x"}]
    assert has(errors(report, ctx), "schema")


# ============================================================================================ verdict, quality and compilation
@pytest.mark.parametrize("checks, edges, issues, expected", [
    (["MATCH"], [], [], "VALID"),
    (["MATCH", "UNCLEAR"], [], [], "UNCLEAR"),
    (["MATCH", "UNCLEAR"], [], ["MISMATCH"], "INVALID"),
    (["MATCH"], ["MISMATCH"], [], "INVALID"),  # an edge-case MISMATCH decides the verdict although every check MATCHes
    (["MATCH"], ["UNCLEAR"], [], "UNCLEAR"),
    (["MATCH"], [], ["MISMATCH"], "INVALID"),  # a whole-dialogue issue decides it too
    (["MATCH"], [], ["UNCLEAR"], "UNCLEAR"),
])
def test_verdict_precedence_covers_checks_edge_cases_and_issues(checks, edges, issues, expected):
    as_rows = lambda values: [{"assessment": v} for v in values]  # noqa: E731
    assert derive_verdict(as_rows(checks), as_rows(edges), as_rows(issues)) == expected


def test_repair_and_quality_derivation():
    assert derive_repair("VALID", []) is None and derive_repair("UNCLEAR", []) == "REVIEW"
    issue = lambda scope, assessment="MISMATCH": {"repair_scope": scope, "assessment": assessment}  # noqa: E731
    assert derive_repair("INVALID", [issue("G6_ONLY")]) == "G6_ONLY"
    assert derive_repair("INVALID", [issue("G6_ONLY"), issue("REVIEW")]) == "REVIEW"
    assert derive_repair("INVALID", [issue("REVIEW"), issue("UPSTREAM_REVIEW"), issue("G6_ONLY")]) == "UPSTREAM_REVIEW"
    assert derive_repair("INVALID", [issue("G6_ONLY"), issue("UPSTREAM_REVIEW", "UNCLEAR")]) == "G6_ONLY"  # only MISMATCH scopes count
    assert derive_quality_review([]) == "CLEAR" and derive_quality_review([{"severity": "NOTE"}]) == "CLEAR"
    assert derive_quality_review([{"severity": "NOTE"}, {"severity": "SERIOUS"}]) == "REVIEW_REQUIRED"


def compiled(report, call_id=X):
    ctx = ctx_of(call_id)
    return compile_artifact(shared_inputs(), ctx, report, generation(call_id), builder().response_schema)


def test_an_edge_case_mismatch_with_twenty_matching_checks_makes_the_report_invalid():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    report["edge_cases"][1].update(assessment="MISMATCH", issue_ids=["I1"], evidence=[])
    report["issues"] = [{"issue_id": "I1", "assessment": "MISMATCH", "code": "MISSING_SUPPORT", "question_ids": [], "assignment_ids": ["EA_028"],
                         "evidence": [], "explanation": "The shorthand was never spoken.", "repair_scope": "G6_ONLY"}]
    artifact = compiled(report)
    assert artifact["summary"]["checks"] == {"match": 20, "mismatch": 0, "unclear": 0}
    assert artifact["verdict"] == "INVALID" and artifact["recommended_repair"] == "G6_ONLY"
    assert artifact["edge_cases"][1]["edge_case_id"] == "EC_FINANCIAL_SHORTHAND" and artifact["edge_cases"][1]["assessment"] == "MISMATCH"
    assert list(Draft202012Validator(builder().artifact_schema).iter_errors(artifact)) == []


def test_a_global_issue_alone_makes_the_report_invalid():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    report["issues"] = [{"issue_id": "I1", "assessment": "MISMATCH", "code": "SCENARIO_CONTRADICTION", "question_ids": [], "assignment_ids": [],
                         "evidence": [{"turn_id": 66, "quote": ctx.turns[66]["text"][:30]}], "explanation": "Contradicts the scenario.",
                         "repair_scope": "UPSTREAM_REVIEW"}]
    artifact = compiled(report)
    assert artifact["verdict"] == "INVALID" and artifact["recommended_repair"] == "UPSTREAM_REVIEW"
    assert artifact["summary"]["issues_by_code"] == {"SCENARIO_CONTRADICTION": 1}


def test_a_serious_quality_finding_never_changes_the_semantic_verdict():
    ctx = ctx_of()
    report = synthesize_response(ctx)
    report["quality_findings"] = [{"quality_id": "Q1", "severity": "SERIOUS", "code": "ARTIFICIAL_EXTENSION",
                                  "evidence": [{"turn_id": 62, "quote": ctx.turns[62]["text"][:40]}], "explanation": "Administrative filler."},
                                 {"quality_id": "Q2", "severity": "NOTE", "code": "REPETITION",
                                  "evidence": [{"turn_id": 64, "quote": ctx.turns[64]["text"][:40]}], "explanation": "Mild repetition."}]
    artifact = compiled(report)
    assert artifact["verdict"] == "VALID" and artifact["recommended_repair"] is None
    assert artifact["quality_review"] == "REVIEW_REQUIRED" and artifact["summary"]["quality_by_severity"] == {"NOTE": 1, "SERIOUS": 1}
    assert [f["evidence"][0]["speaker"] for f in artifact["quality_findings"]] == [ctx.turns[62]["speaker"], ctx.turns[64]["speaker"]]


def test_compilation_is_deterministic_python_owned_and_records_expected_gold():
    first, second = compiled(synthesize_response(ctx_of())), compiled(synthesize_response(ctx_of()))
    assert serialize_json(first) == serialize_json(second)
    inputs = shared_inputs()
    assert first["dialogue_sha256"] == inputs.dialogue_hashes[X] and first["plan_verification_sha256"] == inputs.report_hashes[X]
    assert first["generation"]["params"]["thinking_level"] == "high" and first["stage"] == "G7"
    t05 = next(c for c in first["checks"] if c["question_id"] == "T05")
    assert t05["kind"] == "EXTRACTION" and t05["expected"] == {"kind": "extraction", "value_type": "text", "expected_value": "SELL"}
    c04 = next(c for c in first["checks"] if c["question_id"] == "C04")
    assert c04["expected"] == {"kind": "compliance", "expected_verdict": "FAIL"}
    assert [c["question_id"] for c in first["checks"]] == ctx_of().order
    assert parse_json_text(serialize_json(first)) == first
