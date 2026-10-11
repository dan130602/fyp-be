"""G5 core: grounding, brief and request, response rules, verdict derivation, provider parsing, cost. Offline only."""

import copy
import json

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.plan_verifications import config
from fyp_evaluation.plan_verifications.compile import derive_repair, derive_verdict
from fyp_evaluation.plan_verifications.grounding import ground, normalize
from fyp_evaluation.plan_verifications.request import RequestBuilder, VerificationContext, build_brief
from fyp_evaluation.plan_verifications.run import estimate_cost, parse_interaction, synthesize_response, usage_summary
from fyp_evaluation.plan_verifications.validate_response import VerificationValidator, expand
from fyp_evaluation.llm.fake_gemini_transport import interaction_body
from g5_support import CALL_IDS, SCHEMAS, mismatch_report, shared_inputs

CALL = "CALL_021"  # C05 FAIL (absence), EV_001 = employee C01 act, EV_015 = client FINAL T05 statement


@pytest.fixture(scope="module")
def builder():
    return RequestBuilder(shared_inputs(), SCHEMAS)


@pytest.fixture(scope="module")
def validator(builder):
    return VerificationValidator(builder.response_schema)


@pytest.fixture()
def ctx(builder):
    return builder.context(CALL)


# ============================================================================== grounding
def test_grounding_exact_and_narrow_normalisation():
    source = "I’d like to use—the “best”  option… really."
    assert ground("I’d like to use", source) == ("I’d like to use", None)
    assert ground("I'd like to use-the \"best\" option...", source) == ("I’d like to use—the “best”  option…", None)
    assert ground("  the \"best\" option  ", source)[0] == "the “best”  option"
    assert normalize("a  \t b")[0] == "a b"


def test_grounding_rejections():
    assert ground("", "text")[1].startswith("quote is empty")
    assert ground("   ", "text")[1].startswith("quote is empty")
    assert ground("Text", "text")[1].startswith("quote is not a substring")  # case-sensitive
    assert ground("best option", "the bestoption")[1].startswith("quote is not a substring")  # no whitespace invention
    assert ground("a-b", "first a—b then a-b")[1].startswith("quote is ambiguous")  # two different exact substrings
    assert ground("a-b", "a-b and a-b") == ("a-b", None)  # identical repeats are not ambiguous


# ============================================================================== brief and request
def test_brief_is_deterministic_complete_and_free_of_operational_fields(builder, ctx):
    inputs = shared_inputs()
    brief = build_brief(inputs, ctx)
    assert build_brief(inputs, VerificationContext(inputs, CALL)) == brief
    assert [c["question_id"] for c in brief["checks"]] == list(config.QUESTION_IDS)
    assert [e["event_id"] for e in brief["events"]] == [e["event_id"] for e in inputs.plans[CALL]["planned_events"]]
    text = builder.brief_text(CALL)
    for forbidden in ("sha256", "text_template", '"start"', '"end"', "spoken_form", "allowed_speakers", "attempt_id"):
        assert forbidden not in text, forbidden
    plan = inputs.plans[CALL]
    assert len(brief["constraints"]) == len(plan["constraints"]) + len(plan["derived_constraints"])
    c05 = next(c for c in brief["checks"] if c["question_id"] == "C05")
    assert c05["expected"] == "FAIL" and c05["target"]["must"] == "NOT_ESTABLISH" and c05["evidence_event_ids"] == []
    assert c05["target"]["condition_text"] == ctx.semantics["C05"]["pass_condition"]
    t05 = next(c for c in brief["checks"] if c["question_id"] == "T05")
    assert t05["expected"] == {"value_type": "text", "value": "BUY"}


def test_request_settings_are_the_approved_ones_and_attempt_independent(builder):
    body = builder.build(CALL, "A001")
    assert body == builder.build(CALL, "A042")
    assert body["model"] == "gemini-3.8-flash" and body["service_tier"] == "flex" and body["store"] is False
    assert body["generation_config"] == {"thinking_level": "medium", "max_output_tokens": 32768}
    assert body["response_format"] == {"type": "text", "mime_type": "application/json", "schema": builder.response_schema}
    for absent in ("tools", "previous_interaction_id", "background", "stream", "response_mime_type", "metadata"):
        assert absent not in body
    assert json.loads(body["input"])["call"]["call_id"] == CALL
    assert "{{" not in body["system_instruction"] and "C03_PASS_REQUIRES_C02_PASS" in body["system_instruction"]


def test_response_schema_uses_only_gemini_supported_keywords(builder):
    text = json.dumps(builder.response_schema)
    for keyword in ('"pattern"', '"const"', '"oneOf"', '"uniqueItems"', '"$ref"', '"allOf"', '"not"', '"$defs"'):
        assert keyword not in text, keyword


# ============================================================================== response rules
def test_every_call_accepts_its_synthetic_report(builder, validator):
    for call_id in CALL_IDS:
        context = builder.context(call_id)
        assert validator.errors(synthesize_response(context), context) == [], call_id


def _errors(validator, ctx, mutate):
    report = synthesize_response(ctx)
    mutate(report)
    return validator.errors(report, ctx)


def _check(report, question_id):
    return next(c for c in report["checks"] if c["question_id"] == question_id)


@pytest.mark.parametrize("mutate, expected", [
    (lambda r: r["checks"].reverse(), "checklist order"),
    (lambda r: r["checks"].pop(), "schema"),
    (lambda r: _check(r, "C01").update(issue_ids=["I1"]), "unknown issue"),
    (lambda r: _check(r, "C01").update(assessment="MISMATCH"), "needs at least one linked issue"),
    (lambda r: _check(r, "C01").update(rationale="   "), "rationale is empty"),
    (lambda r: _check(r, "C01").update(rationale="x" * 701), "rationale longer"),
    (lambda r: _check(r, "C01").update(evidence=[]), "must quote one of its own evidence events"),
    (lambda r: _check(r, "C01").update(evidence=[{"source": "G4", "ref": "EV_001", "quote": None}]), "must quote one of its own"),
    (lambda r: _check(r, "C01").update(evidence=[{"source": "G4", "ref": "EV_999", "quote": "x"}]), "is not an event of this plan"),
    (lambda r: _check(r, "C01").update(evidence=[{"source": "G4", "ref": "EV_001", "quote": "words that do not appear"}]),
     "not a substring"),
    (lambda r: _check(r, "C01").update(evidence=[{"source": "G4", "ref": "EV_001", "quote": " "}]), "empty or whitespace-only"),
    (lambda r: _check(r, "C04").update(evidence=[{"source": "G3", "ref": "/gold/C04", "quote": None}]), "under /products or /scenario"),
    (lambda r: _check(r, "C04").update(evidence=[{"source": "G3", "ref": "/scenario/missing", "quote": None}]), "does not resolve"),
    (lambda r: _check(r, "C04").update(evidence=[{"source": "G3", "ref": "/products/0", "quote": "x"}]), "points at a text value"),
])
def test_rule_violations_are_rejected(validator, ctx, mutate, expected):
    errors = _errors(validator, ctx, mutate)
    assert any(expected in e for e in errors), errors


def test_issue_linkage_and_severity_rules(validator, ctx):
    quote = ctx.planned_text("EV_001")[:30]
    base = mismatch_report(ctx, "C05", "EV_001", quote)
    assert validator.errors(base, ctx) == []

    one_way = copy.deepcopy(base)
    _check(one_way, "C05").update(assessment="MATCH", issue_ids=[])
    assert any("does not reference it" in e for e in validator.errors(one_way, ctx))

    severity = copy.deepcopy(base)
    severity["issues"][0]["assessment"] = "UNCLEAR"
    assert any("most severe linked issue" in e for e in validator.errors(severity, ctx))

    duplicate = copy.deepcopy(base)
    duplicate["issues"].append(copy.deepcopy(duplicate["issues"][0]))
    assert any("distinct" in e for e in validator.errors(duplicate, ctx))

    badly_named = copy.deepcopy(base)
    badly_named["issues"][0]["issue_id"] = "issue-1"
    _check(badly_named, "C05")["issue_ids"] = ["issue-1"]
    assert any("must look like I1" in e for e in validator.errors(badly_named, ctx))

    foreign = copy.deepcopy(base)
    foreign["issues"][0]["assignment_ids"] = ["EA_999"]
    assert any("not assignments of this call" in e for e in validator.errors(foreign, ctx))

    whole_plan = synthesize_response(ctx)  # a whole-plan issue needs no checklist question
    whole_plan["issues"] = [{"issue_id": "I1", "assessment": "UNCLEAR", "code": "PLAN_COHERENCE", "question_ids": [],
                             "assignment_ids": [], "evidence": [], "explanation": "A named dependency is unresolved.", "repair_scope": "REVIEW"}]
    assert validator.errors(whole_plan, ctx) == []


def test_absence_findings_may_have_empty_evidence(validator, ctx):
    report = synthesize_response(ctx)
    assert _check(report, "C05")["evidence"] == []
    assert validator.errors(report, ctx) == []


def test_expansion_stores_exact_source_substrings_and_artifact_pointers(ctx):
    text = ctx.planned_text("EV_015")
    report = mismatch_report(ctx, "T05", "EV_015", text.replace("’", "'")[:40])
    checks, issues, errors = expand(report, ctx)
    assert errors == []
    evidence = issues[0]["evidence"][0]
    assert evidence["quote"] == text[:40].strip() and evidence["event_id"] == "EV_015"  # the quote's ends are trimmed
    assert evidence["pointer"] == f"/planned_events/{ctx.event_index['EV_015']}/authoring/planned_text"
    g3 = synthesize_response(ctx)
    _check(g3, "C04")["evidence"] = [{"source": "G3", "ref": "/scenario/client/name", "quote": ctx.g3["scenario"]["client"]["name"]}]
    checks, _, errors = expand(g3, ctx)
    assert errors == [] and next(c for c in checks if c["question_id"] == "C04")["evidence"][0]["pointer"] == "/scenario/client/name"


# ============================================================================== verdict
def test_verdict_precedence_and_derived_repair():
    match = {"assessment": "MATCH"}
    assert derive_verdict([match] * 20, []) == "VALID"
    assert derive_verdict([match] * 19 + [{"assessment": "UNCLEAR"}], [{"assessment": "UNCLEAR"}]) == "UNCLEAR"
    assert derive_verdict([match] * 20, [{"assessment": "MISMATCH"}]) == "INVALID"  # a whole-plan MISMATCH alone invalidates
    assert derive_verdict([{"assessment": "UNCLEAR"}] + [match] * 19, [{"assessment": "MISMATCH"}, {"assessment": "UNCLEAR"}]) == "INVALID"
    issues = [{"assessment": "MISMATCH", "repair_scope": "G4_ONLY"}, {"assessment": "MISMATCH", "repair_scope": "G3_PLUS_G4"}]
    assert derive_repair("INVALID", issues) == "G3_PLUS_G4"
    assert derive_repair("INVALID", [{"assessment": "MISMATCH", "repair_scope": "REVIEW"}, issues[0]]) == "REVIEW"
    assert derive_repair("INVALID", [issues[0], {"assessment": "UNCLEAR", "repair_scope": "G3_PLUS_G4"}]) == "G4_ONLY"
    assert derive_repair("UNCLEAR", issues) == "REVIEW" and derive_repair("VALID", []) is None


# ============================================================================== provider parsing and cost
def test_parse_interaction_outcomes():
    ok = interaction_body('{"checks": [], "issues": []}')
    assert parse_interaction(ok) == ("ok", {"checks": [], "issues": []}, None)
    no_text = dict(ok, output_text=None)
    assert parse_interaction(no_text)[0] == "ok"  # falls back to the model_output text steps
    assert parse_interaction(dict(ok, status="incomplete"))[0] == "rejected"
    assert parse_interaction(dict(ok, status="budget_exceeded"))[0] == "rejected"
    assert parse_interaction(dict(ok, status="requires_action"))[0] == "rejected"
    assert parse_interaction(dict(ok, status="failed"))[0] == "transport"
    assert parse_interaction(dict(ok, steps=[{"type": "function_call"}]))[0] == "rejected"
    assert parse_interaction(interaction_body("not json"))[0] == "rejected"
    assert parse_interaction(interaction_body("[1]"))[0] == "rejected"
    missing_tier = interaction_body('{"checks": [], "issues": []}', service_tier=None)
    assert "service_tier" not in missing_tier and parse_interaction(missing_tier)[0] == "ok"
    kind, _, reason = parse_interaction(interaction_body("{}", service_tier="standard"))
    assert kind == "fatal" and reason.startswith("TIER_MISMATCH")
    kind, _, reason = parse_interaction(interaction_body("{}", model="gemini-3.6-flash"))
    assert kind == "fatal" and reason.startswith("MODEL_MISMATCH")


def test_usage_and_cost_bill_thinking_as_output():
    usage = usage_summary({"total_input_tokens": 1_000_000, "total_cached_tokens": 400_000, "total_output_tokens": 100_000,
                           "total_thought_tokens": 300_000, "total_tokens": 1_400_000})
    expected = (600_000 * 0.375 + 400_000 * 0.0375 + 400_000 * 1.875) / 1_000_000
    assert estimate_cost(usage) == round(expected, 6)
    assert estimate_cost(usage, tier="standard") is None


def test_schemas_are_valid_and_the_artifact_schema_accepts_a_compiled_report(builder, ctx):
    from jsonschema import Draft202012Validator

    from fyp_evaluation.plan_verifications.compile import compile_artifact

    for name in ("plan_verification_response", "plan_verification", "plan_verification_provenance", "plan_verification_record"):
        Draft202012Validator.check_schema(parse_json_text((SCHEMAS / "stage5" / f"{name}.schema.json").read_text(encoding="utf-8")))
    generation = {"attempt_id": "A001", "model": config.MODEL, "params": {"thinking_level": "medium", "service_tier": "flex",
                  "max_output_tokens": 32768}, "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
    report = mismatch_report(ctx, "C05", "EV_001", ctx.planned_text("EV_001")[:30])
    artifact = compile_artifact(shared_inputs(), ctx, report, generation, builder.response_schema)
    assert list(Draft202012Validator(builder.artifact_schema).iter_errors(artifact)) == []
    assert artifact["verdict"] == "INVALID" and artifact["recommended_repair"] == "G4_ONLY"
    assert artifact["summary"] == {"match": 19, "mismatch": 1, "unclear": 0, "issues_by_code": {"OMISSION_LEAKAGE": 1}}
    assert artifact["checks"][4]["expected"] == {"kind": "compliance", "expected_verdict": "FAIL"}


# ============================================================================== regression: Gemini HTTP 400 (nested array bounds)
def _bounds(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("minItems", "maxItems"):
                yield path, key, value
            yield from _bounds(value, f"{path}/{key}")


def test_provider_schema_has_array_bounds_only_on_the_top_level_checks_array(builder):
    """Real-API finding (diagnostics D05-D09, google-genai 2.25, gemini-3.8-flash on Flex): the provider answers HTTP 400
    'Request contains an invalid argument.' as soon as bounds are also set on arrays nested inside the bounded `checks`
    array (issues, evidence). Bounds on `checks` alone were accepted. Those limits are enforced by Python instead."""
    assert sorted(_bounds(builder.response_schema)) == [("/properties/checks", "maxItems", 20), ("/properties/checks", "minItems", 20)]


def test_python_enforces_the_limits_that_are_no_longer_in_the_provider_schema(validator, ctx):
    seven = [{"source": "G4", "ref": "EV_001", "quote": None}] * 7
    assert any("more than 6 evidence items" in e for e in _errors(validator, ctx, lambda r: _check(r, "C01").update(evidence=seven)))
    issue = {"issue_id": "I1", "assessment": "UNCLEAR", "code": "PLAN_COHERENCE", "question_ids": [], "assignment_ids": [], "evidence": seven,
             "explanation": "x", "repair_scope": "REVIEW"}
    assert any("more than 6 evidence items" in e for e in _errors(validator, ctx, lambda r: r["issues"].append(issue)))
    many = [dict(issue, issue_id=f"I{n}", evidence=[]) for n in range(1, 22)]
    assert any("more than 20 issues" in e for e in _errors(validator, ctx, lambda r: r["issues"].extend(many)))
    assert any("schema" in e for e in _errors(validator, ctx, lambda r: r["checks"].pop()))  # exactly 20 checks stays a schema rule
    assert any("schema" in e for e in _errors(validator, ctx, lambda r: r["checks"].append(copy.deepcopy(r["checks"][0]))))
