"""Phase 3A-3/3A-4: G5 v2 (request, report validation, grounding, controls) and the offline G3 -> G4 -> G5 chain on the
multilingual release. Fake transports only; nothing here reads a credential or sends a request."""

import json

import pytest
from jsonschema import Draft202012Validator

import g5_support
import multilingual_support as M
from fyp_evaluation import register_guide
from fyp_evaluation.plan_verifications import controls
from fyp_evaluation.plan_verifications.grounding import ground
from fyp_evaluation.plan_verifications.inputs import load_inputs as load_g5_inputs
from fyp_evaluation.plan_verifications.request import RequestBuilder
from fyp_evaluation.plan_verifications.run import synthesize_response
from fyp_evaluation.plan_verifications.validate_response import VerificationValidator, expand

CALLS = {profile: M.first_call(profile) for profile in M.PROFILES}
PROMPTS = M.RELEASE.parents[1] / "prompts" / "stage5"


@pytest.fixture(scope="module")
def builder():
    return RequestBuilder(M.verification_inputs(list(CALLS.values())), M.SCHEMAS)


# ============================================================================================ the v2 request
@pytest.mark.parametrize("profile", M.PROFILES)
def test_the_g5_brief_carries_the_language_block_and_mention_languages(builder, profile):
    brief = json.loads(builder.brief_text(CALLS[profile]))
    assert brief["call"]["language"]["language_profile"] == profile
    assert all("language_variant" in e for e in brief["edge_cases"])
    mentions = [m for e in brief["events"] for m in e["value_mentions"]]
    assert mentions and all(m["language"] in ("en", "zh") for m in mentions)
    assert ("register_guide" in brief) is (profile == "SINGLISH")


def test_the_g5_system_prompt_and_response_schema_are_the_v2_generation(builder):
    body = builder.build(CALLS["MANDARIN"], "A001")
    assert "THE LANGUAGE OF THE CALL" in body["system_instruction"] and "LANGUAGE_REQUIREMENT_NOT_MET" in body["system_instruction"]
    assert "{{CHECKLIST}}" not in body["system_instruction"] and "{{SEMANTICS}}" not in body["system_instruction"]
    assert "LANGUAGE_REQUIREMENT_NOT_MET" in json.dumps(body["response_format"]["schema"])
    pilot = RequestBuilder(g5_support.shared_inputs(), M.SCHEMAS).build("CALL_001", "A001")
    assert "THE LANGUAGE OF THE CALL" not in pilot["system_instruction"]
    assert "LANGUAGE_REQUIREMENT_NOT_MET" not in json.dumps(pilot["response_format"])


def test_the_v2_prompt_keeps_every_v1_line_but_three_and_adds_the_language_section():
    v1 = (PROMPTS / "plan_verification.v1.md").read_text(encoding="utf-8").splitlines()
    v2 = (PROMPTS / "plan_verification.v2.md").read_text(encoding="utf-8").splitlines()
    assert len([line for line in v1 if line not in v2]) == 3  # the mention note, the issue-code list and the quote rule
    assert "THE LANGUAGE OF THE CALL" in v2 and "THE LANGUAGE OF THE CALL" not in v1


def test_all_250_multilingual_plans_yield_valid_v2_reports_and_briefs():
    inputs = M.verification_inputs(list(M.upstream_inputs().call_ids))
    builder = RequestBuilder(inputs, M.SCHEMAS)
    validator = VerificationValidator(builder.response_schema)
    for call_id in inputs.call_ids:
        ctx = builder.context(call_id)
        assert not validator.errors(synthesize_response(ctx), ctx), call_id
        assert builder.brief_text(call_id)
    assert len(inputs.call_ids) == 250


def language_issue_report(ctx, event_id, quote):
    report = synthesize_response(ctx)
    report["issues"] = [{"issue_id": "I1", "assessment": "MISMATCH", "code": "LANGUAGE_REQUIREMENT_NOT_MET", "question_ids": [],
                         "assignment_ids": [], "evidence": [{"source": "G4", "ref": event_id, "quote": quote}],
                         "explanation": "The employee wording is not professional English.", "repair_scope": "G4_ONLY"}]
    return report


def test_the_language_issue_code_exists_only_in_the_v2_schemas(builder):
    ctx = builder.context(CALLS["SINGLISH"])
    event = next(e for e in ctx.events if e["authoring"]["speaker"] == "EMPLOYEE")
    report = language_issue_report(ctx, event["event_id"], event["authoring"]["planned_text"][:20])
    assert VerificationValidator(builder.response_schema).errors(report, ctx) == []
    pilot_schema = json.loads((M.SCHEMAS / "stage5/plan_verification_response.schema.json").read_text(encoding="utf-8"))
    assert list(Draft202012Validator(pilot_schema).iter_errors(report))


# ============================================================================================ grounding of Chinese quotes
ZH = "您好，请问您想通过这笔投资实现什么目标？我想稳健增值，不想冒太大风险。"


def test_a_chinese_quote_grounds_to_its_exact_source_substring():
    assert ground("稳健增值，不想冒太大风险", ZH) == ("稳健增值，不想冒太大风险", None)
    assert ground("  您好，请问  ", ZH) == ("您好，请问", None)  # the ends are trimmed, nothing else


def test_a_chinese_quote_that_is_not_in_the_text_is_rejected():
    assert ground("我想保本，不想亏损", ZH)[0] is None
    assert ground("稳健 增值", ZH)[0] is None, "a space is never invented inside Han text"
    assert ground("您好,请问", ZH)[0] is None, "full-width punctuation is not folded to ASCII"
    assert ground("", ZH)[1] and ground("   ", ZH)[1]


def test_an_ambiguous_quote_is_rejected_even_next_to_han_text():
    source = "我是“好”人，我是\"好\"人。"
    exact, reason = ground("我是\"好\"人", source)
    assert exact is None and "ambiguous" in reason
    assert ground("人，我是", source) == ("人，我是", None)


def test_mixed_script_quotes_keep_both_languages_verbatim():
    text = "好的，I want to buy 这只基金，谢谢。"
    assert ground("I want to buy 这只基金", text) == ("I want to buy 这只基金", None)
    assert ground("I  want to buy   这只基金", text) == ("I want to buy 这只基金", None)  # whitespace runs collapse


def test_a_quote_is_checked_against_the_cited_event_never_the_target(builder):
    ctx = builder.context(CALLS["MANDARIN"])
    event_id = ctx.checks["C01"]["evidence_event_ids"][0]
    parsed = synthesize_response(ctx)
    parsed["checks"][0]["evidence"] = [{"source": "G4", "ref": event_id, "quote": "客户确认了投资目标"}]
    assert any("not a substring" in e for e in expand(parsed, ctx)[2])


# ============================================================================================ the chain and the semantic controls
def test_the_chain_accepts_one_call_of_each_profile_offline(tmp_path):
    chain = M.chain(tmp_path, list(CALLS.values()))
    assert set(chain.run_g3().values()) == {"ACCEPTED"}
    assert set(chain.run_g4().values()) == {"ACCEPTED"}
    assert set(chain.run_g5().values()) == {"ACCEPTED"}
    verified = chain.root / "authoring/plan_verifications"
    singlish = json.loads((verified / f"{CALLS['SINGLISH']}.json").read_text(encoding="utf-8"))
    assert singlish["schema_version"] == "2.0" and singlish["source"]["register_guide_sha256"] == register_guide.guide_sha256()
    mandarin = json.loads((verified / f"{CALLS['MANDARIN']}.json").read_text(encoding="utf-8"))
    assert "register_guide_sha256" not in mandarin["source"] and mandarin["source"]["language_profiles_sha256"]


def _plan_events(call_id):
    return json.loads(M.plan_texts([call_id])[call_id])["planned_events"]


def _find(profile, wanted):
    """The first call of `profile` whose artificial plan has an event for which wanted(event, spec) holds."""
    specs = M.plan_inputs().specs
    for call_id in M.calls_of(profile):
        if any(wanted(e, specs[call_id]) for e in _plan_events(call_id)):
            return call_id
    raise AssertionError(f"no {profile} call has the event")


def _c05_fails(spec):
    return next(g for g in spec["gold"] if g["question_id"] == "C05")["expected"]["expected_verdict"] == "FAIL"


def _c01_employee_act(event):
    return event["kind"] == "COMPLIANCE_ACT" and event["question_ids"] == ["C01"] and event["authoring"]["speaker"] == "EMPLOYEE"


def _control_calls():
    calls = {}
    for profile in M.PROFILES:
        calls[("M_omission_leak", profile)] = _find(profile, lambda e, s: _c01_employee_act(e) and _c05_fails(s))
        calls[("M_negated_direction", profile)] = _find(profile, lambda e, s: e["kind"] == "ORDER_STATEMENT" and e["status"] == "FINAL")
        calls[("M_omitted_negation", profile)] = _find(profile, lambda e, s: e["kind"] == "GUARANTEE_REFUSAL")
    calls[("S_employee_singlish", "SINGLISH")] = calls[("M_omission_leak", "SINGLISH")]
    return calls


@pytest.fixture(scope="module")
def control_chain(tmp_path_factory):
    calls = _control_calls()
    tmp = tmp_path_factory.mktemp("controls")
    chain = M.chain(tmp, sorted(set(calls.values())))
    assert set(chain.run_g3().values()) == {"ACCEPTED"} and set(chain.run_g4().values()) == {"ACCEPTED"}
    return chain, calls, tmp


def _planned_texts(root, call_id):
    plan = json.loads((root / "authoring/check_plans" / f"{call_id}.json").read_text(encoding="utf-8"))
    return {e["event_id"]: e["authoring"]["planned_text"] for e in plan["planned_events"]}


def test_every_multilingual_control_is_accepted_by_g4_and_reaches_the_g5_brief(control_chain):
    chain, calls, tmp = control_chain
    assert set(controls.MULTILINGUAL_CONTROLS) == {name for name, _ in calls}
    for index, ((name, profile), call_id) in enumerate(sorted(calls.items())):
        destination = tmp / f"control_{index}"
        controls.build_multilingual_control(chain.root, destination, name, call_id, profile, schema_dir=M.SCHEMAS,
                                            audit_root=tmp / f"audit_{index}", selected=chain.call_ids)
        before, after = _planned_texts(chain.root, call_id), _planned_texts(destination, call_id)
        changed = [(old, after[event_id]) for event_id, old in before.items() if after[event_id] != old]
        assert len(changed) == 1, (name, profile, changed)
        old, new = changed[0]
        expected = {"M_omission_leak": new.startswith(old), "S_employee_singlish": new.endswith(controls.SINGLISH_EMPLOYEE),
                    "M_negated_direction": new.endswith(old), "M_omitted_negation": new == controls.GUARANTEE[profile]}[name]
        assert expected, (name, profile, old, new)
        brief = RequestBuilder(load_g5_inputs(destination, M.SCHEMAS, [call_id]), M.SCHEMAS).brief_text(call_id)
        assert json.dumps(new, ensure_ascii=False)[1:-1] in brief


def test_a_control_must_apply_to_its_profile_and_name_a_known_control(control_chain):
    chain, calls, tmp = control_chain
    mandarin = calls[("M_omission_leak", "MANDARIN")]
    with pytest.raises(ValueError, match="does not apply"):
        controls.build_multilingual_control(chain.root, tmp / "x1", "S_employee_singlish", mandarin, "MANDARIN", schema_dir=M.SCHEMAS)
    with pytest.raises(ValueError, match="unknown multilingual control"):
        controls.build_multilingual_control(chain.root, tmp / "x2", "nope", mandarin, "MANDARIN", schema_dir=M.SCHEMAS)


def test_a_language_requirement_report_on_the_singlish_employee_control_validates_and_expands(control_chain):
    chain, calls, tmp = control_chain
    call_id = calls[("S_employee_singlish", "SINGLISH")]
    destination = tmp / "control_singlish"
    controls.build_multilingual_control(chain.root, destination, "S_employee_singlish", call_id, "SINGLISH", schema_dir=M.SCHEMAS,
                                        audit_root=tmp / "audit_s", selected=chain.call_ids)
    builder = RequestBuilder(load_g5_inputs(destination, M.SCHEMAS, [call_id]), M.SCHEMAS)
    ctx = builder.context(call_id)
    event_id = next(i for i, text in _planned_texts(destination, call_id).items() if text.endswith(controls.SINGLISH_EMPLOYEE))
    report = language_issue_report(ctx, event_id, "Can lah, no problem")
    assert VerificationValidator(builder.response_schema).errors(report, ctx) == []
    _, issues, errors = expand(report, ctx)
    assert not errors and issues[0]["code"] == "LANGUAGE_REQUIREMENT_NOT_MET" and issues[0]["evidence"][0]["quote"] == "Can lah, no problem"
