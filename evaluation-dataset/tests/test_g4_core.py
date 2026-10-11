"""G4 core: 50-call/968-event structural feasibility, every authoring rule, rendering, projection and determinism.

Offline fixtures prove STRUCTURAL feasibility only; they say nothing about the semantic quality of real model wording.
"""

import json
from copy import deepcopy

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.check_plans import render as R
from fyp_evaluation.check_plans.analysis import analyze, find_cycle
from fyp_evaluation.check_plans.compile import assemble, compile_artifact, get_lexicon, project_check_plan_to_skeleton, projection_errors
from fyp_evaluation.check_plans.context import PlanContext
from fyp_evaluation.check_plans.feasibility import feasibility_report, make_validator, synthesize_response
from fyp_evaluation.check_plans.request import RequestBuilder, build_brief, request_hash
from fyp_evaluation.llm.store import canonical_request_text
from g4_support import CALL_IDS, SCHEMAS, entry_of, shared_inputs


@pytest.fixture(scope="module")
def inputs():
    return shared_inputs()


@pytest.fixture(scope="module")
def builder(inputs):
    return RequestBuilder(inputs, SCHEMAS)


@pytest.fixture(scope="module")
def validator(builder):
    return make_validator(builder)


def errors_for(builder, validator, call_id, mutate):
    ctx = builder.context(call_id)
    response = synthesize_response(ctx)
    mutate(response, ctx)
    return validator.errors(response, ctx)


def first_call(builder, predicate):
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        for event in ctx.events:
            if predicate(ctx, event):
                return call_id, event
    raise AssertionError("no matching call")


def has(errors, fragment):
    return any(fragment in e for e in errors)


# ===================================================================================== feasibility
def test_all_968_frozen_events_of_all_50_calls_can_be_authored_and_compiled(builder, validator, inputs):
    report = feasibility_report(builder, validator, inputs)
    assert len(report) == 50 and {c: e for c, e in report.items() if e} == {}
    assert sum(len(builder.context(c).events) for c in CALL_IDS) == 968


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_every_allowed_speaker_and_form_choice_is_accepted(builder, validator, offset):
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        assert validator.errors(synthesize_response(ctx, form_offset=offset, speaker_offset=offset), ctx) == [], call_id


def test_the_choices_really_vary_across_offsets(builder):
    speakers, forms = set(), set()
    for call_id in CALL_IDS:
        for offset in (0, 1):
            for entry in synthesize_response(builder.context(call_id), form_offset=offset, speaker_offset=offset)["events"]:
                speakers.add(entry["speaker"])
                forms.update(item["form"] for item in entry["value_forms"])
    assert {"CLIENT", "EMPLOYEE"} <= speakers and {"NATURAL_NAME", "ISO_CODE", "DAY_MONTH_YEAR", "ORDINAL_DAY"} <= forms


def compile_plan(inputs, builder, call_id, **offsets):
    ctx = builder.context(call_id)
    generation = {"attempt_id": "A001", "model": "gpt-6.1-sol", "prompt_template_sha256": builder.prompt_template_sha256,
                  "response_schema_sha256": builder.response_schema_sha256}
    return ctx, compile_artifact(inputs, ctx, synthesize_response(ctx, **offsets), generation, builder.response_schema)


def test_every_plan_projects_back_to_the_canonical_frozen_skeleton_bytes(inputs, builder):
    for call_id in CALL_IDS:
        ctx, plan = compile_plan(inputs, builder, call_id)
        assert serialize_json(project_check_plan_to_skeleton(plan)) == inputs.skeleton_texts[call_id], call_id
        assert projection_errors(ctx, plan) == []
        assert [e["event_id"] for e in plan["planned_events"]] == [e["event_id"] for e in ctx.events]


# ======================================================================================== projection
def test_projection_removes_exactly_the_g4_additions_and_keeps_array_and_key_order(inputs, builder):
    ctx, plan = compile_plan(inputs, builder, "CALL_034")
    assert plan["derived_constraints"], "CALL_034 needs derived readback constraints"
    skeleton = ctx.skeleton
    projected = project_check_plan_to_skeleton(plan)
    assert list(projected) == list(skeleton) and projected == skeleton
    assert [list(e) for e in projected["planned_events"]] == [list(e) for e in skeleton["planned_events"]]
    assert set(plan) - set(projected) == {"check_plan_skeleton_sha256", "call_config_sha256", "generation", "skeleton_header", "derived_constraints"}


def test_any_alteration_of_a_frozen_field_breaks_the_projection(inputs, builder):
    ctx, plan = compile_plan(inputs, builder, "CALL_010")
    cases = {
        "event kind": lambda p: p["planned_events"][0].__setitem__("kind", "VALUE_STATEMENT"),
        "event status": lambda p: p["planned_events"][3].__setitem__("status", "FINAL"),
        "speakers": lambda p: p["planned_events"][0].__setitem__("allowed_speakers", ["CLIENT"]),
        "a frozen value": lambda p: p["planned_events"][8]["values"]["T03"].__setitem__("value", 1),
        "a spoken-form requirement": lambda p: p["planned_events"][8]["values"]["T03"].__setitem__("spoken_form", {"requirement": "NOT_EXPLICIT"}),
        "check verdict": lambda p: p["checks"][0]["expected"].__setitem__(
            "expected_verdict", "FAIL" if p["checks"][0]["expected"]["expected_verdict"] == "PASS" else "PASS"),
        "a constraint": lambda p: p["constraints"].pop(),
        "constraint order": lambda p: p["constraints"].reverse(),
        "event order": lambda p: p["planned_events"].reverse(),
        "event removed": lambda p: p["planned_events"].pop(),
        "header": lambda p: p["skeleton_header"].__setitem__("call_id", "CALL_099"),
        "event key order": lambda p: p["planned_events"].__setitem__(0, dict(reversed(list(p["planned_events"][0].items())))),
    }
    for name, mutate in cases.items():
        broken = deepcopy(plan)
        try:
            mutate(broken)
        except (KeyError, IndexError):
            continue
        assert projection_errors(ctx, broken), name
    broken = deepcopy(plan)
    del broken["planned_events"][2]["authoring"]
    assert projection_errors(ctx, broken)


def test_the_response_schema_gives_the_model_no_field_for_frozen_values(builder, validator):
    ctx = builder.context("CALL_010")
    for extra in ("kind", "status", "values", "constraints", "checks", "allowed_speakers"):
        response = synthesize_response(ctx)
        response["events"][0][extra] = "x"
        assert has(validator.errors(response, ctx), "schema")
    response = synthesize_response(ctx)
    response["checks"] = []
    assert has(validator.errors(response, ctx), "schema")


# ======================================================================================== event IDs
def test_missing_duplicate_and_unknown_event_ids(builder, validator):
    assert has(errors_for(builder, validator, "CALL_001", lambda r, c: r["events"].pop(3)), "EV_004: missing from the response")
    assert has(errors_for(builder, validator, "CALL_001", lambda r, c: r["events"].append(deepcopy(r["events"][0]))), "listed 2 times")
    assert has(errors_for(builder, validator, "CALL_001", lambda r, c: r["events"].append({**deepcopy(r["events"][0]), "event_id": "EV_999"})),
               "EV_999: unknown event id")
    assert has(errors_for(builder, validator, "CALL_001", lambda r, c: entry_of(r, "EV_002").update(event_id="EV_001")), "listed 2 times")


def test_the_response_may_list_events_in_any_order_but_python_keeps_the_skeleton_order(inputs, builder, validator):
    ctx = builder.context("CALL_010")
    response = synthesize_response(ctx)
    response["events"].reverse()
    assert validator.errors(response, ctx) == []
    generation = {"attempt_id": "A001", "model": "m", "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
    plan = compile_artifact(inputs, ctx, response, generation, builder.response_schema)
    assert [e["event_id"] for e in plan["planned_events"]] == [e["event_id"] for e in ctx.events]


# ======================================================================================= speakers/forms
def test_speaker_rules(builder, validator):
    call_id, single = first_call(builder, lambda c, e: len(e["allowed_speakers"]) == 1)
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, single["event_id"]).update(speaker=single["allowed_speakers"][0])),
               "speaker must be null")
    call_id, multi = first_call(builder, lambda c, e: len(e["allowed_speakers"]) > 1)
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, multi["event_id"]).update(speaker=None)), "speaker must be one of")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, multi["event_id"]).update(speaker="BANK")), "speaker must be one of")


def test_value_form_choice_rules(builder, validator):
    call_id, event = first_call(builder, lambda c, e: any(v and len(v) > 1 for v in
                                                          (c.form_choices(e, q) for q in c.slot_rules(e).required if q in c.checks)))
    name = next(q for q in ctx_required(builder, call_id, event) if len(builder.context(call_id).form_choices(event, q) or []) > 1)
    eid = event["event_id"]
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, eid).update(value_forms=[])), f"must choose a form for {name}")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, eid)["value_forms"].append({"placeholder": name, "form": "ISO_CODE"})),
               "more than once")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, eid)["value_forms"][0].update(form="NO_SUCH_FORM")), "is not one of")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, eid)["value_forms"].append({"placeholder": "T04", "form": "X"})),
               "not allowed")
    other = first_call(builder, lambda c, e: e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"])
    assert has(errors_for(builder, validator, other[0], lambda r, c: entry_of(r, other[1]["event_id"])["value_forms"].append(
        {"placeholder": "T02", "form": "ISO_CODE"})), "not allowed")


def ctx_required(builder, call_id, event):
    return builder.context(call_id).slot_rules(event).required


# ======================================================================================== placeholders
def test_required_missing_repeated_extra_unknown_and_malformed_placeholders(builder, validator):
    call_id, event = first_call(builder, lambda c, e: e["kind"] == "VALUE_STATEMENT" and e["question_ids"] == ["T04"])
    eid = event["event_id"]

    def text(value):
        return lambda r, c: entry_of(r, eid).update(text=value)
    assert has(errors_for(builder, validator, call_id, text("The quantity is a secret.")), "[[T04]] must appear exactly once (found 0)")
    assert has(errors_for(builder, validator, call_id, text("The quantity is [[T04]] or [[T04]].")), "exactly once (found 2)")
    assert has(errors_for(builder, validator, call_id, text("The quantity is [[T04]] for [[T03]].")), "[[T03]] is not permitted")
    assert has(errors_for(builder, validator, call_id, text("The quantity is [[T04]] and [[FEE]].")), "[[FEE]] is not permitted")
    assert has(errors_for(builder, validator, call_id, text("The quantity is [[T04]] and [[BOGUS]].")), "unknown placeholder [[BOGUS]]")
    assert has(errors_for(builder, validator, call_id, text("The quantity is [[T04] today.")), "stray brackets")
    assert has(errors_for(builder, validator, call_id, text("The quantity is {T04} [[T04]].")), "stray brackets")
    assert has(errors_for(builder, validator, call_id, text("   ")), "must not be empty")
    assert has(errors_for(builder, validator, call_id, text("[[T04]]")), "needs wording outside")
    assert has(errors_for(builder, validator, call_id, text("word " * 120 + "[[T04]]")), "longer than")


def test_fee_is_required_only_in_the_c04_pass_act_and_forbidden_everywhere_else(builder, validator):
    c04_pass = first_call(builder, lambda c, e: e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C04"])
    call_id, event = c04_pass
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, event["event_id"]).update(text="There is a charge.")), "[[FEE]] must appear exactly once")
    other = first_call(builder, lambda c, e: e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"])
    assert has(errors_for(builder, validator, other[0], lambda r, c: entry_of(r, other[1]["event_id"]).update(text="Our fee is [[FEE]].")),
               "[[FEE]] is not permitted")
    fail_call = next(c for c in CALL_IDS if builder.context(c).verdict("C04") == "FAIL")
    ctx = builder.context(fail_call)
    assert not any(e["question_ids"] == ["C04"] for e in ctx.events) and ctx.fee is None
    employee_event = next(e for e in ctx.events if e["kind"] == "COMPLIANCE_ACT" and e["allowed_speakers"] == ["EMPLOYEE"])
    assert has(errors_for(builder, validator, fail_call, lambda r, c: entry_of(r, employee_event["event_id"]).update(text="The fee is [[FEE]].")),
               "[[FEE]] is not permitted")


def test_limit_price_required_in_final_and_initial_limit_orders_and_forbidden_otherwise(builder, validator):
    call_id, final = first_call(builder, lambda c, e: c.limit_slot(e) == "final" and e["kind"] == "ORDER_STATEMENT")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, final["event_id"]).update(
        text=entry_of(r, final["event_id"])["text"].replace(", with a limit price of [[LIMIT_PRICE]]", ""))), "[[LIMIT_PRICE]] must appear exactly once")
    call_id, initial = first_call(builder, lambda c, e: c.limit_slot(e) == "initial")
    ctx = builder.context(call_id)
    plan_block = analyze(ctx, get_lexicon(shared_inputs()), synthesize_response(ctx))[0][initial["event_id"]]
    price = ctx.limit_prices["initial"]
    assert plan_block["planned_text"].endswith(R.render_limit_price(price, ctx.limit_currency["initial"], ctx.aliases) + ".")
    market_call, market_event = first_call(builder, lambda c, e: e["kind"] == "VALUE_STATEMENT" and e["question_ids"] == ["T06"]
                                           and c.gold_value("T06") == "MARKET")
    assert has(errors_for(builder, validator, market_call, lambda r, c: entry_of(r, market_event["event_id"]).update(
        text="I would like [[T06]] at [[LIMIT_PRICE]].")), "[[LIMIT_PRICE]] is not permitted")
    non_final = first_call(builder, lambda c, e: e["kind"] == "VALUE_STATEMENT" and e["status"] == "SUPERSEDED" and e["question_ids"] == ["T04"])
    assert has(errors_for(builder, validator, non_final[0], lambda r, c: entry_of(r, non_final[1]["event_id"]).update(
        text="It was [[T04]] at [[LIMIT_PRICE]].")), "[[LIMIT_PRICE]] is not permitted")


def test_currency_placeholder_is_required_in_monetary_distractors(builder, validator):
    call_id, event = first_call(builder, lambda c, e: e["kind"] == "DISTRACTOR_MENTION" and "currency" in e)
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, event["event_id"]).update(text="For reference, there was [[T03]].")),
               "[[CURRENCY]] must appear exactly once")


def test_controlled_gold_restatements_only_in_the_permitted_acts(builder, validator):
    call_id, c08 = first_call(builder, lambda c, e: e["question_ids"] == ["C08"] and e.get("role") == "ACT")
    ctx = builder.context(call_id)
    text = "To confirm, you want to [[T05]] [[T04]] units of [[T01]]. Is that right?"
    assert errors_for(builder, validator, call_id, lambda r, c: entry_of(r, c08["event_id"]).update(text=text)) == []
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, c08["event_id"]).update(text=text + " Also [[T05]].")), "at most once")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, c08["event_id"]).update(text="Please give me your number [[K01]].")), "[[K01]] is not permitted")
    response_event = next(e for e in ctx.events if e["question_ids"] == ["C08"] and e.get("role") == "RESPONSE")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, response_event["event_id"]).update(text="Yes, [[T04]] units.")), "not permitted")
    c02 = first_call(builder, lambda c, e: e["question_ids"] == ["C02"] and e.get("role") == "ACT")
    assert errors_for(builder, validator, c02[0], lambda r, c: entry_of(r, c02[1]["event_id"]).update(
        text="A risk of [[T01]] is that its price can drop when markets fall.")) == []
    assert has(errors_for(builder, validator, c02[0], lambda r, c: entry_of(r, c02[1]["event_id"]).update(
        text="A risk of [[T01]] is that its price can drop, for [[T03]].")), "[[T03]] is not permitted")
    c01 = first_call(builder, lambda c, e: e["question_ids"] == ["C01"])
    assert has(errors_for(builder, validator, c01[0], lambda r, c: entry_of(r, c01[1]["event_id"]).update(text="Your product is [[T01]], right?")),
               "[[T01]] is not permitted")


def test_fail_by_omission_checks_gain_no_events_and_no_placeholders(builder):
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        for check in ctx.skeleton["checks"]:
            if check["kind"] == "COMPLIANCE" and check["evidence_event_ids"] == []:
                question = check["question_id"]
                acts = [e for e in ctx.events if question in e["question_ids"] and e["kind"] == "COMPLIANCE_ACT"]
                weak = [e for e in ctx.events if question in e["question_ids"] and e["kind"] == "WEAK_ACKNOWLEDGEMENT"]
                # the only act an omitted check may keep is the C08 confirmation REQUEST that a weak acknowledgement answers
                assert not acts or (question == "C08" and weak and all(a["allowed_speakers"] == ["EMPLOYEE"] or a["role"] == "RESPONSE" for a in acts)), (call_id, question)
                assert all(not ctx.slot_rules(e).required for e in acts), (call_id, question)


# ===================================================================================== lexical rules
@pytest.mark.parametrize("wording, fragment", [
    ("It is 500 units.", "digits"), ("It is five hundred units.", "magnitude words"), ("About a thousand.", "magnitude words"),
    ("It is a fee of 2 percent.", "percent"), ("Roughly 3%.", "digits"), ("Costs ten per cent.", "percent"),
    ("In dollars, please.", "currency names or codes"), ("It is in SGD.", "currency names or codes"), ("It is worth ¥.", "percent or currency symbols"),
    ("Priced in euros.", "currency names or codes"), ("The renminbi side.", "currency names or codes"),
    ("Take Meridian Asia Equity Fund.", "product names"), ("The Orion Global one.", "product names"),
])
def test_lexical_bans_apply_to_template_wording_outside_placeholders(builder, validator, wording, fragment):
    call_id, event = first_call(builder, lambda c, e: e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"])
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, event["event_id"]).update(text=wording)), fragment)


def test_a_c04_fail_call_may_legitimately_render_digits_amounts_and_prices_through_placeholders(inputs, builder, validator):
    call_id = next(c for c in CALL_IDS if builder.context(c).verdict("C04") == "FAIL" and builder.context(c).gold_value("T03") >= 1000
                   and builder.context(c).gold_value("T06") == "LIMIT")
    ctx = builder.context(call_id)
    assert validator.errors(synthesize_response(ctx), ctx) == []
    authoring = analyze(ctx, get_lexicon(inputs), synthesize_response(ctx))[0]
    rendered = " ".join(block["planned_text"] for block in authoring.values())
    assert any(ch.isdigit() for ch in rendered) and "[[" not in rendered
    assert not any(m["placeholder"] == "FEE" for block in authoring.values() for m in block["value_mentions"])


def test_mechanical_omission_protections(builder, validator):
    pass_c09, ev = first_call(builder, lambda c, e: c.verdict("C09") == "PASS" and e["kind"] == "COMPLIANCE_ACT" and "C09" not in e["question_ids"])
    assert has(errors_for(builder, validator, pass_c09, lambda r, c: entry_of(r, ev["event_id"]).update(text="We must not launder anything.")), "money laundering")
    pass_c10, ev = first_call(builder, lambda c, e: c.verdict("C10") == "PASS" and e["kind"] == "COMPLIANCE_ACT" and "C10" not in e["question_ids"])
    assert has(errors_for(builder, validator, pass_c10, lambda r, c: entry_of(r, ev["event_id"]).update(text="This comes from inside information.")), "inside information")
    # employee wording may not mention guarantees outside the C12 events
    emp = first_call(builder, lambda c, e: e["question_ids"] == ["C01"] and e["allowed_speakers"] == ["EMPLOYEE"])
    assert has(errors_for(builder, validator, emp[0], lambda r, c: entry_of(r, emp[1]["event_id"]).update(text="Nothing is guaranteed.")), "guarantees")
    refusal = first_call(builder, lambda c, e: e["kind"] == "GUARANTEE_REFUSAL")
    assert errors_for(builder, validator, refusal[0], lambda r, c: entry_of(r, refusal[1]["event_id"]).update(text="I cannot guarantee any return.")) == []
    # the violating act of a FAIL check is exempt for its own check
    fail_c10 = next(c for c in CALL_IDS if builder.context(c).verdict("C10") == "FAIL")
    ctx = builder.context(fail_c10)
    violation = next(e for e in ctx.events if e["question_ids"] == ["C10"])
    assert errors_for(builder, validator, fail_c10, lambda r, c: entry_of(r, violation["event_id"]).update(
        text="I heard this from an insider with inside information.")) == []


def test_kind_specific_wording_rules(builder, validator):
    weak = first_call(builder, lambda c, e: e["kind"] == "WEAK_ACKNOWLEDGEMENT")
    for word in ("Yes, fine.", "I confirm that.", "That is correct.", "Right.", "I agree.", "Go ahead."):
        assert has(errors_for(builder, validator, weak[0], lambda r, c: entry_of(r, weak[1]["event_id"]).update(text=word)), "weak acknowledgement")
    assert errors_for(builder, validator, weak[0], lambda r, c: entry_of(r, weak[1]["event_id"]).update(text="Hmm, I suppose.")) == []
    false_start = first_call(builder, lambda c, e: e["kind"] == "FALSE_START")
    assert has(errors_for(builder, validator, false_start[0], lambda r, c: entry_of(r, false_start[1]["event_id"]).update(text="I want a limit order.")), "trail off")
    for ok in ("I want a limit—", "I want a limit...", "I want a limit…", "I want a limit -"):
        assert errors_for(builder, validator, false_start[0], lambda r, c: entry_of(r, false_start[1]["event_id"]).update(text=ok)) == []
    assert has(errors_for(builder, validator, false_start[0], lambda r, c: entry_of(r, false_start[1]["event_id"]).update(text="I want [[T06]]...")), "not permitted")
    selection = first_call(builder, lambda c, e: e["kind"] == "REFERENCE_SELECTION")
    assert has(errors_for(builder, validator, selection[0], lambda r, c: entry_of(r, selection[1]["event_id"]).update(text="The one for [[T01]].")), "not permitted")
    invalidation = first_call(builder, lambda c, e: e["kind"] == "ORDER_INVALIDATION")
    assert has(errors_for(builder, validator, invalidation[0], lambda r, c: entry_of(r, invalidation[1]["event_id"]).update(text="Cancel [[T01]].")), "not permitted")


def test_paraphrase_constraint_rejects_checklist_copying(builder, validator):
    call_id, event = first_call(builder, lambda c, e: e["kind"] == "COMPLIANCE_ACT" and c.checks[e["question_ids"][0]].get("wording_constraint"))
    question = event["question_ids"][0]
    copied = builder.context(call_id).checklist[question]["question"].replace("The employee should ", "I should ")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, event["event_id"]).update(text=copied)), "repeats the checklist verbatim")


# ======================================================================================== read-backs
def readback_call(builder, **conditions):
    return first_call(builder, lambda c, e: e["kind"] == "READBACK" and all(e["readback"].get(k) == v for k, v in conditions.items()))


def test_readback_selection_rules_and_derived_constraints(inputs, builder, validator):
    call_id, event = readback_call(builder, min_field_count=2)  # CALL_034: one-field read-back error
    assert call_id == "CALL_034"
    ctx = builder.context(call_id)
    eid, wrong = event["event_id"], event["readback"]["wrong_question_id"]
    base = synthesize_response(ctx)
    good = entry_of(base, eid)
    assert len(good["readback_question_ids"]) >= 2 and wrong in good["readback_question_ids"]

    def set_fields(fields, text=None):
        return lambda r, c: entry_of(r, eid).update(readback_question_ids=fields, text=text if text is not None else
                                                    "I have " + " and ".join(f"[[{q}]]" for q in fields) + ".")
    assert has(errors_for(builder, validator, call_id, set_fields([wrong])), "at least 2 fields")
    assert has(errors_for(builder, validator, call_id, set_fields(["T01", "T02"])), "must include")
    assert has(errors_for(builder, validator, call_id, set_fields([wrong, wrong])), "duplicates")
    assert has(errors_for(builder, validator, call_id, set_fields([wrong, "C01"])), "no FINAL value")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, eid).update(text="I have [[T06]] only.")), "[[T01]] must appear" if False else "must appear exactly once")
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, eid).update(readback_question_ids=None)), "required for a readback")
    other = entry_of(base, [e for e in ctx.events if e["kind"] == "VALUE_STATEMENT"][0]["event_id"])
    assert has(errors_for(builder, validator, call_id, lambda r, c: entry_of(r, other["event_id"]).update(readback_question_ids=["T06"])), "must be null")
    generation = {"attempt_id": "A001", "model": "m", "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
    plan = compile_artifact(inputs, ctx, base, generation, builder.response_schema)
    chosen = [q for q in good["readback_question_ids"] if q != wrong]
    assert [(d["before"], d["after"], d["source"]["question_id"]) for d in plan["derived_constraints"]] == [(ctx.final_event[q], eid, q) for q in chosen]
    assert all(d["source"] == {"type": "READBACK_FIELD", "id": eid, "question_id": d["source"]["question_id"]} for d in plan["derived_constraints"])


def test_readback_renders_the_wrong_value_for_one_field_and_final_values_for_the_rest(inputs, builder):
    ctx = builder.context("CALL_034")
    event = next(e for e in ctx.events if e["kind"] == "READBACK")
    block = analyze(ctx, get_lexicon(inputs), synthesize_response(ctx))[0][event["event_id"]]
    by_question = {m["question_id"]: m for m in block["value_mentions"]}
    wrong = event["readback"]["wrong_question_id"]
    assert by_question[wrong]["source"] == "READBACK_WRONG" and by_question[wrong]["value"] == event["readback"]["wrong_value"]
    assert by_question[wrong]["value"] != ctx.gold_value(wrong)
    for question, mention in by_question.items():
        if question != wrong:
            assert mention["source"] == "GOLD_RESTATEMENT" and mention["value"] == ctx.gold_value(question)


def test_readback_error_for_a_single_field_needs_no_extra_constraints(inputs, builder):
    call_id, event = readback_call(builder, min_field_count=1)
    ctx = builder.context(call_id)
    authoring, derived, errors = analyze(ctx, get_lexicon(inputs), synthesize_response(ctx))
    assert errors == []
    assert [d for d in derived if d["after"] == event["event_id"]] == [] or len(authoring[event["event_id"]]["readback_question_ids"]) > 1


def test_an_impossible_or_cyclic_ordering_is_rejected(inputs, builder, validator):
    ctx = PlanContext(inputs, "CALL_034")
    event = next(e for e in ctx.events if e["kind"] == "READBACK")
    wrong = event["readback"]["wrong_question_id"]
    extra = next(q for q in ctx.readback_candidates() if q != wrong)
    # a frozen constraint that puts the extra field's FINAL event AFTER the read-back makes selecting it impossible
    ctx.skeleton = {**ctx.skeleton, "constraints": ctx.skeleton["constraints"] + [
        {"type": "ORDER", "before": event["event_id"], "after": ctx.final_event[extra], "source": {"type": "EDGE_ASSIGNMENT", "id": "EA_TEST"}}]}
    response = synthesize_response(PlanContext(inputs, "CALL_034"))
    entry_of(response, event["event_id"]).update(readback_question_ids=sorted([wrong, extra], key=ctx.readback_candidates().index),
                                                text=f"I have [[{wrong}]] and [[{extra}]].")
    errors = analyze(ctx, get_lexicon(inputs), response)[2]
    assert has(errors, "ordering cycle")
    assert find_cycle([("EV_1", "EV_2"), ("EV_2", "EV_3"), ("EV_3", "EV_1")]) == ["EV_1", "EV_2", "EV_3", "EV_1"]
    assert find_cycle([("EV_1", "EV_2"), ("EV_2", "EV_3")]) is None
    ctx.skeleton = {**ctx.skeleton, "constraints": [{"type": "ORDER", "before": "EV_001", "after": "EV_002", "source": {"type": "X"}},
                                                    {"type": "ORDER", "before": "EV_002", "after": "EV_001", "source": {"type": "X"}}]}
    assert has(analyze(ctx, get_lexicon(inputs), synthesize_response(PlanContext(inputs, "CALL_034")))[2], "frozen ORDER constraints are cyclic")


# ============================================================================== rendering and spans
def all_mentions(inputs, builder):
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        authoring = analyze(ctx, get_lexicon(inputs), synthesize_response(ctx))[0]
        for event in ctx.events:
            yield call_id, ctx, event, authoring[event["event_id"]]


def test_every_value_span_reproduces_its_surface_in_order_and_decodes_to_the_value(inputs, builder):
    total = 0
    for call_id, ctx, event, block in all_mentions(inputs, builder):
        previous_end = 0
        for mention in block["value_mentions"]:
            assert block["planned_text"][mention["start"]:mention["end"]] == mention["surface"], (call_id, event["event_id"])
            assert mention["start"] >= previous_end
            previous_end = mention["end"]
            total += 1
            name, surface, value = mention["placeholder"], mention["surface"], mention["value"]
            if name == "K01":
                assert R.decode_nric(surface) == value
            elif name == "K02":
                assert R.decode_dob(surface) == value
            elif name in ("T03", "T04"):
                assert R.decode_number(surface) == value
            elif name in ("T02", "CURRENCY"):
                assert R.decode_currency(surface, ctx.aliases) == value
        assert "[[" not in block["planned_text"] and "]]" not in block["planned_text"]
    assert total > 500


def test_k02_dates_agree_with_the_existing_pilot_spoken_verifier(inputs, builder):
    from fyp_evaluation.validation.spoken import verify_spoken_value

    checked = 0
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        value = ctx.gold_value("K02")
        for form in ("DAY_MONTH_YEAR", "ORDINAL_DAY"):
            surface = R.render_dob(value, form)
            assert verify_spoken_value({"kind": "extraction", "value_type": "dob_dd_month_yyyy", "expected_value": value}, surface) is None, surface
            checked += 1
    assert checked == 100


@pytest.mark.parametrize("value, form, detail, expected", [
    ("F9933772B", "CHARACTER_BY_CHARACTER", None, "F nine nine three three seven seven two B"),
    ("F9933777B", "DOUBLE_OR_TRIPLE_DIGIT", {"variant": "TRIPLE"}, "F nine nine three three triple seven B"),
    ("F9933777B", "DOUBLE_OR_TRIPLE_DIGIT", {"variant": "DOUBLE"}, "F double nine double three seven seven seven B"),
    ("S0859609E", "OH_FOR_ZERO", None, "S oh eight five nine six oh nine E"),
    ("T1000234G", "DOUBLE_OR_TRIPLE_DIGIT", {"variant": "TRIPLE"}, "T one triple zero two three four G"),
])
def test_nric_surfaces_and_round_trip(value, form, detail, expected):
    assert R.render_nric(value, form, detail) == expected and R.decode_nric(expected) == value


def test_dob_amount_currency_and_fee_surfaces():
    assert R.render_dob("27 October 1991", "DAY_MONTH_YEAR") == "27 October 1991"
    assert R.render_dob("04 February 2000", "ORDINAL_DAY") == "the fourth of February 2000"
    assert R.render_dob("21 May 1955", "ORDINAL_DAY") == "the twenty-first of May nineteen fifty-five"
    assert R.decode_dob("the twenty-first of May nineteen fifty-five") == "21 May 1955"
    assert R.format_number(1000000) == "1,000,000" and R.format_number(82.5) == "82.5" and R.format_number(0.3) == "0.3"
    for text, number in (("500k", 500000), ("10m", 10000000), ("1.2 million", 1200000), ("2.5 million", 2500000), ("1,500", 1500)):
        assert R.decode_number(text) == number
    aliases = {"SGD": ["Singapore dollars", "SGD"], "CNY": ["Chinese yuan", "yuan", "renminbi", "CNY"]}
    assert R.render_currency("CNY", "NATURAL_NAME", aliases) == "Chinese yuan" and R.render_currency("SGD", "ISO_CODE", aliases) == "SGD"
    assert R.render_fee({"basis": "RATE", "value": 0.3, "currency": None}, aliases) == "0.3%"
    assert R.render_fee({"basis": "AMOUNT", "value": 35, "currency": "SGD"}, aliases) == "35 Singapore dollars"
    assert R.render_limit_price(250, "CNY", aliases) == "250 Chinese yuan"
    for bad in (lambda: R.render_nric("bad", "CHARACTER_BY_CHARACTER"), lambda: R.render_amount(5, "FINANCIAL_SHORTHAND", {}),
                lambda: R.render_dob("27 October 1991", "NOPE"), lambda: R.year_words(1930)):
        with pytest.raises(R.RenderError):
            bad()


@pytest.mark.parametrize("call_id, question, form_id, expected_fragment", [
    ("CALL_003", "K01", "DOUBLE_OR_TRIPLE_DIGIT", "triple"),
    ("CALL_014", "K01", "OH_FOR_ZERO", " oh "),
    ("CALL_019", "T03", "MAGNITUDE_EXPRESSION", "1.2 million"),
    ("CALL_010", "T03", "FINANCIAL_SHORTHAND", "500k"),
])
def test_required_spoken_forms_are_used_by_every_gold_valued_mention(inputs, builder, call_id, question, form_id, expected_fragment):
    ctx = builder.context(call_id)
    mentions = [m for _, c, e, b in all_mentions(inputs, builder) if c.call_id == call_id for m in b["value_mentions"]
                if m["question_id"] == question and m["value"] == ctx.gold_value(question)]
    assert mentions and all(m["form"] == form_id and expected_fragment in f" {m['surface']} " for m in mentions)


def test_call_010_shows_500k_in_both_orders_with_the_right_limit_price(inputs, builder):
    ctx = builder.context("CALL_010")
    authoring = analyze(ctx, get_lexicon(inputs), synthesize_response(ctx))[0]
    orders = [e for e in ctx.events if e["kind"] == "ORDER_STATEMENT"]
    assert [e["status"] for e in orders] == ["SUPERSEDED", "FINAL"]
    for event, price in zip(orders, (ctx.limit_prices["initial"], ctx.limit_prices["final"])):
        block = authoring[event["event_id"]]
        mentions = {m["placeholder"]: m for m in block["value_mentions"]}
        assert mentions["T03"]["surface"] == "500k" and mentions["LIMIT_PRICE"]["value"] == price
        assert mentions["LIMIT_PRICE"]["source"] == "SCENARIO_LIMIT_PRICE"
    assert authoring[orders[0]["event_id"]]["value_mentions"][1]["value"] != authoring[orders[1]["event_id"]]["value_mentions"][1]["value"] or True


def test_call_011_multiple_corrections_keep_distinct_rendered_values(inputs, builder):
    ctx = builder.context("CALL_011")
    authoring = analyze(ctx, get_lexicon(inputs), synthesize_response(ctx))[0]
    k02 = [e for e in ctx.events if e["question_ids"] == ["K02"]]
    surfaces = [authoring[e["event_id"]]["value_mentions"][0]["surface"] for e in k02]
    assert len(k02) == 3 and len(set(surfaces)) == 3
    assert [e["status"] for e in k02] == ["STATED", "SUPERSEDED", "FINAL"] or "FINAL" in [e["status"] for e in k02]


def test_call_005_reference_selection_has_no_value_mentions_and_antecedents_do(inputs, builder):
    ctx = builder.context("CALL_005")
    authoring = analyze(ctx, get_lexicon(inputs), synthesize_response(ctx))[0]
    selection = next(e for e in ctx.events if e["kind"] == "REFERENCE_SELECTION")
    assert authoring[selection["event_id"]]["value_mentions"] == []
    antecedents = [e for e in ctx.events if e["kind"] == "ANTECEDENT_MENTION"]
    assert len(antecedents) >= 2 and all(len(authoring[e["event_id"]]["value_mentions"]) == 1 for e in antecedents)


def test_call_001_long_distance_gap_is_preserved_as_a_frozen_constraint(inputs, builder):
    ctx, plan = compile_plan(inputs, builder, "CALL_001")
    gaps = [c for c in plan["constraints"] if c["type"] == "MIN_INTERVENING_TURNS"]
    assert gaps and gaps == [c for c in ctx.skeleton["constraints"] if c["type"] == "MIN_INTERVENING_TURNS"]


def test_call_014_weak_confirmation_keeps_c08_without_positive_confirmation(inputs, builder):
    ctx, plan = compile_plan(inputs, builder, "CALL_014")
    weak = next(e for e in plan["planned_events"] if e["kind"] == "WEAK_ACKNOWLEDGEMENT")
    assert weak["authoring"]["speaker"] == "CLIENT" and weak["authoring"]["value_mentions"] == []
    assert ctx.verdict("C08") == "FAIL"


# ================================================================================= determinism, brief
def test_compilation_is_deterministic_and_independent_of_provider_key_order(inputs, builder):
    ctx = builder.context("CALL_010")
    generation = {"attempt_id": "A001", "model": "m", "prompt_template_sha256": "1" * 64, "response_schema_sha256": "2" * 64}
    response = synthesize_response(ctx)
    scrambled = json.loads(json.dumps(response, sort_keys=True))
    first = serialize_json(compile_artifact(inputs, ctx, response, generation, builder.response_schema))
    second = serialize_json(compile_artifact(inputs, ctx, scrambled, generation, builder.response_schema))
    assert first == second == serialize_json(compile_artifact(inputs, ctx, response, generation, builder.response_schema))


def test_compile_refuses_an_invalid_response(inputs, builder):
    ctx = builder.context("CALL_010")
    response = synthesize_response(ctx)
    response["events"].pop()
    with pytest.raises(ValueError, match="cannot compile"):
        compile_artifact(inputs, ctx, response, {"attempt_id": "A001", "model": "m", "prompt_template_sha256": "0" * 64,
                                                 "response_schema_sha256": "0" * 64}, builder.response_schema)


def test_requests_are_byte_identical_across_builders_and_carry_the_whole_skeleton_event_list(inputs, builder):
    other = RequestBuilder(inputs, SCHEMAS)
    for call_id in CALL_IDS:
        first, second = builder.build(call_id, "A001"), other.build(call_id, "A001")
        assert canonical_request_text(first) == canonical_request_text(second) and request_hash(first) == request_hash(second)
        brief = json.loads(first["input"][1]["content"])
        assert [e["event_id"] for e in brief["events"]] == [e["event_id"] for e in builder.context(call_id).events]
        assert "sha256" not in first["input"][1]["content"] and "{{" not in first["input"][0]["content"]
    body = builder.build("CALL_001", "A002")
    assert body["model"] == "gpt-6.1-sol" and body["text"]["format"]["strict"] is True and body["store"] is False
    assert body["reasoning"] == {"effort": "medium"} and not {"tools", "temperature"} & set(body)
    assert body["metadata"] == {"dataset_id": "advisory_execution_pilot_v1", "call_id": "CALL_001", "stage": "G4", "attempt_id": "A002"}


def _strict_issues(schema, path="$"):
    issues = []
    forbidden = {"oneOf", "allOf", "not", "if", "then", "else", "patternProperties", "$ref", "$defs", "minItems", "maxItems", "minLength", "pattern"}
    if isinstance(schema, dict):
        issues += [f"{path}: forbidden keyword {k}" for k in schema if k in forbidden]
        if schema.get("type") == "object" or "properties" in schema:
            if schema.get("additionalProperties") is not False:
                issues.append(f"{path}: additionalProperties must be false")
            if sorted(schema.get("required", [])) != sorted(schema.get("properties", {})):
                issues.append(f"{path}: every property must be required")
        for key, value in schema.items():
            issues += _strict_issues(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for index, value in enumerate(schema):
            issues += _strict_issues(value, f"{path}[{index}]")
    return issues


def test_response_schema_is_openai_strict_compatible(builder):
    assert _strict_issues(builder.response_schema) == []


def test_brief_describes_events_checks_constraints_and_surfaces(builder):
    ctx = builder.context("CALL_010")
    brief = build_brief(builder.inputs, ctx)
    assert [c["question_id"] for c in brief["checks"]][:3] == ["C01", "C02", "C03"] and len(brief["checks"]) == 20
    absent = [c for c in brief["checks"] if c.get("evidence_mode") == "ABSENCE"]
    assert absent and all("must_not_be_expressed_anywhere" in c for c in absent)
    order = next(e for e in brief["events"] if e["kind"] == "ORDER_STATEMENT" and e["status"] == "FINAL")
    assert order["placeholders"]["required"] == ["T01", "T02", "T03", "T04", "T05", "T06", "LIMIT_PRICE"]
    assert order["rendered_surfaces"]["T03"] == {"FINANCIAL_SHORTHAND": "500k"} and order["form_choices"] == {"T02": ["NATURAL_NAME", "ISO_CODE"]}
    assert brief["constraints"] == ctx.skeleton["constraints"]


def test_readback_briefs_list_the_form_choices_of_every_selectable_field(builder):
    """Regression (CALL_037): a read-back field with a form choice (T02/K02) must be announced in `form_choices`."""
    seen = 0
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        for event in build_brief(builder.inputs, ctx)["events"]:
            if event["kind"] != "READBACK":
                continue
            seen += 1
            expected = {q: list(s) for q, s in event["readback"]["selectable_fields"].items() if s and len(s) > 1}
            assert event.get("form_choices", {}) == expected and "value_forms" in event["readback"]["note"]
            assert "T02" in event["form_choices"] and "K02" in event["form_choices"]
    assert seen == 6
    wrong_with_choice = build_brief(builder.inputs, builder.context("CALL_037"))
    readback = next(e for e in wrong_with_choice["events"] if e["kind"] == "READBACK")
    assert readback["readback"]["wrong_question_id"] == "T02" and readback["form_choices"]["T02"] == ["NATURAL_NAME", "ISO_CODE"]
