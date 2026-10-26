"""G6 enforces the temporal meaning of a cross-language distractor variant (the distractor mentions come before the final value).

The frozen variant says "the earlier state is X, the later state is Y"; for competing-values distractors the plan had no order
against the final transaction value, so a dialogue could pass structural validation with the order reversed (seen live in
CALL_030). The rule is derived in G6 from the frozen spec and accepted plan alone. Offline; nothing upstream changes."""

import json

import pytest

import g6_support
import multilingual_support as M
from fyp_evaluation.dialogues.feasibility import preflight_errors, synthesize_dialogue
from fyp_evaluation.dialogues.language_order import language_order_constraints
from fyp_evaluation.dialogues.request import DialogueContext, RequestBuilder

SPECS = M.upstream_inputs().specs


def affected():
    return [c for c, s in SPECS.items()
            if any((e.get("language_variant") or "").startswith("CROSS_LANGUAGE") and e.get("family") == "DISTRACTOR_COMPETING_VALUES"
                   for e in s["edge_cases"])]


@pytest.fixture(scope="module")
def inputs():
    return M.verification_inputs(affected() + [M.first_call(p) for p in M.PROFILES])


def test_exactly_the_cross_language_distractor_calls_get_the_derived_order(inputs):
    calls = affected()
    assert len(calls) == 6 and {SPECS[c]["metadata"]["language_profile"] for c in calls} == {"EN_ZH_MIXED"}
    for call_id in calls:
        ctx = DialogueContext(inputs, call_id)
        derived = [c for c in ctx.constraints if c.get("source", {}).get("type") == "LANGUAGE_VARIANT"]
        assert derived and all(c["type"] == "ORDER" for c in derived), call_id
        plan = inputs.plans[call_id]
        assert ctx.constraints[:len(plan["constraints"]) + len(plan["derived_constraints"])] == plan["constraints"] + plan["derived_constraints"]
        for constraint in derived:  # a distractor mention of that assignment, before the final event of a target question
            before = next(e for e in plan["planned_events"] if e["event_id"] == constraint["before"])
            assert before["kind"] == "DISTRACTOR_MENTION" and constraint["source"]["id"] in before["assignment_ids"]
            assert any(c["final_event_id"] == constraint["after"] for c in plan["checks"] if c.get("final_event_id"))
        assert preflight_errors(ctx) == [], call_id


def test_every_other_call_and_the_pilot_release_keep_exactly_their_plan_constraints(inputs):
    others = [c for c in M.upstream_inputs().call_ids if c not in affected()]
    sample = [M.first_call(p) for p in M.PROFILES] + others[:5]
    covered = M.verification_inputs(sample)
    for call_id in sample:
        ctx = DialogueContext(covered, call_id)
        assert ctx.constraints == covered.plans[call_id]["constraints"] + covered.plans[call_id]["derived_constraints"], call_id
    pilot = g6_support.shared_inputs()
    pilot_ctx = DialogueContext(pilot, "CALL_001")
    assert not pilot_ctx.multilingual and pilot_ctx.constraints == pilot.plans["CALL_001"]["constraints"] + pilot.plans["CALL_001"]["derived_constraints"]


def swapped_distractor_after_final(ctx):
    good = synthesize_dialogue(ctx)
    derived = next(c for c in ctx.constraints if c.get("source", {}).get("type") == "LANGUAGE_VARIANT")
    where = lambda event: next(i for i, t in enumerate(good["turns"]) if f"[[{event}]]" in t["text"])  # noqa: E731
    i, j = where(derived["before"]), where(derived["after"])
    assert i != j and ctx.speakers[derived["before"]] == ctx.speakers[derived["after"]]
    bad = json.loads(json.dumps(good))
    bad["turns"][i], bad["turns"][j] = good["turns"][j], good["turns"][i]  # whole turns: strict alternation is preserved
    return good, bad, derived


@pytest.mark.parametrize("call_id", affected())
def test_a_dialogue_with_the_distractor_after_the_final_value_is_rejected(inputs, call_id):
    builder = RequestBuilder(inputs, M.SCHEMAS)
    ctx = builder.context(call_id)
    from fyp_evaluation.dialogues.validate_response import DialogueValidator

    validator = DialogueValidator(builder.response_schema)
    good, bad, derived = swapped_distractor_after_final(ctx)
    assert validator.errors(good, ctx) == []
    errors = validator.errors(bad, ctx)
    assert f"ORDER: {derived['before']} must come before {derived['after']}" in errors, errors


def test_the_brief_states_the_order_to_the_author(inputs):
    call_id = "CALL_030"
    builder = RequestBuilder(inputs, M.SCHEMAS)
    brief = json.loads(builder.build(call_id, "A001")["input"][1]["content"])
    derived = [c for c in builder.context(call_id).constraints if c.get("source", {}).get("type") == "LANGUAGE_VARIANT"]
    for constraint in derived:
        assert {"type": "ORDER", "before": constraint["before"], "after": constraint["after"]} in brief["placement_constraints"]


def test_the_rule_reads_only_the_frozen_spec_and_the_accepted_plan(inputs):
    call_id = "CALL_030"
    spec, plan = SPECS[call_id], inputs.plans[call_id]
    assert language_order_constraints(spec, plan) == language_order_constraints(json.loads(json.dumps(spec)), json.loads(json.dumps(plan)))
    english_edges = json.loads(json.dumps(spec))
    for edge in english_edges["edge_cases"]:
        edge["language_variant"] = "NONE"
    assert language_order_constraints(english_edges, plan) == []
