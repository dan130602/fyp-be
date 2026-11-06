"""Python-derived advisory recommendation: derivation, offline conflict detection, the advisory-only lint and the G7 expectation.

Offline; the committed release is only read. The derivation is rule-based (call_intents, the C07 plan structure, gold T05, edge-case
targets), never keyed on a call ID."""

import copy
import json
import re

import pytest

from fyp_evaluation.dialogue_verifications.inputs import load_inputs as load_g7_inputs
from fyp_evaluation.dialogue_verifications.request import RequestBuilder as G7Builder, advisory_expectation
from fyp_evaluation.dialogues.advisory import advisory_conflicts, advisory_lint, advisory_recommendation
from fyp_evaluation.dialogues.feasibility import preflight_errors
from fyp_evaluation.dialogues.request import DialogueContext
from g6_support import CALL_IDS, RELEASE, SCHEMAS, shared_inputs

FAIL_CALLS = ["CALL_001", "CALL_024", "CALL_031", "CALL_036", "CALL_042", "CALL_047", "CALL_049"]
PROMPTS = SCHEMAS.parent / "prompts"


def parts(call_id):
    inputs = shared_inputs()
    spec, plan = inputs.specs[call_id], inputs.plans[call_id]
    return spec, inputs.configs[call_id], plan["planned_events"], plan["constraints"] + plan["derived_constraints"]


def squashed(text):
    return re.sub(r"\s+", " ", text)


# ---- derivation -------------------------------------------------------------------------------------------------------------------------

def test_exactly_the_calls_without_a_planned_recommendation_get_data_and_it_comes_from_frozen_inputs():
    keyed = {c: advisory_recommendation(*parts(c)[:3]) for c in CALL_IDS}
    assert sorted(c for c, d in keyed.items() if d) == FAIL_CALLS
    for call_id in FAIL_CALLS:
        spec, config, events, _ = parts(call_id)
        data = keyed[call_id]
        assert data["product_name"] == next(p["name"] for p in config["products"] if "FINAL" in p["roles"])
        assert data["direction"] == next(g["expected"]["expected_value"] for g in spec["gold"] if g["question_id"] == "T05")
        ids = {e["event_id"]: e for e in events}
        assert all("C08" in ids[e]["question_ids"] for e in data["place_before_event_ids"]) and data["place_before_event_ids"]
        assert all(set(ids[e]["question_ids"]) & {"C01", "C06"} for e in data["keep_apart_from_event_ids"])
        assert set(data) == {"product_name", "direction", "place_after_event_ids", "place_before_event_ids", "keep_apart_from_event_ids"}


def test_the_trigger_follows_call_intents_the_plan_structure_and_the_gold_not_a_call_id():
    spec, config, events, _ = parts("CALL_001")
    only_advisory = copy.deepcopy(spec); only_advisory["metadata"]["call_intents"] = ["ADVISORY"]
    data = advisory_recommendation(only_advisory, config, events)
    assert data is not None and "direction" not in data  # advice without an order names the product only
    execution_only = copy.deepcopy(spec); execution_only["metadata"]["call_intents"] = ["EXECUTION"]
    assert advisory_recommendation(execution_only, config, events) is None
    with_planned = copy.deepcopy(events); with_planned[0]["question_ids"] = with_planned[0]["question_ids"] + ["C07"]
    assert advisory_recommendation(spec, config, with_planned) is None  # a planned recommendation event already carries it
    swapped = copy.deepcopy(spec)
    for g in swapped["gold"]:
        if g["question_id"] == "T05":
            g["expected"]["expected_value"] = "SELL"
    assert advisory_recommendation(swapped, config, events)["direction"] == "SELL"


def test_place_after_events_are_those_whose_edge_case_targets_the_product_or_direction_only():
    # CALL_031: EC_PRODUCT_REFERENCE targets T01 -> its antecedent mentions and the selection must come first
    assert advisory_recommendation(*parts("CALL_031")[:3])["place_after_event_ids"] == ["EV_014", "EV_015", "EV_016"]
    # CALL_001/036 have REFERENCE_SELECTION events too, but they resolve a previous QUANTITY: nothing about the product is settled by them
    for call_id in ("CALL_001", "CALL_036"):
        events = parts(call_id)[2]
        assert any(e["kind"] == "REFERENCE_SELECTION" for e in events)
        assert advisory_recommendation(*parts(call_id)[:3])["place_after_event_ids"] == []
    spec, config, events, _ = parts("CALL_001")
    retargeted = copy.deepcopy(spec)
    for edge in retargeted["edge_cases"]:
        if edge["edge_case_id"] == "EC_PREVIOUS_QUANTITY_REFERENCE":
            edge["target_question_ids"] = ["T05"]
    assert advisory_recommendation(retargeted, config, events)["place_after_event_ids"]  # a direction-determining edge case is derived the same way


# ---- offline conflict detection -----------------------------------------------------------------------------------------------------------------

def test_the_seven_calls_have_no_conflicting_placement_requirement_and_preflight_would_report_one():
    inputs = shared_inputs()
    for call_id in FAIL_CALLS:
        ctx = DialogueContext(inputs, call_id)
        assert advisory_conflicts(ctx.advisory, ctx.events, ctx.constraints) == [] and preflight_errors(ctx) == []
    spec, config, events, constraints = parts("CALL_031")
    data = advisory_recommendation(spec, config, events)
    confirmation = data["place_before_event_ids"][0]
    contradictory = constraints + [{"type": "ORDER", "before": confirmation, "after": "EV_016"}]  # the selection forced AFTER the confirmation
    errors = advisory_conflicts(data, events, contradictory)
    assert any("EV_016" in e and confirmation in e for e in errors)
    ctx = DialogueContext(inputs, "CALL_031")
    ctx.constraints = contradictory
    assert any("must precede the recommendation" in e for e in preflight_errors(ctx))


def test_a_shared_turn_that_leaves_no_valid_recommendation_slot_is_reported():
    spec, config, events, constraints = parts("CALL_031")
    data = advisory_recommendation(spec, config, events)
    employee_after = next(e["event_id"] for e in events if e["authoring"]["speaker"] == "EMPLOYEE" and e["event_id"] not in data["place_before_event_ids"])
    forced = copy.deepcopy(data)
    forced["place_after_event_ids"] = [employee_after]
    keep = data["keep_apart_from_event_ids"][0]
    same_turn = constraints + [{"type": "SAME_TURN", "events": [employee_after, data["place_before_event_ids"][0]]},
                               {"type": "SAME_TURN", "events": [employee_after, keep]}]
    assert any("also holds objective/loss-tolerance evidence" in e for e in advisory_conflicts(forced, events, same_turn))
    mixed = constraints + [{"type": "SAME_TURN", "events": [data["place_before_event_ids"][1], data["place_before_event_ids"][0]]}]
    client_event = next(e["event_id"] for e in events if e["authoring"]["speaker"] == "CLIENT")
    forced_client = copy.deepcopy(data)
    forced_client["place_after_event_ids"] = [data["place_before_event_ids"][1]]
    assert any("not an employee turn" in e for e in advisory_conflicts(forced_client, events, mixed)) or client_event


# ---- briefs and prompts -------------------------------------------------------------------------------------------------------------------------

def test_the_g6_brief_carries_the_derived_data_for_the_seven_and_nothing_new_for_the_other_43():
    from fyp_evaluation.dialogues.request import RequestBuilder
    builder = RequestBuilder(shared_inputs(), SCHEMAS, PROMPTS / "stage6")
    for call_id in CALL_IDS:
        brief = json.loads(builder.build(call_id, "A001")["input"][1]["content"])
        if call_id in FAIL_CALLS:
            assert brief["advisory_recommendation"] == advisory_recommendation(*parts(call_id)[:3])
            assert "T05" not in json.dumps(brief) and "expected" not in brief  # derived facts only: no question IDs or gold in the brief
        else:
            assert "advisory_recommendation" not in brief


def test_the_g7_prompt_and_brief_state_the_same_expectation_with_resolved_turn_ids():
    text = squashed((PROMPTS / "stage7" / "dialogue_verification.v1.md").read_text(encoding="utf-8"))
    for required in ("advisory_expectation", "clearly recommend the frozen `product_name`", "WITHOUT a concrete rationale",
                     "place_after_turn_ids", "place_before_turn_ids", "keep_apart_turn_ids", "by meaning and never by keywords",
                     'A bare "go ahead with this" that names neither', "MISSING_SUPPORT", "Adjacency alone is not a rationale",
                     'Words such as "as" or "since" are not decisive', "OMISSION_LEAKAGE", "gives the answer away", "code EDGE_CASE",
                     "Earlier placement and ending its turn are preferences, never defects", "a statement of understanding is a MISMATCH"):
        assert required in text, required
    inputs = load_g7_inputs(RELEASE, SCHEMAS)
    builder = G7Builder(inputs, SCHEMAS, PROMPTS / "stage7")
    for call_id in CALL_IDS:
        brief = json.loads(builder.brief_text(call_id))
        assert ("advisory_expectation" in brief) == (call_id in FAIL_CALLS)
    ctx = builder.context("CALL_031")
    expectation = advisory_expectation(ctx)
    assert expectation["place_after_turn_ids"] == sorted({ctx.turn_of_event[e] for e in ("EV_014", "EV_015", "EV_016")})
    assert expectation["product_name"] == "Crest Emerging Markets Bond Fund" and expectation["direction"] == "SELL"


# ---- advisory-only lint ---------------------------------------------------------------------------------------------------------------------------

DATA = {"product_name": "Acme Fund", "direction": "BUY", "place_after_event_ids": ["EV_001"], "place_before_event_ids": ["EV_004"],
        "keep_apart_from_event_ids": ["EV_002"]}


def dialogue(*texts, spans=()):
    """EMPLOYEE/CLIENT alternating turns; `spans` are (event_id, turn_id, start, end)."""
    turns = [{"turn_id": i, "speaker": ("EMPLOYEE", "CLIENT")[i % 2], "text": t} for i, t in enumerate(texts)]
    return {"turns": turns, "event_alignment": [{"event_id": e, "turn_id": t, "start": s, "end": n} for e, t, s, n in spans]}


SPANS = [("EV_001", 1, 0, 3), ("EV_002", 1, 0, 3), ("EV_004", 6, 0, 15)]  # settled at turn 1, loss-tolerance at turn 1, confirmation at turn 6


def with_recommendation(employee, client="Okay, noted."):
    return dialogue("Hello.", "Hi.", "Thank you.", "Alright.", employee, client, "Please confirm.", "Yes.", spans=SPANS)


GOOD = with_recommendation("I recommend that you buy Acme Fund.")


def test_the_lint_is_quiet_for_a_clean_recommendation_and_lists_signals_otherwise():
    assert advisory_lint(DATA, GOOD)["signals"] == [] and advisory_lint(None, GOOD)["required"] is False
    cases = {
        "possible rationale wording": ("I recommend that you buy Acme Fund because it suits your goals.", "Okay, noted."),
        "no BUY action word": ("Acme Fund is the one I have in mind.", "Okay, noted."),
        "order detail": ("I recommend that you buy 100 units of Acme Fund.", "Okay, noted."),
        "more than a mild acknowledgement": ("I recommend that you buy Acme Fund.", "Okay, I understand."),
    }
    for needle, (employee, client) in cases.items():
        report = advisory_lint(DATA, with_recommendation(employee, client))
        assert any(needle in s for s in report["signals"]), (needle, report["signals"])
    none = advisory_lint(DATA, dialogue("Hello.", "Hi.", "Shall we continue?", "Yes.", "Please confirm.", "Yes."))
    assert none["signals"] == ["no employee sentence outside the planned events names the product"]


def test_the_lint_flags_wrong_placement_and_adjacency_without_gating_anything():
    late = advisory_lint(DATA, dialogue("Hello.", "Hi.", "Please confirm.", "Yes.", "Fine.", "Hm.", "I recommend that you buy Acme Fund.", "Okay, noted.",
                                        spans=[("EV_001", 1, 0, 3), ("EV_002", 1, 0, 3), ("EV_004", 2, 0, 15)]))
    text = " ".join(late["signals"])
    assert "comes after final-confirmation event EV_004" in text and "adjacent to objective/loss-tolerance event EV_002" not in text
    early = advisory_lint(DATA, dialogue("Hello.", "Hi.", "I recommend that you buy Acme Fund.", "Okay, thanks.", "Fine.", "Hm.", "Please confirm.", "Yes.",
                                         spans=[("EV_001", 5, 0, 3), ("EV_002", 1, 0, 3), ("EV_004", 6, 0, 15)]))
    assert any("comes before EV_001" in s for s in early["signals"]) and any("adjacent to objective/loss-tolerance event EV_002" in s for s in early["signals"])
    assert isinstance(early, dict) and set(early) == {"required", "signals", "recommendation"}  # a report, never a verdict


def test_the_lint_ignores_words_and_digits_that_are_part_of_the_product_name():
    data = {**DATA, "product_name": "Orion Global Income Fund 2031"}
    assert advisory_lint(data, with_recommendation("I recommend that you buy Orion Global Income Fund 2031."))["signals"] == []
    flagged = advisory_lint(data, with_recommendation("I recommend that you buy Orion Global Income Fund 2031 for the income."))
    assert any("possible rationale wording" in s for s in flagged["signals"])
