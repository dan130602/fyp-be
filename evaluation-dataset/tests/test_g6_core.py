"""G6 core: marker rendering, structural validation, placement constraints, length policy, brief, schemas. Offline."""

import json

import pytest
from jsonschema import Draft202012Validator

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.dialogues import config
from fyp_evaluation.dialogues.compile import compile_artifact
from fyp_evaluation.dialogues.feasibility import preflight_errors, synthesize_dialogue
from fyp_evaluation.dialogues.inputs import g5_report_fingerprint
from fyp_evaluation.dialogues.render import render, word_count
from fyp_evaluation.dialogues.request import DialogueContext, RequestBuilder
from fyp_evaluation.dialogues.validate_response import DialogueValidator
from g6_support import CALL_IDS, RELEASE, SCHEMAS, shared_inputs

PROMPTS = SCHEMAS.parent / "prompts" / "stage6"
FILLER = "Thank you, that is noted."


@pytest.fixture(scope="module")
def builder():
    return RequestBuilder(shared_inputs(), SCHEMAS, PROMPTS)


@pytest.fixture(scope="module")
def validator(builder):
    return DialogueValidator(builder.response_schema)


def contexts(builder, call_id):
    return builder.context(call_id)


def turn_of(response, event_id):
    marker = f"[[{event_id}]]"
    return next(i for i, turn in enumerate(response["turns"]) if marker in turn["text"])


def other(speaker):
    return "CLIENT" if speaker == "EMPLOYEE" else "EMPLOYEE"


def errors_for(builder, validator, call_id, mutate):
    ctx = builder.context(call_id)
    response = synthesize_dialogue(ctx)
    mutate(ctx, response)
    return validator.errors(response, ctx)


# ============================================================================================ all 50 plans
def test_every_real_plan_is_feasible_and_the_synthetic_dialogue_is_accepted(builder, validator):
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        assert preflight_errors(ctx) == [], call_id
        response = synthesize_dialogue(ctx)
        assert validator.errors(response, ctx) == [], call_id
        turns, alignment, _ = render(ctx, response)
        assert {span["event_id"] for span in alignment} == set(ctx.texts)
        assert all(turns[s["turn_id"]]["text"][s["start"]:s["end"]] == ctx.texts[s["event_id"]] for s in alignment), call_id
        assert [t["turn_id"] for t in turns] == list(range(len(turns)))


def test_schemas_are_valid_and_a_compiled_dialogue_validates_against_the_artifact_schema(builder):
    for name in ("dialogue_response", "dialogue", "dialogue_provenance", "dialogue_record"):
        Draft202012Validator.check_schema(parse_json_text((SCHEMAS / "stage6" / f"{name}.schema.json").read_text(encoding="utf-8")))
    ctx = builder.context("CALL_017")
    generation = {"attempt_id": "A001", "model": config.MODEL, "params": {"reasoning_effort": "medium", "max_output_tokens": 32768,
                  "length_policy": config.LENGTH_POLICY}, "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
    artifact = compile_artifact(shared_inputs(), ctx, synthesize_dialogue(ctx), generation, builder.response_schema)
    assert list(Draft202012Validator(builder.artifact_schema).iter_errors(artifact)) == []
    assert artifact["length"]["policy_version"] == "g6-length-v1" and artifact["length"]["assumed_wpm"] == 160
    assert artifact["length"]["estimated_seconds"] == round(artifact["length"]["words"] * 60 / 160, 1)


# ============================================================================================ rendering
def test_call_017_nested_anchor_is_not_a_duplicate_placement(builder, validator):
    ctx = builder.context("CALL_017")
    assert ctx.texts["EV_011"] in ctx.texts["EV_013"]  # the approved identifier statement is contained in the correction
    response = synthesize_dialogue(ctx)
    assert validator.errors(response, ctx) == []
    _, alignment, _ = render(ctx, response)
    spans = {s["event_id"]: s for s in alignment}
    assert spans["EV_011"]["turn_id"] != spans["EV_013"]["turn_id"]
    placed_together = {"turns": [{"speaker": "CLIENT", "text": "[[EV_013]]"}]}
    assert not any("repeated outside" in e for e in validator.errors({"turns": response["turns"][:1] + placed_together["turns"]}, ctx))


def test_false_start_and_unicode_text_is_inserted_exactly(builder, validator):
    ctx = builder.context("CALL_010")
    start_text = ctx.texts["EV_013"]
    assert start_text.endswith("—") or start_text.endswith("…")
    turns, alignment, _ = render(ctx, synthesize_dialogue(ctx))
    span = next(s for s in alignment if s["event_id"] == "EV_013")
    assert turns[span["turn_id"]]["text"][span["start"]:span["end"]] == start_text
    shared = [s for s in alignment if s["turn_id"] == span["turn_id"]]
    assert len(shared) == 2 and shared[0]["end"] <= shared[1]["start"]  # multiple events in one turn, non-overlapping


def test_filler_around_a_marker_shifts_the_span_offsets(builder):
    ctx = builder.context("CALL_027")
    response = synthesize_dialogue(ctx)
    index = turn_of(response, "EV_001")
    response["turns"][index]["text"] = "Sure, one moment please. [[EV_001]] And that is all."
    turns, alignment, errors = render(ctx, response)
    span = next(s for s in alignment if s["event_id"] == "EV_001")
    assert errors == [] and span["start"] == len("Sure, one moment please. ")
    assert turns[index]["text"][span["start"]:span["end"]] == ctx.texts["EV_001"]


# ============================================================================================ marker and speaker rules
def test_missing_duplicate_unknown_and_glued_markers_are_rejected(builder, validator):
    missing = errors_for(builder, validator, "CALL_027", lambda c, r: r["turns"][turn_of(r, "EV_002")].update(text=FILLER))
    assert any("EV_002: missing" in e for e in missing)
    duplicate = errors_for(builder, validator, "CALL_027",
                           lambda c, r: r["turns"][turn_of(r, "EV_002")].update(text="[[EV_002]] [[EV_002]]"))
    assert any("EV_002: placed 2 times" in e for e in duplicate)
    unknown = errors_for(builder, validator, "CALL_027", lambda c, r: r["turns"][0].update(text="[[EV_999]] hello"))
    assert any("unknown event marker [[EV_999]]" in e for e in unknown)
    glued = errors_for(builder, validator, "CALL_027",
                       lambda c, r: r["turns"][turn_of(r, "EV_002")].update(text="word[[EV_002]]"))
    assert any("separate token" in e for e in glued)
    stray = errors_for(builder, validator, "CALL_027", lambda c, r: r["turns"][0].update(text="hello [[oops]]"))
    assert any("stray" in e for e in stray)


def test_wrong_speaker_non_alternation_and_blank_turns_are_rejected(builder, validator):
    def flip(ctx, response):
        response["turns"][turn_of(response, "EV_002")]["speaker"] = other(ctx.speakers["EV_002"])
    assert any("its speaker is" in e for e in errors_for(builder, validator, "CALL_027", flip))
    assert any("consecutive" in e for e in errors_for(builder, validator, "CALL_027",
                                                      lambda c, r: r["turns"].insert(1, dict(r["turns"][0]))))
    assert any("empty turn" in e for e in errors_for(builder, validator, "CALL_027", lambda c, r: r["turns"][0].update(text="  ")))
    ctx = builder.context("CALL_027")
    assert validator.errors({"turns": []}, ctx)
    assert validator.errors({"turns": [{"speaker": "EMPLOYEE", "text": "x", "extra": 1}]}, ctx)
    assert validator.errors({"turns": [{"speaker": "ADVISOR", "text": "x"}]}, ctx)


# ============================================================================================ placement constraints
def test_order_violation_is_rejected(builder, validator):
    def swap(ctx, response):
        order = next(c for c in ctx.constraints if c["type"] == "ORDER")
        i, j = turn_of(response, order["before"]), turn_of(response, order["after"])
        response["turns"][i], response["turns"][j] = response["turns"][j], response["turns"][i]
    assert any(e.startswith("ORDER:") for e in errors_for(builder, validator, "CALL_027", swap))


def test_same_turn_split_is_rejected(builder, validator):
    def split(ctx, response):
        index = turn_of(response, "EV_010")
        speaker = response["turns"][index]["speaker"]
        response["turns"][index]["text"] = "[[EV_010]]"
        response["turns"][index + 1:index + 1] = [{"speaker": other(speaker), "text": FILLER}, {"speaker": speaker, "text": "[[EV_013]]"}]
    errors = errors_for(builder, validator, "CALL_010", split)
    assert any(e.startswith("SAME_TURN:") for e in errors)


def test_not_same_turn_and_minimum_gap_violations_are_rejected(builder, validator):
    def merge(ctx, response):
        first, second = turn_of(response, "EV_021"), turn_of(response, "EV_022")
        response["turns"][first]["text"] = "[[EV_021]] [[EV_022]]"
        response["turns"][second]["text"] = FILLER
    errors = errors_for(builder, validator, "CALL_026", merge)
    assert any(e.startswith("NOT_SAME_TURN:") for e in errors)

    def too_close(ctx, response):  # CALL_001: EV_013 and EV_017 need at least 6 turns between them
        near, far = turn_of(response, "EV_013"), turn_of(response, "EV_017")
        response["turns"][far]["text"] = FILLER
        speaker = response["turns"][far]["speaker"]
        response["turns"][near + 1:near + 1] = [{"speaker": other(speaker), "text": FILLER}, {"speaker": speaker, "text": "[[EV_017]]"}]
    errors = errors_for(builder, validator, "CALL_001", too_close)
    assert any(e.startswith("MIN_INTERVENING_TURNS:") for e in errors)


def test_minimum_gap_boundary_is_at_least_n_turns_strictly_between(builder, validator):
    ctx = builder.context("CALL_001")
    response = synthesize_dialogue(ctx)
    positions = {s["event_id"]: s["turn_id"] for s in render(ctx, response)[1]}
    gap = abs(positions["EV_013"] - positions["EV_017"]) - 1
    assert gap >= 6 and validator.errors(response, ctx) == []


def test_impossible_grouped_constraints_fail_the_preflight(builder):
    ctx = DialogueContext(shared_inputs(), "CALL_010")  # fresh contexts: the builder's cached ones are shared by other tests
    ctx.constraints = ctx.constraints + [{"type": "NOT_SAME_TURN", "events": ["EV_010", "EV_013"], "source": {"type": "TEST", "id": "x"}}]
    assert any("conflicts with SAME_TURN" in e for e in preflight_errors(ctx))
    cyclic = DialogueContext(shared_inputs(), "CALL_027")
    first = next(c for c in cyclic.constraints if c["type"] == "ORDER")
    cyclic.constraints = cyclic.constraints + [{"type": "ORDER", "before": first["after"], "after": first["before"],
                                                "source": {"type": "TEST", "id": "x"}}]
    assert any("cycle" in e for e in preflight_errors(cyclic))
    with pytest.raises(ValueError):
        synthesize_dialogue(cyclic)


# ============================================================================================ filler rules
def test_planned_text_repeated_in_filler_and_unplanned_identity_values_are_rejected(builder, validator):
    repeated = errors_for(builder, validator, "CALL_027", lambda c, r: r["turns"][0].update(text=r["turns"][0]["text"] + " " + c.texts["EV_001"]))
    assert any("repeated outside its event marker" in e for e in repeated)
    ctx = builder.context("CALL_001")
    assert ctx.identity_surfaces
    surface = ctx.identity_surfaces[0][1]
    identity = errors_for(builder, validator, "CALL_001", lambda c, r: r["turns"][0].update(text=f"Your number is {surface}."))
    assert any("unplanned" in e and ctx.identity_surfaces[0][0] in e for e in identity)


def test_other_value_surfaces_become_review_notes_not_errors(builder, validator):
    ctx = builder.context("CALL_001")
    assert ctx.value_surfaces
    question_id, surface = ctx.value_surfaces[0]
    response = synthesize_dialogue(ctx)
    response["turns"][0]["text"] += f" Just to mention {surface} in passing."
    assert validator.errors(response, ctx) == []
    generation = {"attempt_id": "A001", "model": config.MODEL, "params": {"reasoning_effort": "medium", "max_output_tokens": 1,
                  "length_policy": config.LENGTH_POLICY}, "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
    artifact = compile_artifact(shared_inputs(), ctx, response, generation, builder.response_schema)
    assert {"turn_id": 0, "question_id": question_id, "surface": surface} in artifact["review_notes"]


def test_harmless_family_and_administrative_filler_is_accepted(builder, validator):
    ctx = builder.context("CALL_006")
    response = synthesize_dialogue(ctx)
    response["turns"][0]["text"] += " Good morning. My family is hosting a gathering next Saturday, so I will be away on the 12th; call me back on the 14th."
    assert validator.errors(response, ctx) == []


# ============================================================================================ length policy
def test_length_bounds_are_inclusive_and_use_the_recorded_policy(builder, validator):
    inputs = shared_inputs()
    base = builder.context("CALL_027")
    response = synthesize_dialogue(base)
    words = word_count(render(base, response)[0])

    def policy(low, high):
        return {**config.LENGTH_POLICY, "profiles": {**config.LENGTH_POLICY["profiles"],
                                                     "SHORT": {"accepted_words": [low, high], "target_words": [low, high]}}}
    assert validator.errors(response, DialogueContext(inputs, "CALL_027", policy(words, words))) == []
    assert any(e.startswith("length:") for e in validator.errors(response, DialogueContext(inputs, "CALL_027", policy(words + 1, words + 9))))
    assert any(e.startswith("length:") for e in validator.errors(response, DialogueContext(inputs, "CALL_027", policy(words - 9, words - 1))))
    assert config.LENGTH_POLICY["profiles"]["SHORT"]["accepted_words"] == [510, 750]
    assert config.LENGTH_POLICY["profiles"]["LONG"]["accepted_words"] == [5103, 9000]


def test_length_policy_keeps_both_profiles_inside_the_frozen_duration_at_150_to_170_wpm():
    for profile, (low_s, high_s) in {"SHORT": (180, 300), "LONG": (1801, 3600)}.items():
        low, high = config.LENGTH_POLICY["profiles"][profile]["accepted_words"]
        assert low * 60 / 170 >= low_s and high * 60 / 150 <= high_s, profile


# ============================================================================================ brief and request
def test_brief_excludes_speakable_compliance_content_and_gold(builder):
    inputs = shared_inputs()
    config_scenario = inputs.configs["CALL_006"]["scenario"]
    body = builder.build("CALL_006", "A001")
    text = body["input"][1]["content"] + body["input"][0]["content"]
    for risk in config_scenario["final_product"]["concrete_risks"]:
        assert risk not in text
    for secret in (config_scenario["final_product"]["recommendation_basis"], config_scenario["red_flags"]["C09"],
                   config_scenario["client"]["investment_objective"], config_scenario["client"]["loss_attitude"]):
        assert secret not in text
    brief = json.loads(body["input"][1]["content"])
    assert set(brief) == {"call", "people", "setting", "holding_context", "final_product", "discussion_topics", "events",
                          "planned_recommendation_event_ids", "placement_constraints", "must_never_be_expressed_outside_events",
                          "edge_cases"}
    assert "question_ids" not in body["input"][1]["content"] and "expected" not in brief
    assert body["max_output_tokens"] == 32768 and body["store"] is False and body["text"]["format"]["strict"] is True
    assert body["metadata"]["stage"] == "G6" and body["model"] == "gpt-6.1-sol"
    assert builder.build("CALL_006", "A001") == body  # deterministic


# ============================================================================================ advisory component (call intents)
C07_FAIL_CALLS = ["CALL_001", "CALL_024", "CALL_031", "CALL_036", "CALL_042", "CALL_047", "CALL_049"]
# natural wordings of the unplanned recommendation: it names the frozen product and action; "recommend" is optional
RECOMMENDATION_SHAPES = [
    "I recommend that you {verb} {product}.",
    "I'd suggest that we {verb} {product}.",
    "My advice is to {verb} {product}.",
    "I would back a decision to {verb} {product}.",
]
VERBS = {"BUY": "buy", "SELL": "sell"}


def squashed(text):
    import re
    return re.sub(r"\s+", " ", text)


def test_the_brief_carries_the_call_intents_unchanged_and_the_planned_recommendation_events_from_the_c07_plan(builder):
    inputs = shared_inputs()
    empty = []
    for call_id in CALL_IDS:
        brief = json.loads(builder.build(call_id, "A001")["input"][1]["content"])
        assert brief["call"]["call_intents"] == inputs.specs[call_id]["metadata"]["call_intents"] == ["ADVISORY", "EXECUTION"]
        carriers = [e["event_id"] for e in inputs.plans[call_id]["planned_events"] if "C07" in e["question_ids"]]
        assert brief["planned_recommendation_event_ids"] == carriers
        gold = next(g["expected"]["expected_verdict"] for g in inputs.specs[call_id]["gold"] if g["question_id"] == "C07")
        assert (not carriers) == (gold == "FAIL")  # C07 FAIL is an absence check: the plan has no recommendation event
        if carriers:
            (event_id,) = carriers
            assert next(e for e in brief["events"] if e["event_id"] == event_id)["speaker"] == "EMPLOYEE"
        else:
            empty.append(call_id)
    assert empty == C07_FAIL_CALLS
    for call_id in CALL_IDS:  # the advisory brief key exists for exactly the calls whose plan has no planned recommendation
        assert ("advisory_recommendation" in json.loads(builder.build(call_id, "A001")["input"][1]["content"])) == (call_id in C07_FAIL_CALLS)
    assert "C07" not in json.dumps(brief)  # event IDs only: question IDs and gold stay out of the brief


def test_the_g6_prompt_asks_for_a_named_rationale_free_recommendation_only_where_the_plan_has_no_recommendation():
    text = squashed((PROMPTS / "dialogue.v1.md").read_text(encoding="utf-8"))
    for required in ("the first one comes first", "ADVISORY and then EXECUTION", "BEFORE the final confirmation exchange",
                     "The client may mention a product, an amount or an intention earlier", "planned_recommendation_event_ids",
                     "those events ARE the recommendation", "do not add any recommendation or endorsement of your own",
                     "When it is empty, the brief has `advisory_recommendation`", "ONE clear employee recommendation",
                     "names the product exactly as `product_name`", "when `direction` is BUY", "when `direction` is SELL",
                     'You do not have to use the word "recommend"', "NO reason, basis, benefit, feature, performance, suitability or comparison",
                     "a listener must not be able to tell WHY", '"because", "since", "given"',
                     "no quantity, amount, currency, price, order type, date or other order detail",
                     "does not copy, repeat or paraphrase any planned event", "After every event in `place_after_event_ids`",
                     "before every event in `place_before_event_ids`", "naming the product earlier would give the answer away",
                     "Not in the same turn as, and not as the employee's immediate reply to, the events in `keep_apart_from_event_ids`",
                     "Preferred, not required", "contains no planned question", "begins with a mild acknowledgement",
                     '"Okay, noted." or "Okay, thanks."', "is not an instruction, a commitment, agreement to proceed or a confirmation",
                     'It never says "I understand"', "The only exceptions to these rules are the single recommendation and the client's mild acknowledgement"):
        assert required in text, required
    assert "must use the word" not in text and "fixed sentence" not in text and "[[ADV" not in text
    assert "Do not restate the product name" not in text and "It must only recommend or endorse" not in text  # the old endorsement rule is gone
    for kept in ("explain any risk, possible loss, fee, charge or reason for a recommendation", "promise or guarantee any return",
                 "must_never_be_expressed_outside_events", "ask whether the client understands any product, risk or disclosure"):
        assert kept in text  # the unrelated prohibitions survive


def confirmation_turn(ctx, response):
    """Index of the first model turn that carries a C08 event marker."""
    c08 = [e["event_id"] for e in ctx.events if "C08" in e["question_ids"]]
    return min(turn_of(response, event_id) for event_id in c08)


@pytest.mark.parametrize("call_id", C07_FAIL_CALLS)
@pytest.mark.parametrize("shape", RECOMMENDATION_SHAPES)
def test_an_isolated_recommendation_fixture_is_structurally_accepted_and_preserves_every_frozen_event(builder, validator, call_id, shape):
    ctx = builder.context(call_id)
    sentence = shape.format(verb=VERBS[ctx.advisory["direction"]], product=ctx.advisory["product_name"])
    base = synthesize_dialogue(ctx)
    before, _, _ = render(ctx, base)
    first = confirmation_turn(ctx, base)
    index = max(i for i, t in enumerate(base["turns"][:first]) if t["speaker"] == "EMPLOYEE")
    response = json.loads(json.dumps(base))
    response["turns"][index]["text"] = response["turns"][index]["text"].rstrip() + " " + sentence
    response["turns"][index + 1]["text"] = "Okay, noted. " + response["turns"][index + 1]["text"]  # the client's mild acknowledgement
    assert validator.errors(response, ctx) == []  # structure only: acceptance does not judge advisory meaning
    turns, alignment, errors = render(ctx, response)
    assert errors == [] and index < first
    assert {s["event_id"]: s["turn_id"] for s in alignment} == {s["event_id"]: s["turn_id"] for s in render(ctx, base)[1]}
    assert all(turns[s["turn_id"]]["text"][s["start"]:s["end"]] == ctx.texts[s["event_id"]] for s in alignment)  # frozen text is exact
    assert [t["speaker"] for t in turns] == [t["speaker"] for t in before]
    generation = {"attempt_id": "A001", "model": config.MODEL, "params": {"reasoning_effort": "medium", "max_output_tokens": 32768,
                  "length_policy": config.LENGTH_POLICY}, "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
    artifact = compile_artifact(shared_inputs(), ctx, response, generation, builder.response_schema)
    assert sentence in artifact["turns"][index]["text"]
    # naming the product outside the events is never a rejection: it is recorded for G7 (acceptance stays structural)
    assert {n["surface"] for n in artifact["review_notes"]} <= {ctx.advisory["product_name"], VERBS[ctx.advisory["direction"]]}  # product and action only
    assert [a["event_id"] for a in artifact["event_alignment"]] == [a["event_id"] for a in sorted(alignment, key=lambda s: (s["turn_id"], s["start"]))]


def test_a_recommendation_that_echoes_planned_text_is_still_rejected_and_naming_the_product_is_only_a_review_note(builder, validator):
    ctx = builder.context("CALL_001")
    base = synthesize_dialogue(ctx)
    index = max(i for i, t in enumerate(base["turns"][:confirmation_turn(ctx, base)]) if t["speaker"] == "EMPLOYEE")
    naming = json.loads(json.dumps(base))
    naming["turns"][index]["text"] += " I recommend that you buy Nova Technology Equity Fund."
    assert validator.errors(naming, ctx) == []  # accepted structurally, but G7 gets the review note
    generation = {"attempt_id": "A001", "model": config.MODEL, "params": {"reasoning_effort": "medium", "max_output_tokens": 32768,
                  "length_policy": config.LENGTH_POLICY}, "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
    notes = compile_artifact(shared_inputs(), ctx, naming, generation, builder.response_schema)["review_notes"]
    assert any(n["question_id"] == "T01" and n["surface"] == "Nova Technology Equity Fund" for n in notes)
    echoing = json.loads(json.dumps(base))
    echoing["turns"][index]["text"] += " " + ctx.texts[next(e["event_id"] for e in ctx.events if "C08" in e["question_ids"])]
    assert any("repeated outside its event marker" in e for e in validator.errors(echoing, ctx))


def test_other_unrelated_prohibitions_are_not_weakened_by_the_recommendation_exception(builder):
    """The brief still hides every speakable compliance fact an endorsement could leak, and still lists the absence conditions."""
    brief = json.loads(builder.build("CALL_001", "A001")["input"][1]["content"])
    scenario = shared_inputs().configs["CALL_001"]["scenario"]
    text = json.dumps(brief)
    assert scenario["final_product"]["recommendation_basis"] not in text and scenario["client"]["investment_objective"] not in text
    assert any("concrete rationale for recommending the product" in c for c in brief["must_never_be_expressed_outside_events"])


# ============================================================================================ G5 fingerprint reader
def test_g5_fingerprint_distinguishes_present_absent_and_unreadable(tmp_path, monkeypatch):
    present = g5_report_fingerprint(RELEASE, "CALL_001")
    assert isinstance(present, str) and len(present) == 64
    assert g5_report_fingerprint(tmp_path, "CALL_001") is None  # confirmed absence

    def denied(self):
        raise PermissionError("access denied")
    monkeypatch.setattr("pathlib.Path.read_bytes", denied)
    assert g5_report_fingerprint(RELEASE, "CALL_001") == config.G5_UNREADABLE  # never mistaken for deletion
