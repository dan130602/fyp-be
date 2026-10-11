"""G3 core: product references, required slots, feasibility over all 50 frozen calls, request, validator, compile."""

import json
import re
import subprocess
import sys
from copy import deepcopy

import pytest

from fyp_evaluation.call_configs.compile import canonical_response, compile_artifact
from fyp_evaluation.call_configs.feasibility import feasibility_report, make_validator, synthesize_response
from fyp_evaluation.call_configs.products import derive_products
from fyp_evaluation.call_configs.request import RequestBuilder, request_hash
from fyp_evaluation.call_configs.slots import CallContext
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.llm.store import canonical_request_text
from g3_support import CALL_IDS, SCHEMAS, shared_inputs


@pytest.fixture(scope="module")
def inputs():
    return shared_inputs()


@pytest.fixture(scope="module")
def builder(inputs):
    return RequestBuilder(inputs, SCHEMAS)


@pytest.fixture(scope="module")
def validator(builder):
    return make_validator(builder)


def products_of(builder, call_id):
    return [(p["product_ref"], p["name"], p["roles"]) for p in builder.context(call_id).products]


# ------------------------------------------------------------------------------------------ products
def test_call_005_final_and_candidate_overlap(builder):
    assert products_of(builder, "CALL_005") == [
        ("P1", "Meridian Asia Equity Fund", ["FINAL", "CANDIDATE"]), ("P2", "Nova Technology Equity Fund", ["CANDIDATE"])]


def test_call_011_final_and_initial_order_overlap(builder):
    assert products_of(builder, "CALL_011") == [("P1", "Pioneer Balanced Opportunities Fund", ["FINAL", "INITIAL_ORDER"])]


def test_call_041_final_and_superseded(builder):
    assert products_of(builder, "CALL_041") == [
        ("P1", "Meridian Asia Equity Fund", ["FINAL"]), ("P2", "Summit Global Equity ETF", ["SUPERSEDED"])]


def test_product_refs_are_sequential_deduplicated_and_start_with_the_final_product(inputs):
    for call_id in CALL_IDS:
        products = derive_products(inputs.specs[call_id], inputs.product_category)
        assert [p["product_ref"] for p in products] == [f"P{n}" for n in range(1, len(products) + 1)]
        assert len({p["name"] for p in products}) == len(products)
        gold = next(g for g in inputs.specs[call_id]["gold"] if g["question_id"] == "T01")["expected"]["expected_value"]
        assert products[0]["name"] == gold and "FINAL" in products[0]["roles"]
        assert sum("FINAL" in p["roles"] for p in products) == 1


def test_first_seen_order_follows_assignment_then_parameter_order_never_alphabetical(inputs):
    spec = deepcopy(inputs.specs["CALL_001"])
    spec["gold"] = [dict(g) for g in spec["gold"]]
    final = "Orion Global Income Fund"
    next(g for g in spec["gold"] if g["question_id"] == "T01")["expected"] = {"kind": "extraction", "value_type": "text", "expected_value": final}
    spec["edge_cases"] = [
        {"assignment_id": "EA_901", "edge_case_id": "EC_PRODUCT_REFERENCE", "family": "REFERENCE_RESOLUTION", "target_question_ids": ["T01"],
         "parameters": {"antecedent_speaker": "CLIENT", "candidate_products": ["Summit Global Equity ETF", final, "Atlas USD Corporate Bond 2031"],
                        "selected_product": final}},
        {"assignment_id": "EA_902", "edge_case_id": "EC_MULTIPLE_CORRECTIONS", "family": "CORRECTION_SUPERSESSION", "target_question_ids": ["T01"],
         "parameters": {"superseded_values": ["Crest Emerging Markets Bond Fund", "Summit Global Equity ETF"], "final_value": final}},
    ]
    products = derive_products(spec, inputs.product_category)
    assert [(p["name"], p["roles"]) for p in products] == [
        (final, ["FINAL", "CANDIDATE"]), ("Summit Global Equity ETF", ["CANDIDATE", "SUPERSEDED"]),
        ("Atlas USD Corporate Bond 2031", ["CANDIDATE"]), ("Crest Emerging Markets Bond Fund", ["SUPERSEDED"])]
    assert derive_products(spec, inputs.product_category) == products


def test_unmapped_t01_edge_cases_and_parameters_fail_loudly(inputs):
    spec = deepcopy(inputs.specs["CALL_001"])
    spec["edge_cases"] = [{"assignment_id": "EA_901", "edge_case_id": "EC_DOUBLE_TRIPLE_DIGITS", "family": "X",
                           "target_question_ids": ["T01"], "parameters": {}}]
    with pytest.raises(ValueError, match="no product mapping"):
        derive_products(spec, inputs.product_category)
    spec["edge_cases"] = [{"assignment_id": "EA_902", "edge_case_id": "EC_FALSE_START", "family": "X",
                           "target_question_ids": ["T01"], "parameters": {"final_value": "Orion Global Income Fund", "surprise": "x"}}]
    with pytest.raises(ValueError, match="unmapped parameters"):
        derive_products(spec, inputs.product_category)


def test_final_product_contradicting_gold_fails(inputs):
    spec = deepcopy(inputs.specs["CALL_005"])
    spec["edge_cases"][0]["parameters"]["selected_product"] = "Orion Global Income Fund"
    with pytest.raises(ValueError, match="contradicts gold"):
        derive_products(spec, inputs.product_category)


# ---------------------------------------------------------------------------------------------- slots
def test_slot_counts_match_the_frozen_gold(inputs, builder):
    fee = red9 = red10 = holding = 0
    for call_id in CALL_IDS:
        gold = {g["question_id"]: g["expected"] for g in inputs.specs[call_id]["gold"]}
        slots = builder.context(call_id).required_slots()
        assert slots["fee_disclosure"]["required"] == (gold["C04"]["expected_verdict"] == "PASS")
        assert slots["red_flags"]["C09"]["required"] == (gold["C09"]["expected_verdict"] == "FAIL")
        assert slots["red_flags"]["C10"]["required"] == (gold["C10"]["expected_verdict"] == "FAIL")
        sells = {gold["T05"]["expected_value"]}
        for edge in inputs.specs[call_id]["edge_cases"]:
            if edge["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT":
                sells.add(edge["parameters"]["initial_order"]["T05"])
        assert slots["holding_context"]["required"] == ("SELL" in sells)
        limit_final = gold["T06"]["expected_value"] == "LIMIT"
        assert slots["limit_prices"]["final"]["required"] == limit_final
        fee += slots["fee_disclosure"]["required"]
        red9 += slots["red_flags"]["C09"]["required"]
        red10 += slots["red_flags"]["C10"]["required"]
        holding += slots["holding_context"]["required"]
    assert (fee, red9, red10) == (42, 7, 7)
    assert holding >= 25


def test_initial_limit_price_is_required_only_for_a_limit_initial_order(inputs, builder):
    for call_id in CALL_IDS:
        replacement = next((e for e in inputs.specs[call_id]["edge_cases"] if e["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT"), None)
        initial = builder.context(call_id).required_slots()["limit_prices"]["initial"]
        expected = bool(replacement and replacement["parameters"]["initial_order"]["T06"] == "LIMIT")
        assert initial["required"] == expected


def test_fee_amount_currencies_include_the_fee_currency_distractor_only_when_assigned(inputs, builder):
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        distractors = [e["parameters"]["distractor_currency"] for e in inputs.specs[call_id]["edge_cases"]
                       if e["edge_case_id"] == "EC_MULTIPLE_CURRENCIES" and e["parameters"]["distractor_role"] == "FEE_CURRENCY"]
        assert set(ctx.fee_currencies) == {ctx.value("T02"), *distractors}


def test_edge_context_excludes_the_spoken_numeric_family_long_distance_and_paraphrase(inputs, builder):
    for call_id in CALL_IDS:
        expected = [e["assignment_id"] for e in inputs.specs[call_id]["edge_cases"]
                    if e["family"] != "SPOKEN_NUMERIC_NORMALISATION"
                    and e["edge_case_id"] not in ("EC_LONG_DISTANCE_DEPENDENCY", "EC_PARAPHRASED_COMPLIANCE_STATEMENT")]
        assert builder.context(call_id).edge_context_ids == expected


# -------------------------------------------------------------------------------------- feasibility
def test_every_frozen_call_has_a_valid_synthetic_response(builder, validator, inputs):
    report = feasibility_report(builder, validator, inputs)
    assert len(report) == 50
    assert {call_id: errors for call_id, errors in report.items() if errors} == {}


def test_amount_fee_variant_is_valid_wherever_a_fee_is_required(builder, validator):
    checked = 0
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        if ctx.fee_required:
            assert validator.errors(synthesize_response(ctx, fee_basis="AMOUNT"), ctx) == []
            checked += 1
    assert checked == 42


def _wrong_nullness(response, ctx):
    """Every single-slot nullness flip that must be rejected for this call."""
    flips = {}
    flips["holding_context"] = {**response, "holding_context": None if ctx.holding_required else "A holding."}
    flips["fee_disclosure"] = {**response, "fee_disclosure": None if ctx.fee_required else {"basis": "RATE", "value": 0.5, "currency": None}}
    for key in ("final", "initial"):
        required = bool(ctx.order_states.get(key) and ctx.order_states[key]["limit_required"])
        flips[f"limit_{key}"] = {**response, "limit_prices": {**response["limit_prices"], key: None if required else 12.5}}
    for q, required in ctx.red_flag_required.items():
        flips[f"red_{q}"] = {**response, "red_flags": {**response["red_flags"], q: None if required else "A flag."}}
    return flips


def test_flipping_any_conditional_slot_is_rejected_for_all_50_calls(builder, validator):
    for call_id in CALL_IDS:
        ctx = builder.context(call_id)
        for name, mutated in _wrong_nullness(synthesize_response(ctx), ctx).items():
            assert any("required" in e or "must be null" in e for e in validator.errors(mutated, ctx)), (call_id, name)


# ------------------------------------------------------------------------------------ brief/request
def test_requests_are_byte_identical_across_builders_and_exclude_the_skeleton_body(inputs, builder):
    other = RequestBuilder(inputs, SCHEMAS)
    for call_id in CALL_IDS:
        first, second = builder.build(call_id, "A001"), other.build(call_id, "A001")
        assert canonical_request_text(first) == canonical_request_text(second)
        assert request_hash(first) == request_hash(second)
        text = canonical_request_text(first)
        for forbidden in ("planned_events", "EV_0", "allowed_speakers", "MIN_INTERVENING", "check_plan_skeleton", "evidence_event_ids"):
            assert forbidden not in text, (call_id, forbidden)



def test_brief_never_exposes_kyc_values_or_spoken_numeric_parameters(inputs, builder):
    for call_id in CALL_IDS:
        gold = {g["question_id"]: g["expected"]["expected_value"] for g in inputs.specs[call_id]["gold"] if g["question_id"] in ("K01", "K02")}
        text = builder.build(call_id, "A001")["input"][1]["content"]
        assert gold["K01"] not in text and gold["K02"] not in text
        for edge in inputs.specs[call_id]["edge_cases"]:
            if edge["family"] == "SPOKEN_NUMERIC_NORMALISATION":
                assert edge["assignment_id"] not in text


def test_request_settings_are_the_approved_ones(builder):
    body = builder.build("CALL_001", "A002")
    assert body["model"] == "gpt-6.1-sol" and body["reasoning"] == {"effort": "medium"}
    assert body["max_output_tokens"] == 16000 and body["store"] is False
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "g3_call_config_v1"
    assert not {"tools", "tool_choice", "temperature", "top_p"} & set(body)
    assert body["metadata"] == {"dataset_id": "advisory_execution_pilot_v1", "call_id": "CALL_001", "stage": "G3", "attempt_id": "A002"}
    assert [m["role"] for m in body["input"]] == ["developer", "user"]
    assert "{{" not in body["input"][0]["content"]


def test_developer_prefix_is_shared_and_holds_the_checklist(builder):
    assert builder.build("CALL_001", "A001")["input"][0] == builder.build("CALL_050", "A001")["input"][0]
    prefix = builder.developer_message
    for question in ("C01 (COMPLIANCE)", "T06 (TRANSACTION)", "PASS when:", "Resolution:", "C03_AFTER_C02_DISCLOSURE"):
        assert question.split(" ")[0] in prefix
    assert "scenario fact is not dialogue evidence" in prefix.lower()


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
    assert len(builder.response_schema["properties"]) == 11


# ------------------------------------------------------------------------------------------ validator
def valid_for(builder, call_id, **kwargs):
    ctx = builder.context(call_id)
    return ctx, synthesize_response(ctx, **kwargs)


def call_where(builder, predicate):
    return next(c for c in CALL_IDS if predicate(builder.context(c)))


@pytest.mark.parametrize("mutation, expected", [
    (lambda r: r.update(extra="x"), "schema"),
    (lambda r: r.pop("client"), "schema"),
    (lambda r: r["client"].update(name="  "), "must not be empty"),
    (lambda r: r["final_product"].update(concrete_risks=[]), "needs at least one risk"),
    (lambda r: r["client"].update(name=3), "schema"),
])
def test_schema_and_emptiness_rules(builder, validator, mutation, expected):
    ctx, response = valid_for(builder, "CALL_001")
    mutation(response)
    assert any(expected in e for e in validator.errors(response, ctx))


def test_product_reference_and_edge_context_sets_must_equal_python_sets(builder, validator):
    call_id = call_where(builder, lambda c: c.other_product_refs and c.edge_context_ids)
    ctx, response = valid_for(builder, call_id)
    wrong = deepcopy(response)
    wrong["other_products"][0]["product_ref"] = "P9"
    assert any("product_ref set" in e for e in validator.errors(wrong, ctx))
    wrong = deepcopy(response)
    wrong["other_products"].append(dict(wrong["other_products"][0]))
    assert any("product_ref set" in e for e in validator.errors(wrong, ctx))
    wrong = deepcopy(response)
    wrong["edge_case_context"].pop()
    assert any("assignment_id set" in e for e in validator.errors(wrong, ctx))
    wrong = deepcopy(response)
    wrong["edge_case_context"].append({"assignment_id": "EA_999", "motivation": "invented"})
    assert any("assignment_id set" in e for e in validator.errors(wrong, ctx))


def test_fee_disclosure_rules(builder, validator):
    call_id = call_where(builder, lambda c: c.fee_required)
    ctx, response = valid_for(builder, call_id)
    cases = {
        "must be positive": {"basis": "RATE", "value": -1, "currency": None},
        "must be null for a RATE": {"basis": "RATE", "value": 1, "currency": "SGD"},
        "not in": {"basis": "AMOUNT", "value": 40, "currency": "JPY" if ctx.fee_currencies[0] != "JPY" else "USD"},
        "equals an order amount": {"basis": "AMOUNT", "value": ctx.order_states["final"]["amount"], "currency": ctx.order_states["final"]["currency"]},
    }
    for message, fee in cases.items():
        assert any(message in e for e in validator.errors({**response, "fee_disclosure": fee}, ctx)), message


def test_limit_price_rules(builder, validator):
    call_id = call_where(builder, lambda c: c.order_states["final"]["limit_required"])
    ctx, response = valid_for(builder, call_id)
    state = ctx.order_states["final"]
    for price, message in ((-1, "must be positive"), (state["max_unit_price"] + 1, "outside the"), (state["min_unit_price"] / 2, "outside the")):
        mutated = {**response, "limit_prices": {**response["limit_prices"], "final": price}}
        assert any(message in e for e in validator.errors(mutated, ctx)), price
    ctx.monetary_values.append((state["currency"], response["limit_prices"]["final"]))  # an order amount equal to the price
    assert any("equals an order amount" in e for e in validator.errors(response, ctx))


def test_exact_numeric_equality_alone_is_not_an_error(builder, validator):
    """A limit price equal to a TRANSACTION QUANTITY is allowed: only monetary collisions are ambiguous."""
    call_id = call_where(builder, lambda c: c.order_states["final"]["limit_required"] and c.order_states["final"]["min_unit_price"] <= 100 <= c.order_states["final"]["max_unit_price"])
    ctx, response = valid_for(builder, call_id)
    ctx.order_states["final"]["quantity"] = 100
    mutated = {**response, "limit_prices": {**response["limit_prices"], "final": 100}}
    assert validator.errors(mutated, ctx) == []


def lexical_errors(builder, validator, call_id, text, field="background"):
    ctx, response = valid_for(builder, call_id)
    response["client"][field] = text
    return validator.errors(response, ctx)


def test_lexical_scans(builder, validator):
    ctx = builder.context("CALL_005")
    outsider = next(n for n in ctx.product_pool if n not in {p["name"] for p in ctx.products})
    forbidden = next(c for c in ctx.currency_codes if c not in ctx.allowed_currencies)
    alias = next(a for a in ctx.currency_aliases[forbidden] if a != forbidden)
    assert any("not part of this call" in e for e in lexical_errors(builder, validator, "CALL_005", f"They also like {outsider}."))
    assert any("not part of this call" in e for e in lexical_errors(builder, validator, "CALL_005", f"Holds {outsider.upper()} units."))
    assert any(forbidden in e for e in lexical_errors(builder, validator, "CALL_005", f"Funds sit in {forbidden} accounts."))
    assert any(alias in e for e in lexical_errors(builder, validator, "CALL_005", f"Funds sit in {alias} accounts."))
    assert not any(forbidden in e for e in lexical_errors(builder, validator, "CALL_005", f"The {forbidden.lower()}x word is fine {alias}ish."))
    assert any("NRIC" in e for e in lexical_errors(builder, validator, "CALL_005", "ID is S1234567A today."))
    for text in ("About EUR 5,000 was moved.", "They moved 5,000 euros.", "Around $ 50 was spent.", "Paid S$20."):
        assert any("currency-plus-number" in e for e in lexical_errors(builder, validator, "CALL_005", text)), text
    assert not lexical_errors(builder, validator, "CALL_005", "The client keeps euros in a savings account.")


def test_allowed_currency_without_a_number_is_fine(builder, validator):
    ctx = builder.context("CALL_005")
    code = ctx.allowed_currencies[0]
    assert lexical_errors(builder, validator, "CALL_005", f"A {code} account is held.") == []


def test_structured_values_are_not_restated_in_narrative_but_small_integers_are_ignored(builder, validator):
    call_id = call_where(builder, lambda c: c.order_states["final"]["amount"] >= 100000)
    ctx = builder.context(call_id)
    amount = ctx.order_states["final"]["amount"]
    for form in (str(amount), f"{amount:,}"):
        assert any("restates structured value" in e for e in lexical_errors(builder, validator, call_id, f"The client has about {form} in mind."))
    assert not any("restates" in e for e in lexical_errors(builder, validator, call_id, "A track record of 12 years and 100 meetings."))


def test_authorized_product_names_are_exempt_from_currency_and_number_detection(builder, validator):
    call_id = next(c for c in CALL_IDS if any(p["name"] == "Atlas USD Corporate Bond 2031" for p in builder.context(c).products))
    ctx = builder.context(call_id)
    assert "USD" in ctx.currency_codes
    errors = lexical_errors(builder, validator, call_id, "The client likes Atlas USD Corporate Bond 2031 for its steady coupons.")
    assert not any("currency" in e or "restates" in e for e in errors)
    other = next(c for c in CALL_IDS if not any(p["name"] == "Atlas USD Corporate Bond 2031" for p in builder.context(c).products))
    assert any("not part of this call" in e for e in lexical_errors(builder, validator, other, "Considering Atlas USD Corporate Bond 2031."))


def test_currency_aliases_match_on_word_boundaries_only(builder, validator):
    call_id = next(c for c in CALL_IDS if "JPY" not in builder.context(c).allowed_currencies)
    assert not any("yen" in e for e in lexical_errors(builder, validator, call_id, "The ryentex committee met."))
    assert any("yen" in e for e in lexical_errors(builder, validator, call_id, "Thinking about the yen exposure."))


# -------------------------------------------------------------------------------------------- compile
def test_artifact_envelope_holds_no_gold_and_no_edge_parameters(builder, inputs):
    ctx = builder.context("CALL_005")
    generation = {"attempt_id": "A001", "model": "gpt-6.1-sol", "prompt_template_sha256": "1" * 64, "response_schema_sha256": "2" * 64}
    artifact = compile_artifact(inputs, ctx, synthesize_response(ctx), generation, builder.response_schema)
    text = serialize_json(artifact)
    for forbidden in ("expected_verdict", "expected_value", '"gold"', "candidate_products", "edge_cases", "S0859609E", "27 October 1991"):
        assert forbidden not in text
    assert artifact["call_spec_sha256"] == inputs.spec_hashes["CALL_005"]
    assert artifact["check_plan_skeleton_sha256"] == inputs.skeleton_hashes["CALL_005"]
    assert list(artifact) == ["schema_version", "stage", "dataset_id", "call_id", "call_spec_sha256", "check_plan_skeleton_sha256",
                              "source", "generation", "products", "scenario"]
    assert artifact["products"][0]["roles"] == ["FINAL", "CANDIDATE"]


def test_scenario_key_order_is_canonical_regardless_of_the_provider_order(builder):
    ctx = builder.context("CALL_001")
    response = synthesize_response(ctx)
    scrambled = json.loads(json.dumps(response, sort_keys=True))
    assert list(canonical_response(scrambled, builder.response_schema)) == list(builder.response_schema["properties"])
    assert serialize_json(canonical_response(scrambled, builder.response_schema)) == serialize_json(canonical_response(response, builder.response_schema))


def test_importing_the_cli_does_not_import_the_openai_sdk():
    code = "import sys, fyp_evaluation.call_configs.cli, fyp_evaluation.call_configs.run; assert 'openai' not in sys.modules"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
