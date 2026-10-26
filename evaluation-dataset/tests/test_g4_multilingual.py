"""Schema-2.0 (multilingual) G4: surface catalog, shared form choices, guards, scenario slots, schemas and offline compilation.

Everything runs offline over artificial G3 configs (`g4_multilingual_support`); the production G3 refusal for the
multilingual release is untouched and tested in `test_multilingual_planning`. The Mandarin expectations in this file are
PROVISIONAL (code-test completion, not linguistic acceptance): native-speaker review is still outstanding.
"""

import json
import re
import shutil
from copy import deepcopy
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.check_plans import guards
from fyp_evaluation.check_plans import render as R
from fyp_evaluation.check_plans.analysis import _round_trip, scenario_surface
from fyp_evaluation.check_plans.compile import build_record, compile_artifact, get_lexicon, project_check_plan_to_skeleton
from fyp_evaluation.check_plans.config import QUESTION_PLACEHOLDERS
from fyp_evaluation.check_plans.context import PlanContext, PlanInputError
from fyp_evaluation.check_plans.feasibility import make_validator, synthesize_response
from fyp_evaluation.check_plans.request import RequestBuilder, _surfaces
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.check_plans.surfaces import SurfaceCatalog, english_decimal, has_han
from fyp_evaluation.check_plans.template import WEAK_ACK_BANNED
from fyp_evaluation.planning import outputs as g1
from g4_multilingual_support import PROFILES, RELEASE, ROOT, SCHEMAS, calls_of, plan_inputs, upstream_inputs

HAN = re.compile(r"[一-鿿〇]")
GENERATION = {"attempt_id": "A001", "model": "fixture", "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}


@pytest.fixture(scope="module")
def inputs():
    return plan_inputs()


@pytest.fixture(scope="module")
def builder(inputs):
    return RequestBuilder(inputs, SCHEMAS)


@pytest.fixture(scope="module")
def validator(builder):
    return make_validator(builder)


@pytest.fixture(scope="module")
def catalog(inputs):
    return SurfaceCatalog(inputs.documents["normalization"])


@pytest.fixture(scope="module")
def compiled(inputs, builder):
    """{call_id: (ctx, plan)}: the synthetic response of every multilingual call, compiled."""
    result = {}
    for call_id in inputs.ready_call_ids:
        ctx = builder.context(call_id)
        gen = {**GENERATION, "response_schema_sha256": builder.response_schema_sha256}
        result[call_id] = (ctx, compile_artifact(inputs, ctx, synthesize_response(ctx), gen, builder.response_schema))
    return result


def profile_of(call_id):
    return next(p for p in PROFILES if call_id in calls_of(p))


def find_event(profile, kind, *, unpinned=True):
    """(ctx, event) of the first call of the profile with an event of this kind (and no language pin when asked)."""
    inputs = plan_inputs()
    for call_id in calls_of(profile):
        ctx = PlanContext(inputs, call_id)
        for event in ctx.events:
            if event["kind"] == kind and not (unpinned and event.get("required_language")):
                return ctx, event
    raise AssertionError(f"no {kind} event in {profile}")


def errors_with_text(validator, ctx, event, text):
    response = synthesize_response(ctx)
    entry = next(e for e in response["events"] if e["event_id"] == event["event_id"])
    entry["text"] = text
    return validator.errors(response, ctx)


# ============================================================================================ surface catalog
def test_currency_forms_exist_only_where_the_value_has_a_surface(catalog):
    assert [c for c in ("SGD", "USD", "EUR", "CNY", "JPY") if catalog.form_available("T02", c, "SPELLED_CODE")] == ["SGD", "USD", "CNY"]
    assert catalog.surface_options("T02", "CNY", "SPELLED_CODE") == ["RMB"]
    assert catalog.surface_options("T02", "USD", "NATURAL_NAME") == ["US dollars", "American dollars"]
    assert catalog.surface_options("T02", "JPY", "ZH_NATURAL_NAME") == ["日元"]
    assert not catalog.form_available("T02", "JPY", "SPELLED_CODE") and not catalog.form_available("T02", "EUR", "SPELLED_CODE")


def test_english_generation_never_uses_bare_yuan_yen_or_spelled_euro_yen_codes(catalog):
    surfaces = {s for c in ("SGD", "USD", "EUR", "CNY", "JPY") for f in ("NATURAL_NAME", "SPELLED_CODE") for s in (catalog.surface_options("T02", c, f) or [])}
    assert not surfaces & {"yuan", "yen", "EUR", "JPY", "CNY", "E-U-R", "J-P-Y", "C-N-Y"}
    assert "Chinese yuan" in surfaces and catalog.surface_options("T02", "JPY", "NATURAL_NAME") == ["Japanese yen"]


def test_surface_choice_within_a_form_is_deterministic_and_varies_across_calls(catalog):
    first = catalog.render_question("CALL_001", "T02", "USD", "NATURAL_NAME")
    assert first == catalog.render_question("CALL_001", "T02", "USD", "NATURAL_NAME")
    seen = {catalog.render_question(f"CALL_{n:03d}", "T02", "USD", "NATURAL_NAME") for n in range(1, 30)}
    assert seen == {"US dollars", "American dollars"}
    zh = {catalog.render_question(f"CALL_{n:03d}", "T02", "SGD", "ZH_NATURAL_NAME") for n in range(1, 40)}
    assert zh == {"新币", "新元", "新加坡元"}


@pytest.mark.parametrize("surface,code", [("人民币", "CNY"), ("RMB", "CNY"), ("renminbi", "CNY"), ("Chinese yuan", "CNY"), ("yuan", "CNY"),
                                          ("yen", "JPY"), ("Japanese yen", "JPY"), ("日元", "JPY"), ("新币", "SGD"), ("美金", "USD"),
                                          ("US dollars", "USD"), ("euros", "EUR"), ("欧元", "EUR")])
def test_currency_decoder_accepts_the_approved_vocabulary(catalog, surface, code):
    assert catalog.decode_question("T02", surface) == code


@pytest.mark.parametrize("surface", ["dollars", "dollar", "元", "块", "刀", "美", "", "pesos", "新", "人民"])
def test_currency_decoder_rejects_standalone_and_unknown_expressions(catalog, surface):
    with pytest.raises(R.RenderError):
        catalog.decode_question("T02", surface)


def test_an_alias_that_maps_to_two_currencies_is_rejected_not_guessed(inputs):
    document = deepcopy(inputs.documents["normalization"])
    document["profiles"]["currency_iso_4217_alpha3"]["aliases"]["JPY"].append("新币")
    with pytest.raises(R.RenderError, match="ambiguous"):
        SurfaceCatalog(document).decode_question("T02", "新币")


def test_product_names_decode_to_their_canonical_identity_and_unknown_aliases_fail(catalog):
    names = catalog.profiles["synthetic_product_name"]["language_names"]
    assert len(names) == 8
    for canonical, languages in names.items():
        assert catalog.decode_question("T01", languages["zh"]["surface"]) == canonical
        assert catalog.decode_question("T01", canonical) == canonical
        assert catalog.render_question("CALL_001", "T01", canonical, "ZH_FULL_PRODUCT_NAME") == languages["zh"]["surface"]
    for bad in ("猎户座基金", "猎户座环球基金", "Orion Fund", ""):
        with pytest.raises(R.RenderError):
            catalog.decode_question("T01", bad)
    assert not catalog.form_available("T01", "Not A Product", "ZH_FULL_PRODUCT_NAME")


@pytest.mark.parametrize("surface,question,value", [("买入", "T05", "BUY"), ("买", "T05", "BUY"), ("卖出", "T05", "SELL"), ("sell", "T05", "SELL"),
                                                    ("市价单", "T06", "MARKET"), ("限价单", "T06", "LIMIT"), ("a limit order", "T06", "LIMIT")])
def test_direction_and_order_decoders_use_the_declared_aliases(catalog, surface, question, value):
    assert catalog.decode_question(question, surface) == value


@pytest.mark.parametrize("question,surface", [("T05", "不买"), ("T05", "买入卖出"), ("T06", "不要限价单"), ("T06", "order")])
def test_a_negated_or_unknown_direction_or_order_type_never_decodes(catalog, question, surface):
    with pytest.raises(R.RenderError):
        catalog.decode_question(question, surface)


def test_decoders_choose_by_script_and_never_see_the_expected_value(catalog):
    assert catalog.decode_question("K01", "S八七七九二一九E") == "S8779219E"
    assert catalog.decode_question("K01", "S eight seven seven nine two one nine E") == "S8779219E"
    assert catalog.decode_question("K02", "一九六七年一月五日") == catalog.decode_question("K02", "5 January 1967") == "05 January 1967"
    assert catalog.decode_question("T03", "一百二十万") == catalog.decode_question("T03", "1,200,000") == 1_200_000
    assert catalog.decode_question("T04", "七百") == 700
    with pytest.raises(R.RenderError):
        catalog.decode_question("T03", "1200000元")
    with pytest.raises(R.RenderError):
        catalog.decode_question("T03", "一千二")


def test_composite_money_and_rates_round_trip_and_expose_the_currency(catalog):
    surface = catalog.render_money("CALL_001", Decimal("37.5"), "SGD", "zh")
    assert surface in {"三十七点五新币", "三十七点五新元", "三十七点五新加坡元"}
    assert catalog.decode_money(surface) == (Decimal("37.5"), "SGD")
    assert catalog.render_money("CALL_001", 200, "JPY", "en") == "200 Japanese yen"
    assert catalog.decode_money("200 Japanese yen") == (Decimal(200), "JPY")
    assert catalog.decode_money("一千二百日元") == (Decimal(1200), "JPY")
    assert catalog.render_rate(0.75, "en") == "0.75%" and catalog.render_rate(0.75, "zh") == "百分之零点七五"
    assert catalog.decode_rate("0.75%") == catalog.decode_rate("百分之零点七五") == Decimal("0.75")
    for bad in ("日元", "三十七点五", "37.5", "37.5 dollars", "三十七点五元", "三十七点五块"):
        with pytest.raises(R.RenderError):
            catalog.decode_money(bad)


def test_exact_decimal_rendering_never_rounds_silently(catalog):
    assert english_decimal(Decimal("1234.5"), 2) == "1,234.5" and english_decimal(200, 2) == "200"
    with pytest.raises(R.RenderError, match="decimal places"):
        english_decimal(Decimal("12.125"), 2)
    with pytest.raises(R.RenderError, match="decimal places"):
        catalog.render_money("CALL_001", Decimal("12.125"), "SGD", "zh")
    with pytest.raises(R.RenderError, match="decimal places"):
        catalog.render_rate(Decimal("0.12345"), "en")


# ============================================================================================ form choices
@pytest.fixture(scope="module")
def slot_facts(inputs, builder):
    """Every (call, event, placeholder) a multilingual event may use, with its shared choices."""
    facts = []
    for call_id in inputs.ready_call_ids:
        ctx = builder.context(call_id)
        for event in ctx.events:
            rules = ctx.slot_rules(event)
            names = list(rules.required + rules.optional)
            if rules.readback:
                names += ctx.readback_candidates(event)
            for name in names:
                facts.append((ctx, event, name, ctx.choices(event, name)))
    return facts


def test_every_slot_of_all_250_calls_keeps_at_least_one_valid_choice(slot_facts):
    assert len(slot_facts) > 3000
    assert all(choices for _, _, _, choices in slot_facts)


def test_choices_respect_pins_speaker_languages_and_value_availability(slot_facts):
    for ctx, event, name, choices in slot_facts:
        allowed = ctx.allowed_languages(event)
        assert {c.language for c in choices} <= allowed, (ctx.call_id, event["event_id"], name)
        if event.get("required_language"):
            assert {c.language for c in choices} == {event["required_language"]}
        if name == "T02":
            value = ctx.binding(event, name)[2]
            if value in ("EUR", "JPY"):
                assert "SPELLED_CODE" not in [c.form for c in choices]


def test_mandarin_calls_never_offer_english_and_english_calls_never_offer_chinese(slot_facts):
    for ctx, event, name, choices in slot_facts:
        profile = ctx.spec["metadata"]["language_profile"]
        languages = {c.language for c in choices}
        if profile == "MANDARIN":
            assert languages == {"zh"}
        elif profile in ("ENGLISH", "SINGLISH"):
            assert languages == {"en"}


def test_unpinned_mixed_events_keep_both_languages(slot_facts):
    both = [(ctx.call_id, event["event_id"], name) for ctx, event, name, choices in slot_facts
            if ctx.spec["metadata"]["language_profile"] == "EN_ZH_MIXED" and not event.get("required_language")
            and {c.language for c in choices} == {"en", "zh"}]
    assert len(both) > 100


def test_previews_validation_and_rendering_read_the_same_choices(slot_facts, compiled):
    for ctx, event, name, choices in slot_facts:
        preview = _surfaces(ctx, event, name)
        assert preview == {c.form: {"language": c.language, "surface": c.surface} for c in choices}
        if name in QUESTION_PLACEHOLDERS:
            assert [c.form for c in choices] == ctx.form_choices(event, name)
    for call_id, (ctx, plan) in compiled.items():
        for event in plan["planned_events"]:
            skeleton_event = ctx.by_id[event["event_id"]]
            for mention in event["authoring"]["value_mentions"]:
                options = ctx.choices(skeleton_event, mention["placeholder"])
                assert (mention["form"], mention["language"], mention["surface"]) in [(c.form, c.language, c.surface) for c in options]


def test_a_required_form_that_cannot_be_spoken_in_the_event_is_a_contract_error(inputs):
    ctx = PlanContext(inputs, calls_of("EN_ZH_MIXED")[0])
    event = next(e for e in ctx.events if e["kind"] == "VALUE_STATEMENT")
    pinned = {**event, "required_language": "zh", "values": {**event["values"], "K01": {
        "value": "S8779219E", "spoken_form": {"requirement": "REQUIRED", "form": "OH_FOR_ZERO"}}}}
    with pytest.raises(PlanInputError, match="no permitted form"):
        ctx.form_choices(pinned, "K01")


def test_a_readback_never_offers_a_field_whose_only_form_is_in_the_other_language(inputs):
    """CALL_178 reads back T01 in a Mandarin-pinned event while K01 must be spoken with the English-only OH_FOR_ZERO form."""
    ctx = PlanContext(inputs, "CALL_178")
    event = ctx.by_id["EV_016"]
    assert event["kind"] == "READBACK" and event["required_language"] == "zh"
    candidates = ctx.readback_candidates(event)
    assert "K01" in ctx.readback_candidates() and "K01" not in candidates and "T01" in candidates
    assert len(candidates) >= event["readback"]["min_field_count"]


# ============================================================================================ guards
CONTRAST_REJECTED = ["投资十万", "买五股", "二〇〇一年五月五日", "投资千万", "三点五", "一百二", "五十万", "十二", "两千", "三十元", "百分之五",
                     "买五月", "利率一点五", "五百块", "十八岁", "二千二百"]
CONTRAST_ACCEPTED = ["千万别", "千万不要说", "等一下", "不一定", "第二只基金", "第十二只", "万一", "两只基金", "三个产品", "一起", "一定",
                     "十分", "两三天", "这一年", "一日", "一些", "有一点担心", "我想一想"]


@pytest.mark.parametrize("text", CONTRAST_REJECTED)
def test_value_expression_guard_rejects_identifiable_values(text):
    assert guards.value_expression_errors(text), text


@pytest.mark.parametrize("text", CONTRAST_ACCEPTED)
def test_value_expression_guard_leaves_ordinary_wording_alone(text):
    assert not guards.value_expression_errors(text), text


def test_value_expression_guard_reads_the_approved_chinese_currency_names_as_markers(inputs):
    markers = get_lexicon(inputs).han_currency_markers
    assert {"新币", "美元", "人民币", "日元", "欧元"} <= set(markers)
    assert guards.value_expression_errors("五美元", markers) and guards.value_expression_errors("三人民币", markers)
    assert not guards.value_expression_errors("美元走势", markers)


@pytest.mark.parametrize("text,bare", [("可以。", True), ("好的！", True), ("好啦。", True), ("可以啊", True), ("行", True), ("没问题。", True), ("对", True),
                                       ("Can lah.", True), ("OK lah.", True), ("okay", True), ("Sure.", True), ("Yes", True),
                                       ("可以吧", False), ("可以吗？", False), ("好的，不过我再想想", False), ("我再确认一下", False), ("不可以", False),
                                       ("应该可以吧，先不要下单", False), ("Can ah?", False), ("Can ah", False), ("Mm, okay, I see.", False),
                                       ("Can, but let me think first.", False), ("嗯，我先听着。", False)])
def test_bare_affirmative_check_is_whole_utterance_and_keeps_hedges(text, bare):
    assert guards.bare_affirmative(text) is bare


def test_weak_acknowledgement_events_apply_the_bare_affirmative_check_in_every_profile(validator):
    for profile, accepted, rejected in (("MANDARIN", ["可以吧", "我再确认一下", "应该可以吧，先不要下单"], ["可以。", "好的！"]),
                                        ("SINGLISH", ["Can ah?", "Mm, okay, I see."], ["Can lah.", "OK lah."])):
        ctx, event = find_event(profile, "WEAK_ACKNOWLEDGEMENT")
        for text in accepted:
            assert not errors_with_text(validator, ctx, event, text), (profile, text)
        for text in rejected:
            assert any("bare unqualified affirmative" in e for e in errors_with_text(validator, ctx, event, text)), (profile, text)


def test_the_word_boundary_regexes_of_schema_1_0_miss_han_neighbours_but_2_0_does_not(inputs):
    lexicon = get_lexicon(inputs)
    assert WEAK_ACK_BANNED.search("yes啦") is None
    assert lexicon.weak_ack.search("yes啦") and lexicon.guarantee.search("guarantee的") and lexicon.magnitude.search("几million元")
    from fyp_evaluation.check_plans.template import _PERCENT
    assert _PERCENT.search("5％") is None and lexicon.percent.search("5％") and lexicon.percent.search("５％")  # full-width percent sign


def test_script_rules_pins_and_digits_are_enforced_outside_the_slots(validator):
    ctx, event = find_event("MANDARIN", "REFERENCE_SELECTION")
    assert not errors_with_text(validator, ctx, event, "就选您刚才提到的第一只吧。")
    for text, fragment in (("Let us go with the first one.", "Latin letters"), ("就选第2只吧。", "digits"), ("就选 ok 那只吧。", "Latin letters")):
        assert any(fragment in e for e in errors_with_text(validator, ctx, event, text)), text
    for profile in ("ENGLISH", "SINGLISH"):
        ctx, event = find_event(profile, "REFERENCE_SELECTION")
        assert any("Chinese characters" in e for e in errors_with_text(validator, ctx, event, "就选 the first one."))
    ctx = next(PlanContext(plan_inputs(), c) for c in calls_of("EN_ZH_MIXED") if any(e.get("required_language") == "zh" and e["kind"] == "REFERENCE_SELECTION"
                                                                                       for e in PlanContext(plan_inputs(), c).events))
    zh_pinned = next(e for e in ctx.events if e.get("required_language") == "zh" and e["kind"] == "REFERENCE_SELECTION")
    assert any("pinned to Mandarin" in e for e in errors_with_text(validator, ctx, zh_pinned, "Let us go with the first one."))
    assert not errors_with_text(validator, ctx, zh_pinned, "就选您刚才提到的第一只吧。")


def test_chinese_product_and_currency_names_are_only_allowed_through_slots(validator):
    ctx, event = find_event("MANDARIN", "REFERENCE_SELECTION")
    for text, fragment in (("就选猎户座环球收益基金吧。", "product names"), ("就选新币那只吧。", "currency names"), ("就选美元那只吧。", "currency names")):
        assert any(fragment in e for e in errors_with_text(validator, ctx, event, text)), text
    ctx, event = find_event("EN_ZH_MIXED", "REFERENCE_SELECTION")
    assert any("product names" in e for e in errors_with_text(validator, ctx, event, "Let us take the Orion Global Income Fund one."))


def test_standalone_chinese_currency_units_after_a_slot_are_rejected(validator):
    ctx, event = find_event("MANDARIN", "VALUE_STATEMENT")
    required = ctx.slot_rules(event).required
    text = "，".join(f"[[{name}]]" for name in required) + "元。"
    assert any("元/块/刀" in e for e in errors_with_text(validator, ctx, event, text))


def test_mandarin_compliance_wording_is_not_judged_by_g4(validator):
    """G4 has no Mandarin guarantee / laundering / insider guard: the meaning of 我们不能保证回报 is G5/G7's judgement."""
    ctx, event = find_event("MANDARIN", "GUARANTEE_REFUSAL")
    assert not errors_with_text(validator, ctx, event, "我们不能保证回报。")
    ctx, event = find_event("MANDARIN", "REFERENCE_SELECTION")
    assert not errors_with_text(validator, ctx, event, "我们可以担保，也有人说过内幕消息和洗钱。")


# ============================================================================================ scenario slots
def _fee_ctx(basis, profile):
    inputs = plan_inputs(fee_basis=basis)
    for call_id in calls_of(profile):
        ctx = PlanContext(inputs, call_id)
        if ctx.fee and ctx.fee["basis"] == basis:
            return ctx
    raise AssertionError(f"no {basis} fee call in {profile}")


def test_fee_rate_and_amount_render_in_the_wording_language_and_check_amount_currency_and_basis():
    for basis, profile, language in (("RATE", "MANDARIN", "zh"), ("AMOUNT", "MANDARIN", "zh"), ("RATE", "ENGLISH", "en"), ("AMOUNT", "SINGLISH", "en")):
        ctx = _fee_ctx(basis, profile)
        event = next(e for e in ctx.events if e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C04"])
        form = ctx.scenario_forms(event, "FEE")[0]
        surface, value, shown_form, source = scenario_surface(ctx, event, "FEE", form)
        assert source == "SCENARIO_FEE" and ctx.catalog.form_language("FEE", shown_form) == language
        decoded, matches = _round_trip(ctx, event, "FEE", surface, value)
        assert matches, (basis, profile, surface, decoded)
        if basis == "RATE":
            assert surface == ("百分之零点七五" if language == "zh" else "0.75%")
        else:
            assert decoded[2] == ctx.fee["currency"] and decoded[1] == Decimal("37.5")
        if basis == "AMOUNT":  # the same number in another currency is not the fee
            other = "USD" if ctx.fee["currency"] != "USD" else "SGD"
            forged = ctx.catalog.render_money(ctx.call_id, Decimal("37.5"), other, language)
            assert _round_trip(ctx, event, "FEE", forged, value)[1] is False
            assert _round_trip(ctx, event, "FEE", ctx.catalog.render_money(ctx.call_id, Decimal("38.5"), ctx.fee["currency"], language), value)[1] is False


def test_limit_prices_and_distractor_currencies_follow_the_event_language(inputs):
    seen = {"LIMIT_PRICE": set(), "CURRENCY": set()}
    for profile in PROFILES:
        for call_id in calls_of(profile):
            ctx = PlanContext(inputs, call_id)
            for event in ctx.events:
                for name in ("LIMIT_PRICE", "CURRENCY"):
                    if name in ctx.slot_rules(event).required:
                        for choice in ctx.choices(event, name):
                            seen[name].add((profile, choice.language))
                            assert (profile != "MANDARIN") or choice.language == "zh"
                            slot = ctx.limit_slot(event) or "final"
                            decoded, matches = _round_trip(ctx, event, name, choice.surface,
                                                           ctx.limit_prices[slot] if name == "LIMIT_PRICE" else event["currency"])
                            assert matches
    assert {("MANDARIN", "zh"), ("ENGLISH", "en")} <= seen["LIMIT_PRICE"] and {("MANDARIN", "zh"), ("SINGLISH", "en")} <= seen["CURRENCY"]


def test_a_price_needing_more_decimals_than_can_be_rendered_fails_loudly(validator, inputs):
    for call_id in calls_of("MANDARIN"):
        ctx = PlanContext(inputs, call_id)
        event = next((e for e in ctx.events if ctx.limit_slot(e) and ctx.limit_slot(e) in ctx.limit_prices), None)
        if event:
            ctx.limit_prices = {**ctx.limit_prices, ctx.limit_slot(event): 12.125}
            errors = validator.errors(synthesize_response(ctx), ctx)
            assert any("decimal places" in e for e in errors)
            return
    raise AssertionError("no Mandarin LIMIT call")


# ============================================================================================ offline compilation
def test_all_250_calls_compile_project_back_to_their_skeletons_and_validate_against_schema_2(compiled, inputs, builder):
    assert len(compiled) == 250
    schema = Draft202012Validator(builder.artifact_schema)
    for call_id, (ctx, plan) in compiled.items():
        assert plan["schema_version"] == "2.0" and plan["skeleton_header"]["schema_version"] == "2.0"
        assert not list(schema.iter_errors(plan)), call_id
        assert serialize_json(project_check_plan_to_skeleton(plan)) == inputs.skeleton_texts[call_id]


def test_compilation_is_deterministic(compiled, inputs, builder):
    for call_id in ("CALL_001", "CALL_003", "CALL_010", "CALL_048", "CALL_121"):
        ctx, plan = compiled[call_id]
        again = compile_artifact(inputs, ctx, synthesize_response(ctx), {**GENERATION, "response_schema_sha256": builder.response_schema_sha256},
                                 builder.response_schema)
        assert serialize_json(again) == serialize_json(plan)


def test_value_spans_are_python_character_offsets_that_survive_a_json_round_trip(compiled):
    han_mentions = 0
    for call_id, (ctx, plan) in compiled.items():
        reloaded = parse_json_text(serialize_json(plan))
        for event in reloaded["planned_events"]:
            authoring = event["authoring"]
            for mention in authoring["value_mentions"]:
                assert authoring["planned_text"][mention["start"]:mention["end"]] == mention["surface"]
                assert mention["end"] - mention["start"] == len(mention["surface"])
                han_mentions += bool(HAN.search(mention["surface"]))
    assert han_mentions > 1000


def test_each_profile_keeps_to_its_script_rules_in_the_compiled_text(compiled):
    for call_id, (ctx, plan) in compiled.items():
        profile = ctx.spec["metadata"]["language_profile"]
        for event in plan["planned_events"]:
            authoring = event["authoring"]
            text = authoring["planned_text"]
            for mention in sorted(authoring["value_mentions"], key=lambda m: -m["start"]):
                if profile == "MANDARIN" and mention["placeholder"] == "K01":
                    text = text[:mention["start"]] + text[mention["end"]:]  # the identifier span is the only place Latin letters may live
            if profile == "MANDARIN":
                assert not re.search(r"[A-Za-z0-9]", text), (call_id, event["event_id"], text)
            if profile in ("ENGLISH", "SINGLISH"):
                assert not HAN.search(authoring["planned_text"]), (call_id, event["event_id"])
            pin = event.get("required_language")
            if pin:
                assert HAN.search(text) if pin == "zh" else re.search("[A-Za-z]", text)
                assert {m["language"] for m in authoring["value_mentions"]} <= {pin}


def test_mixed_calls_compile_values_in_both_languages(compiled):
    """The synthetic wording alternates languages by event position, so a short call may use only one; the release as a whole uses both."""
    per_call = {call_id: {m["language"] for e in compiled[call_id][1]["planned_events"] for m in e["authoring"]["value_mentions"]}
                for call_id in calls_of("EN_ZH_MIXED")}
    assert set().union(*per_call.values()) == {"en", "zh"}
    assert sum(languages == {"en", "zh"} for languages in per_call.values()) >= 0.8 * len(per_call)


def test_the_request_brief_is_built_for_every_call_with_languages_and_previews(builder, inputs):
    for profile in PROFILES:
        call_id = calls_of(profile)[0]
        body = builder.build(call_id, "A001")
        brief = json.loads(body["input"][1]["content"])
        assert brief["language"]["language_profile"] == profile and set(brief["language"]["speakers"]) == {"CLIENT", "EMPLOYEE"}
        events = brief["events"]
        assert all("spoken_languages" in e for e in events)
        previews = [v for e in events for v in e.get("rendered_surfaces", {}).values() if isinstance(v, dict)]
        assert previews and all(set(next(iter(p.values()))) == {"language", "surface"} for p in previews if p)
        if profile == "MANDARIN":
            assert any("approved_chinese_name" in p for p in brief["products"])
    for call_id in inputs.ready_call_ids:  # the brief of every call can be built and serialised
        assert builder.build(call_id, "A001")["input"][1]["content"]


# ============================================================================================ schemas and provenance
def test_schema_2_plans_and_records_carry_the_language_profile_hash(compiled, inputs, builder):
    expected = g1.sha256((RELEASE / "language_profiles.json").read_bytes())
    assert inputs.source_hashes()["language_profiles_sha256"] == expected
    plan = compiled["CALL_001"][1]
    assert plan["source"]["language_profiles_sha256"] == expected
    texts = {"CALL_001": serialize_json(plan)}
    record = build_record(inputs, texts, {"CALL_001": {"accepted_history": []}})
    record_schema = Draft202012Validator(parse_json_text((SCHEMAS / "stage4/check_plan_record.v2.schema.json").read_text(encoding="utf-8")))
    assert record["schema_version"] == "2.0" and record["upstream"]["language_profiles_sha256"] == expected
    assert not list(record_schema.iter_errors(record))
    del record["upstream"]["language_profiles_sha256"]
    assert list(record_schema.iter_errors(record))
    broken = deepcopy(plan)
    del broken["source"]["language_profiles_sha256"]
    assert list(Draft202012Validator(builder.artifact_schema).iter_errors(broken))


def test_the_historical_schema_rejects_a_schema_2_plan_and_the_2_0_schema_rejects_a_missing_language(compiled, builder):
    v1 = Draft202012Validator(parse_json_text((SCHEMAS / "stage4/check_plan.schema.json").read_text(encoding="utf-8")))
    plan = compiled["CALL_001"][1]
    assert list(v1.iter_errors(plan))
    broken = deepcopy(plan)
    mention = next(m for e in broken["planned_events"] for m in e["authoring"]["value_mentions"])
    del mention["language"]
    assert list(Draft202012Validator(builder.artifact_schema).iter_errors(broken))


def test_a_multilingual_artifact_is_current_until_language_profiles_change(tmp_path):
    """dependency_status compares every freshly read upstream hash with the recorded ones; the language hash must be among them."""
    root = tmp_path / "release"
    for path in sorted(RELEASE.rglob("*")):
        if path.is_file():
            target = root / path.relative_to(RELEASE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    call_id = "CALL_003"
    copy_inputs = plan_inputs(root, [call_id])
    (root / "authoring/call_configs").mkdir(parents=True, exist_ok=True)
    (root / "authoring/call_configs" / f"{call_id}.json").write_bytes(copy_inputs.config_texts[call_id].encode("utf-8"))
    runner = CheckPlanRunner(root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, inputs=copy_inputs)
    ctx = runner.builder.context(call_id)
    plan = compile_artifact(copy_inputs, ctx, synthesize_response(ctx), {**GENERATION, "response_schema_sha256": runner.builder.response_schema_sha256},
                            runner.builder.response_schema)
    (root / "authoring/check_plans").mkdir(parents=True, exist_ok=True)
    (root / "authoring/check_plans" / f"{call_id}.json").write_bytes(serialize_json(plan).encode("utf-8"))
    assert runner.dependency_status([call_id])[call_id] == {"status": "current", "changed": [], "error": None}
    profiles = root / "language_profiles.json"
    profiles.write_bytes(profiles.read_bytes() + b"\n")
    status = runner.dependency_status([call_id])[call_id]
    assert status["status"] == "stale" and "language_profiles_sha256" in status["changed"]


def test_the_g3_gate_still_refuses_the_multilingual_release():
    from fyp_evaluation.call_configs.inputs import G2BStateError, load_inputs as load_g3
    from fyp_evaluation.check_plans.inputs import load_inputs as load_g4

    for loader in (load_g3, load_g4):
        with pytest.raises(G2BStateError, match="multilingual release"):
            loader(RELEASE, SCHEMAS)
