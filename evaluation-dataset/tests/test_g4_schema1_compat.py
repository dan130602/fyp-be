"""Schema-1.0 (historical pilot) behaviour must not move when G4 learns languages.

The byte-level guarantees (G4 --check / --validate over every accepted and historical pilot response, and the pinned
request hashes) live in `test_request_hashes`, `test_g4_flows` and `test_lifecycle`; these tests state the specific
boundaries: same schema files, same hashes, same patterns, same surfaces, no language fields.
"""

from fyp_evaluation.check_plans import render as R
from fyp_evaluation.check_plans import template as T
from fyp_evaluation.check_plans.analysis import scenario_surface
from fyp_evaluation.check_plans.compile import assemble, get_lexicon
from fyp_evaluation.check_plans.config import SCHEMA_VERSION, schema_names, schema_version
from fyp_evaluation.check_plans.context import PlanContext, forms_of
from fyp_evaluation.check_plans.feasibility import synthesize_response
from fyp_evaluation.check_plans.request import RequestBuilder
from g4_support import CALL_IDS, SCHEMAS, shared_inputs


def test_the_pilot_keeps_schema_1_0_and_its_own_schema_files():
    inputs = shared_inputs()
    assert schema_version(inputs.documents) == SCHEMA_VERSION == "1.0"
    assert schema_names(inputs.documents) == ("check_plan", "check_plan_record")
    assert RequestBuilder(inputs, SCHEMAS).artifact_schema["properties"]["schema_version"] == {"const": "1.0"}


def test_the_pilot_records_exactly_the_five_g0_source_hashes():
    assert sorted(shared_inputs().source_hashes()) == sorted(
        ["checklist_sha256", "check_semantics_sha256", "edge_case_catalog_sha256", "normalization_sha256", "generation_plan_sha256"])


def test_the_pilot_lexicon_uses_the_original_patterns():
    lexicon = get_lexicon(shared_inputs())
    assert lexicon.weak_ack is T.WEAK_ACK_BANNED and lexicon.guarantee is T.GUARANTEE_WORD
    assert lexicon.laundering is T.LAUNDERING_WORDS and lexicon.insider is T.INSIDER_WORDS
    assert lexicon.magnitude is T._MAGNITUDE and lexicon.percent is T._PERCENT and lexicon.currency_symbol is T._CURRENCY_SYMBOL
    placeholders, outside, errors = T.parse_template("没有拉丁字母 [[T03]]")
    assert any("needs wording" in e for e in errors)  # 1.0 still needs Latin wording outside the slots


def test_pilot_contexts_have_no_language_fields_and_unchanged_form_choices():
    inputs = shared_inputs()
    for call_id in CALL_IDS:
        ctx = PlanContext(inputs, call_id)
        assert not ctx.multilingual and ctx.catalog is None and ctx.allowed_languages(ctx.events[0]) is None
        for event in ctx.events:
            for name in ctx.slot_rules(event).required:
                if name in ctx.checks:
                    assert ctx.form_choices(event, name) == forms_of(ctx.binding(event, name)[1])
            assert ctx.readback_candidates(event) == ctx.readback_candidates()


def test_pilot_plans_have_unchanged_mentions_and_fixed_scenario_surfaces():
    inputs = shared_inputs()
    ctx = PlanContext(inputs, "CALL_002")
    plan = assemble(ctx, *_blocks(ctx), {"attempt_id": "A001"})
    assert plan["schema_version"] == "1.0"
    mentions = [m for e in plan["planned_events"] for m in e["authoring"]["value_mentions"]]
    assert mentions and all("language" not in m and list(m)[:5] == ["placeholder", "question_id", "value", "form", "surface"] for m in mentions)
    for call_id in CALL_IDS:
        ctx = PlanContext(inputs, call_id)
        for event in ctx.events:
            if ctx.fee and "FEE" in ctx.slot_rules(event).required:
                assert scenario_surface(ctx, event, "FEE")[0] == R.render_fee(ctx.fee, ctx.aliases)
            if ctx.limit_slot(event):
                slot = ctx.limit_slot(event)
                assert scenario_surface(ctx, event, "LIMIT_PRICE")[0] == R.render_limit_price(ctx.limit_prices[slot], ctx.limit_currency[slot], ctx.aliases)
            if "currency" in event:
                assert scenario_surface(ctx, event, "CURRENCY")[2] == "NATURAL_NAME"


def _blocks(ctx):
    from fyp_evaluation.check_plans.analysis import analyze

    authoring, derived, errors = analyze(ctx, get_lexicon(ctx.inputs), synthesize_response(ctx))
    assert not errors
    return authoring, derived
