"""G3 sees the frozen language-variant purpose; old request versions are retained."""
import json

import pytest

from fyp_evaluation.call_configs.request import RequestBuilder, build_brief, request_hash
from fyp_evaluation.call_configs.feasibility import make_validator, synthesize_response
from g4_multilingual_support import upstream_inputs, SCHEMAS
from g3_support import shared_inputs


@pytest.fixture(scope="module")
def builders():
    inputs = upstream_inputs()
    return RequestBuilder(inputs, SCHEMAS), RequestBuilder(inputs, SCHEMAS, prompt_template_name="call_config.v2.md")


def test_variant_brief_explains_the_actual_pinned_pair(builders):
    new, old = builders
    now, previous = new.build("CALL_035", "A002"), old.build("CALL_035", "A002")
    assert "template v3" in now["input"][0]["content"]
    assert "template v2" in previous["input"][0]["content"]
    brief = json.loads(now["input"][1]["content"])
    before = json.loads(previous["input"][1]["content"])
    edge = next(e for e in brief["edge_assignments"] if e["assignment_id"] == "EA_085")
    details = edge.pop("language_variant_details")
    assert details["event_languages"] == {"source":"en", "target":"zh"}
    assert "employee's wrong readback" in details["purpose"]
    assert brief == before
    assert before == build_brief(old.inputs, old.context("CALL_035"))
    assert {k:v for k,v in now.items() if k != "input"} == {k:v for k,v in previous.items() if k != "input"}


def test_every_new_variant_detail_is_an_exact_frozen_catalog_projection(builders):
    new, old = builders
    catalog = {c["id"]:c for c in new.inputs.documents["edge_case_catalog"]["edge_cases"]}
    count = 0
    for call in new.inputs.call_ids:
        brief = json.loads(new.build(call, "A001")["input"][1]["content"])
        for edge in brief["edge_assignments"]:
            details = edge.pop("language_variant_details", None)
            if details:
                variant = next(v for v in catalog[edge["edge_case_id"]]["language_variants"] if v["variant_id"] == edge["language_variant"])
                assert details == {k:variant[k] for k in ("event_languages","purpose") if k in variant}
                count += 1
        assert brief == build_brief(old.inputs, old.context(call))
    assert count > 0


def test_v1_request_is_unchanged_and_unknown_template_is_rejected():
    inputs = shared_inputs()
    default = RequestBuilder(inputs, SCHEMAS)
    explicit = RequestBuilder(inputs, SCHEMAS, prompt_template_name="call_config.v1.md")
    for call in inputs.call_ids:
        assert request_hash(default.build(call,"A001")) == request_hash(explicit.build(call,"A001"))
    with pytest.raises(ValueError, match="Unknown G3 prompt"):
        RequestBuilder(inputs, SCHEMAS, prompt_template_name="../unknown.md")


def test_response_contract_and_feasibility_are_unchanged(builders):
    new, old = builders
    assert new.response_schema == old.response_schema
    validator = make_validator(new)
    for call in new.inputs.call_ids:
        ctx = new.context(call)
        assert validator.errors(synthesize_response(ctx), ctx) == []
