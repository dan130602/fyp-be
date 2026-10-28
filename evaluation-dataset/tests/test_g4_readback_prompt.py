"""Offline regression for literal Q readback slots and versioned request compatibility."""
import json
from copy import deepcopy

import pytest

from fyp_evaluation.check_plans.config import PROMPT_DIR, prompt_template
from fyp_evaluation.check_plans.feasibility import make_validator, synthesize_response
from fyp_evaluation.check_plans.request import RequestBuilder
from g4_multilingual_support import SCHEMAS, plan_inputs
from g4_support import shared_inputs


@pytest.fixture(scope="module")
def builders():
    inputs = plan_inputs(call_ids=["CALL_149"])
    return RequestBuilder(inputs, SCHEMAS, prompt_template_name="check_plan.v3.md"), RequestBuilder(inputs, SCHEMAS, prompt_template_name="check_plan.v2.md")


def test_new_multilingual_requests_clarify_slots_and_preserve_the_old_brief(builders):
    current, old = builders
    assert "template v3" in current.developer_message  # Historical v3 remains reconstructable after activation of later versions.
    new_body, old_body = current.build("CALL_149", "A002"), old.build("CALL_149", "A002")
    assert "template v3" in new_body["input"][0]["content"]
    assert "template v2" in old_body["input"][0]["content"]
    assert current.prompt_template_sha256 != old.prompt_template_sha256
    assert {k: v for k, v in new_body.items() if k != "input"} == {k: v for k, v in old_body.items() if k != "input"}
    new_brief, old_brief = json.loads(new_body["input"][1]["content"]), json.loads(old_body["input"][1]["content"])
    readbacks = 0
    for new, previous in zip(new_brief["events"], old_brief["events"], strict=True):
        if new["kind"] == "READBACK":
            readbacks += 1
            assert "[[Q]]" in previous["readback"]["note"]
            assert "[[Q]]" not in new["readback"]["note"]
            assert "T04 becomes [[T04]]" in new["readback"]["note"]
            new["readback"]["note"] = previous["readback"]["note"]
    assert readbacks
    assert new_brief == old_brief  # Gold, values, languages, constraints and every other brief field are identical.


def test_v3_only_changes_the_readback_instruction_and_version_header():
    old = (PROMPT_DIR / "check_plan.v2.md").read_text(encoding="utf-8")
    new = (PROMPT_DIR / "check_plan.v3.md").read_text(encoding="utf-8")
    expected = old.replace("template v2, multilingual releases", "template v3, multilingual releases", 1).replace(
        "Write one `[[Q]]` per chosen field and no other placeholders.",
        "Use each chosen question ID as its placeholder name, exactly once (for example, `[[T04]]` for chosen field `T04`), "
        "and no other placeholders. The letter Q is only a metavariable, never a literal placeholder name.", 1)
    assert new == expected


def test_concrete_readback_slots_pass_and_literal_q_still_fails(builders):
    current, _ = builders
    ctx = current.context("CALL_149")
    response = synthesize_response(ctx)
    entry = next(e for e in response["events"] if e["event_id"] == "EV_022")
    entry.update(readback_question_ids=["T05", "T01", "T06"], value_forms=[],
                 text="To recap, you want to [[T05]] [[T01]] using [[T06]].")
    validator = make_validator(current)
    assert validator.errors(response, ctx) == []
    wrong = deepcopy(response)
    next(e for e in wrong["events"] if e["event_id"] == "EV_022")["text"] = "To recap, you want to [[Q]] [[Q]] using [[Q]]."
    errors = validator.errors(wrong, ctx)
    assert any("unknown placeholder [[Q]]" in e for e in errors)
    for field in ["T05", "T01", "T06"]:
        assert any(f"[[{field}]] must appear exactly once (found 0)" in e for e in errors)


def test_historical_release_keeps_v1_and_the_historical_readback_note():
    inputs = shared_inputs()
    builder = RequestBuilder(inputs, SCHEMAS)
    assert prompt_template(inputs.documents) == "check_plan.v1.md"
    for call in inputs.call_ids:
        brief = json.loads(builder.build(call, "A001")["input"][1]["content"])
        if any(e["kind"] == "READBACK" for e in brief["events"]):
            assert all("[[Q]]" in e["readback"]["note"] for e in brief["events"] if e["kind"] == "READBACK")
            return
    raise AssertionError("historical readback coverage unexpectedly empty")
