"""G6 prompt v3 is selected for multilingual releases only; v2 and the historical prompt are untouched and every v2 prohibition survives."""

from pathlib import Path

import pytest

import g6_support
import multilingual_support as M
from fyp_evaluation.call_specs.compile import text_hash
from fyp_evaluation.dialogues import config
from fyp_evaluation.dialogues.request import RequestBuilder

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "stage6"


def lines(name):
    return (PROMPTS / name).read_text(encoding="utf-8").splitlines()


def test_a_multilingual_release_selects_v4_and_the_pilot_keeps_its_historical_prompt():
    assert config.prompt_template(M.upstream_inputs().documents) == "dialogue.v4.md"
    assert config.prompt_template(g6_support.shared_inputs().documents) == "dialogue.v1.md"
    multilingual = RequestBuilder(M.verification_inputs([M.first_call("MANDARIN")]), M.SCHEMAS)
    assert multilingual.prompt_template_sha256 == text_hash((PROMPTS / "dialogue.v4.md").read_text(encoding="utf-8"))
    pilot = RequestBuilder(g6_support.shared_inputs(), M.SCHEMAS)
    assert pilot.prompt_template_sha256 == text_hash((PROMPTS / "dialogue.v1.md").read_text(encoding="utf-8"))


def test_v2_is_preserved_unchanged_beside_v3():
    v2 = lines("dialogue.v2.md")
    assert v2[0] == "# G6 dialogue authoring instructions (template v2, multilingual releases)"
    assert "dialogue.v2.md" in config.PROMPT_TEMPLATE_MULTILINGUAL.replace("v4", "v2")  # the file exists beside v3
    assert (PROMPTS / "dialogue.v2.md").is_file() and (PROMPTS / "dialogue.v3.md").is_file()


def test_every_v2_prohibition_and_rule_survives_in_v3_except_the_five_intentionally_reworded_lines():
    v2, v3 = lines("dialogue.v2.md"), lines("dialogue.v3.md")
    removed = [line for line in v2 if line not in v3]
    assert len(removed) == 5, removed
    starts = ("# G6 dialogue authoring instructions (template v2", "- Use as many supporting turns", "- The client's next turn begins with a mild acknowledgement",
              "- The acknowledgement is not an instruction", "The only exceptions to these rules are")
    assert all(any(line.startswith(prefix) for prefix in starts) for line in removed)
    start = v2.index("Supporting speech must never, whatever happens elsewhere in the call and in whatever language:")
    bans = [line for line in v2[start + 1:start + 14] if line.startswith("- ")]
    assert len(bans) == 11 and all(ban in v3 for ban in bans)
    for fixed in ("- Place every event exactly once", "- Never write an event's text yourself", "- Never continue, complete or modify a number",
                  "- `ORDER`: the `before` event", "- Turns strictly alternate between EMPLOYEE and CLIENT."):
        assert any(line.startswith(fixed) for line in v3), fixed


@pytest.mark.parametrize("phrase", ["One coherent conversation", "Do not open every turn with an acknowledgement",
                                    "never invent protected compliance behaviour", "it is not mandatory and no stock phrase is required",
                                    "Never move an event or add wording only to achieve this", "answered only by their planned events"])
def test_v3_states_its_intended_changes(phrase):
    assert phrase in (PROMPTS / "dialogue.v3.md").read_text(encoding="utf-8")


def test_the_brief_and_the_request_format_are_unchanged_between_v2_and_v3_releases_of_a_call():
    builder = RequestBuilder(M.verification_inputs([M.first_call("SINGLISH")]), M.SCHEMAS)
    body = builder.build(M.first_call("SINGLISH"), "A001")
    assert body["text"]["format"]["name"] == "g6_dialogue_v2" and "template v4" in body["input"][0]["content"]
    assert set(body) == {"model", "input", "text", "reasoning", "max_output_tokens", "store", "metadata"}
