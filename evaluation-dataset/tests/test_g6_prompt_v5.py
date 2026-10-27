"""G6 prompt v5: selected for multilingual releases only; v1-v4 are untouched and every v4 rule, ban and fixed instruction survives."""

import hashlib
import json
from pathlib import Path

import pytest

import g6_support
import multilingual_support as M
from fyp_evaluation.call_specs.compile import text_hash
from fyp_evaluation.dialogues import config
from fyp_evaluation.dialogues.request import RequestBuilder

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "stage6"
# SHA-256 of each earlier prompt with line endings normalised to LF, so a checkout's autocrlf setting cannot change the pin.
PINNED = {1: "7f74263dbf1a80a324be8c29d1280987dc5b58a0ebdab195ad25db85baa0329e",
          2: "85f71f1b7b0932b80ecd099ffa59fbed9985d1a9271927a57e75efd98321c50a",
          3: "ada8099f2a5fc21f33fdc4241d48717169a09901354d6d0572e503e609c18e63",
          4: "948ada75ff772ced4c74a99aa8f836ffc360cd1558e4167fdaa50fda2d34ad82"}
REWORDED_V4_LINES = ("# G6 dialogue authoring instructions (template v4", "- EN_ZH_MIXED:", "- Use as many supporting turns",
                     "Supporting speech may greet,")


def lines(name):
    return (PROMPTS / name).read_text(encoding="utf-8").splitlines()


def test_a_multilingual_release_selects_v5_and_the_pilot_keeps_its_historical_prompt():
    assert config.prompt_template(M.upstream_inputs().documents) == "dialogue.v5.md"
    assert config.prompt_template(g6_support.shared_inputs().documents) == "dialogue.v1.md"
    multilingual = RequestBuilder(M.verification_inputs([M.first_call("MANDARIN")]), M.SCHEMAS)
    assert multilingual.prompt_template_sha256 == text_hash((PROMPTS / "dialogue.v5.md").read_text(encoding="utf-8"))
    pilot = RequestBuilder(g6_support.shared_inputs(), M.SCHEMAS)
    assert pilot.prompt_template_sha256 == text_hash((PROMPTS / "dialogue.v1.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize("version", sorted(PINNED))
def test_earlier_prompts_are_byte_stable(version):
    data = (PROMPTS / f"dialogue.v{version}.md").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == PINNED[version]


def test_only_the_four_intended_v4_lines_are_reworded_in_v5():
    v4, v5 = lines("dialogue.v4.md"), lines("dialogue.v5.md")
    removed = [line for line in v4 if line not in v5]
    assert len(removed) == 4, removed
    assert all(any(line.startswith(prefix) for prefix in REWORDED_V4_LINES) for line in removed), removed
    added = [line for line in v5 if line not in v4]
    assert len(added) == 10, added  # title, EN_ZH_MIXED, two Turns bullets, five conversation bullets, the supporting-speech paragraph


def test_every_v4_ban_and_fixed_instruction_survives_verbatim_in_v5():
    v4, v5 = lines("dialogue.v4.md"), lines("dialogue.v5.md")
    start = v4.index("Supporting speech must never, whatever happens elsewhere in the call and in whatever language:")
    bans = [line for line in v4[start + 1:start + 14] if line.startswith("- ")]
    assert len(bans) == 11 and all(ban in v5 for ban in bans)
    for fixed in ("- Place every event exactly once", "- Never write an event's text yourself", "- Never continue, complete or modify a number",
                  "- `ORDER`: the `before` event", "- `SAME_TURN`", "- `NOT_SAME_TURN`", "- `MIN_INTERVENING_TURNS`",
                  "- Turns strictly alternate between EMPLOYEE and CLIENT.", "- Each turn makes sense given everything said so far",
                  "- Do not open every turn with an acknowledgement", "- Avoid accidental non-sequiturs", "- Lead naturally into each planned event",
                  "- When the fixed events, their wording or their constraints make a smooth exchange hard"):
        assert any(line.startswith(fixed) for line in v5), fixed
    for paragraph in ("### The recommendation (only when the brief has `advisory_recommendation`)", "## Repair guidance",
                      "Questions in supporting speech.", "The only exception to these rules is the single recommendation",
                      "Some events are deliberately imperfect:", "Return only the JSON object with `turns`."):
        assert any(line.startswith(paragraph) for line in v5), paragraph
    length = next(line for line in v5 if line.startswith("Count the length in the unit"))
    assert length == next(line for line in v4 if line.startswith("Count the length in the unit"))
    assert "advisory" in length and "Prefer a coherent, natural conversation to padding" in length


@pytest.mark.parametrize("phrase", [
    "Let the conversation decide how much supporting talk there is", "reaches it by developing the conversation, never by repeating it",
    "Let each turn's length fit what is being said", "Open the call as `setting.call_reason` describes it",
    "the opening implies neither", "Let topics develop", "making the same reassurance the employee's default response",
    "without requiring an added acknowledgement", "Preserve intentional non-response, hesitation and avoidance",
    "never introduce a compliance act or a stronger response merely to make a reaction sound appropriate",
    "never imply that something has already been said when it has not", "Facts given in the brief's setting may be used as background",
    "Event texts stay exactly as written", "do not add turns or questions merely to separate them",
    "Avoid an unrelated detour just before the readback or confirmation", "recognisably a private-banking professional",
    "do not force a switch in every turn or follow a fixed pattern", "avoid saying the same thought in both languages"])
def test_v5_states_its_intended_changes(phrase):
    assert phrase in (PROMPTS / "dialogue.v5.md").read_text(encoding="utf-8")


def test_v5_adds_no_quota_cap_or_named_motif_and_keeps_the_switch_rule_g7_checks():
    text = (PROMPTS / "dialogue.v5.md").read_text(encoding="utf-8")
    assert "at least one client turn contains a meaningful switch" in text  # the language row G7 verifies is unchanged
    for forbidden in ("at most", "no more than", "at least two", "notebook", "desk", "folder", "maximum of"):
        assert forbidden not in text.lower().replace("at least one client turn", ""), forbidden


@pytest.mark.parametrize("profile", M.PROFILES)
def test_the_brief_and_request_format_are_unchanged_between_v4_and_v5(profile):
    call = M.first_call(profile)
    builder = RequestBuilder(M.verification_inputs([call]), M.SCHEMAS)
    body = builder.build(call, "A001")
    assert body["text"]["format"]["name"] == "g6_dialogue_v2" and "template v5" in body["input"][0]["content"]
    assert set(body) == {"model", "input", "text", "reasoning", "max_output_tokens", "store", "metadata"}
    brief = json.loads(body["input"][1]["content"])
    assert brief["call"]["length"]["enforcement"] == "advisory"
    # the only difference from a v4 request is the system prompt: re-render it with the v4 text and compare everything else
    v4_text = (PROMPTS / "dialogue.v4.md").read_text(encoding="utf-8")
    v5_text = (PROMPTS / "dialogue.v5.md").read_text(encoding="utf-8")
    assert body["input"][0]["content"] == v5_text and body["input"][0]["content"] != v4_text
