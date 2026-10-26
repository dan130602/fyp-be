"""Phase 3B: G6 v2 on the multilingual release (fake Batch transport only; nothing is sent anywhere)."""

import json
import shutil

import pytest
from jsonschema import Draft202012Validator

import multilingual_support as M
from fyp_evaluation import register_guide
from fyp_evaluation.dialogues import config
from fyp_evaluation.dialogues.language_checks import numeral_extension_errors, script_errors
from fyp_evaluation.dialogues.render import length_count, outside_text, render
from fyp_evaluation.dialogues.request import DialogueContext, RequestBuilder
from fyp_evaluation.dialogues.validate_response import DialogueValidator

CALLS = {profile: M.first_call(profile) for profile in M.PROFILES}


@pytest.fixture(scope="module")
def world(dialogues250_root, tmp_path_factory):
    """A writable copy of the accepted G3-G6 state with a G6 runner over it."""
    root, _ = dialogues250_root
    chain = M.copy_chain(root, tmp_path_factory.mktemp("g6world") / "dataset")
    chain.open_g6()
    return chain


def artifact(chain, call_id):
    return json.loads((chain.root / "authoring/dialogues" / f"{call_id}.json").read_text(encoding="utf-8"))


def validator_and_ctx(chain, call_id):
    runner = chain.g6
    return runner.validator, runner.builder.context(call_id)


# ============================================================================================ acceptance over all 250 calls
def test_all_250_dialogues_are_accepted_valid_v2_artifacts_in_their_own_unit(world):
    schema = world.g6.builder.artifact_schema
    units = {"ENGLISH": "words", "SINGLISH": "words", "MANDARIN": "han_chars", "EN_ZH_MIXED": "han_chars_plus_words"}
    specs = world.g6.inputs.specs
    assert len(list((world.root / "authoring/dialogues").glob("CALL_*.json"))) == 250
    for call_id in world.call_ids:
        dialogue = artifact(world, call_id)
        profile = specs[call_id]["metadata"]["language_profile"]
        assert dialogue["schema_version"] == "2.0" and dialogue["length"]["unit"] == units[profile], call_id
        assert not list(Draft202012Validator(schema).iter_errors(dialogue)), call_id
        low, high = dialogue["length"]["accepted_word_range"]
        assert low <= dialogue["length"]["words"] <= high
        assert ("register_guide_sha256" in dialogue["source"]) is (profile == "SINGLISH")
        assert dialogue["source"]["language_profiles_sha256"] and "language" not in dialogue["metadata"]


def test_the_dialogue_stage_validates_checks_and_records_units_per_group(world):
    assert world.g6.validate() == [] and world.g6.check() == {"missing": [], "changed": [], "extra": []}
    record = json.loads((world.root / "authoring/dialogue_record.json").read_text(encoding="utf-8"))
    schema = json.loads((M.SCHEMAS / "stage6/dialogue_record.v2.schema.json").read_text(encoding="utf-8"))
    assert not list(Draft202012Validator(schema).iter_errors(record))
    groups = record["summary"]["length_by_profile"]
    assert {"ENGLISH/SHORT", "MANDARIN/SHORT", "EN_ZH_MIXED/SHORT", "SINGLISH/SHORT"} <= set(groups)
    assert all(row["unit"] in ("words", "han_chars", "han_chars_plus_words") for row in groups.values())
    assert groups["MANDARIN/SHORT"]["unit"] == "han_chars" and groups["EN_ZH_MIXED/SHORT"]["unit"] == "han_chars_plus_words"


def test_each_attempt_records_its_calls_own_length_policy(world):
    expected = {"ENGLISH": "g6-length-v1", "SINGLISH": "g6-length-v1", "MANDARIN": "g6-length-v2-han-chars",
                "EN_ZH_MIXED": "g6-length-v2-han-chars-plus-words"}
    for profile, call_id in CALLS.items():
        provenance = json.loads((world.root / "authoring/dialogue_provenance" / f"{call_id}.json").read_text(encoding="utf-8"))
        attempt, accepted = provenance["attempts"][0], provenance["accepted"]
        assert attempt["params"]["length_policy"]["version"] == expected[profile] == accepted["params"]["length_policy"]["version"]
        assert attempt["exposure"]["run_kind"] == "offline"
        assert ("register_guide_sha256" in attempt["upstream_sha256"]) is (profile == "SINGLISH")
        assert ("register_guide_sha256" in accepted) is (profile == "SINGLISH")


# ============================================================================================ the v2 brief and prompt
@pytest.mark.parametrize("profile", M.PROFILES)
def test_the_brief_carries_language_names_length_unit_and_the_guide_only_for_singlish(world, profile):
    builder = world.g6.builder
    body = builder.build(CALLS[profile], "A001")
    brief = json.loads(body["input"][1]["content"])
    assert brief["call"]["language"]["language_profile"] == profile and "target_words" not in brief["call"]
    assert brief["call"]["length"]["unit"] == config.unit_of(builder.context(CALLS[profile]).length_policy)
    assert all(brief["people"][s]["spoken_names"] for s in ("client", "employee"))
    assert ("register_guide" in brief) is (profile == "SINGLISH")
    assert body["text"]["format"]["name"] == "g6_dialogue_v2" and "template v2" in body["input"][0]["content"]
    names = (brief.get("advisory_recommendation") or {}).get("product_names")
    if names is not None:
        assert set(names) == {"ENGLISH": {"en"}, "SINGLISH": {"en"}, "MANDARIN": {"zh"}, "EN_ZH_MIXED": {"en", "zh"}}[profile]


def test_the_pilot_g6_request_keeps_the_v1_names_and_unit():
    import g6_support
    builder = RequestBuilder(g6_support.shared_inputs(), M.SCHEMAS)
    body = builder.build("CALL_001", "A001")
    brief = json.loads(body["input"][1]["content"])
    assert body["text"]["format"]["name"] == "g6_dialogue_v1" and "template v1" in body["input"][0]["content"]
    assert "target_words" in brief["call"] and brief["call"]["language"] == "en-SG" and "register_guide" not in brief


# ============================================================================================ marker and script rules
def mandarin_ctx(world):
    return world.g6.builder.context(CALLS["MANDARIN"])


def _events_in_one_turn_each(ctx):
    return " ".join(f"[[{event_id}]]" for event_id in ctx.texts)


def test_markers_may_sit_beside_chinese_but_never_beside_latin_or_digits(world):
    ctx = mandarin_ctx(world)
    first = next(iter(ctx.texts))
    for text, ok in (("好的，[[%s]]谢谢" % first, True), ("[[%s]]。" % first, True), ("好的 [[%s]] 谢谢" % first, True),
                     ("OK[[%s]]" % first, False), ("[[%s]]7" % first, False)):
        errors = render(ctx, {"turns": [{"speaker": "EMPLOYEE", "text": text}]})[2]
        assert (not [e for e in errors if "separate token" in e]) is ok, text
    pilot_ctx = DialogueContext(__import__("g6_support").shared_inputs(), "CALL_001")
    pilot_first = next(iter(pilot_ctx.texts))
    assert render(pilot_ctx, {"turns": [{"speaker": "EMPLOYEE", "text": "好的[[%s]]" % pilot_first}]})[2], "the historical rule is unchanged"


def _spans(ctx, turns):
    parsed = {"turns": turns}
    rendered, alignment, errors = render(ctx, parsed)
    assert not errors
    by_turn = {}
    for span in alignment:
        by_turn.setdefault(span["turn_id"], []).append((span["start"], span["end"]))
    return rendered, alignment, by_turn


def test_the_script_rules_apply_to_the_models_own_words_only(world):
    mandarin, singlish, english = (world.g6.builder.context(CALLS[p]) for p in ("MANDARIN", "SINGLISH", "ENGLISH"))
    first_m, first_s = next(iter(mandarin.texts)), next(iter(singlish.texts))
    speaker_m, speaker_s = mandarin.speakers[first_m], singlish.speakers[first_s]
    rendered, _, spans = _spans(mandarin, [{"speaker": speaker_m, "text": f"[[{first_m}]]"}])
    assert script_errors(mandarin, rendered, spans) == [], "the event's own text is exempt"
    rendered, _, spans = _spans(mandarin, [{"speaker": speaker_m, "text": f"好的 OK [[{first_m}]]"}])
    assert any("Latin letters" in e for e in script_errors(mandarin, rendered, spans))
    rendered, _, spans = _spans(mandarin, [{"speaker": speaker_m, "text": f"好的 7 [[{first_m}]]"}])
    assert any("Arabic digits" in e for e in script_errors(mandarin, rendered, spans))
    rendered, _, spans = _spans(singlish, [{"speaker": speaker_s, "text": f"好的 [[{first_s}]]"}])
    assert any("Chinese characters" in e for e in script_errors(singlish, rendered, spans))
    rendered, _, spans = _spans(singlish, [{"speaker": speaker_s, "text": f"Okay lah [[{first_s}]]"}])
    assert script_errors(singlish, rendered, spans) == []
    assert english.language_requirements["script_rules"]["han_characters"] == "FORBIDDEN"


def test_a_number_cannot_be_extended_by_words_glued_to_its_event():
    turns = [{"turn_id": 0, "speaker": "CLIENT", "text": "数量是七百五十"}]
    span = lambda start, end: [{"event_id": "EV_001", "turn_id": 0, "start": start, "end": end}]  # noqa: E731
    assert numeral_extension_errors(None, turns, span(3, 5)) and "extend" in numeral_extension_errors(None, turns, span(3, 5))[0]  # 七百 | 五十
    assert numeral_extension_errors(None, [{"turn_id": 0, "speaker": "CLIENT", "text": "七百块"}], span(0, 2))  # a currency word added
    assert numeral_extension_errors(None, [{"turn_id": 0, "speaker": "CLIENT", "text": "我想买七百"}], span(3, 5)) == []
    assert numeral_extension_errors(None, [{"turn_id": 0, "speaker": "CLIENT", "text": "五七百"}], span(1, 3))  # prefixed
    assert numeral_extension_errors(None, [{"turn_id": 0, "speaker": "CLIENT", "text": "好的，七百。"}], span(3, 5)) == []


def test_the_length_band_is_counted_in_the_calls_unit_and_rejects_the_wrong_size(world):
    validator, ctx = validator_and_ctx(world, CALLS["MANDARIN"])
    from fyp_evaluation.dialogues.feasibility import synthesize_dialogue
    full = synthesize_dialogue(ctx)
    assert validator.errors(full, ctx) == []
    short = {"turns": [turn for turn in full["turns"] if "[[" in turn["text"]]}  # only the event turns: far below the band
    errors = validator.errors(short, ctx)
    assert any(e.startswith("length:") and "han chars" in e for e in errors)
    assert length_count([{"text": "好的 OK thank 谢谢 7"}], "words") == 5
    assert length_count([{"text": "好的 OK thank 谢谢 7"}], "han_chars") == 4
    assert length_count([{"text": "好的 OK thank 谢谢 7"}], "han_chars_plus_words") == 7


# ============================================================================================ staleness and reconstruction
def test_a_guide_change_stales_only_singlish_dialogues_and_validation_stays_clean(world, tmp_path, monkeypatch):
    directory = tmp_path / "guides"
    directory.mkdir()
    for path in register_guide.GUIDE_DIR.glob("register_singlish.*.md"):
        shutil.copyfile(path, directory / path.name)
    recorded = register_guide.guide_sha256()
    monkeypatch.setattr(register_guide, "GUIDE_DIR", directory)
    (directory / "register_singlish.v1.md").write_text(register_guide.guide_text() + "\nA wording change.\n", encoding="utf-8")
    specs = world.g6.inputs.specs
    singlish = {c for c in world.call_ids if register_guide.applies(specs[c])}
    statuses = world.g6.dependency_status(world.call_ids)
    assert {c for c, s in statuses.items() if s["status"] != "current"} == singlish
    assert all("register_guide_sha256" in statuses[c]["changed"] for c in singlish)
    assert world.g6.validate() == [], "validation rebuilds under the recorded guide version"
    assert artifact(world, sorted(singlish)[0])["source"]["register_guide_sha256"] == recorded
