"""G8 SSML conversion and Polly rendering: offline, fake Polly, isolated copies. No AWS request is ever made."""

import io
import json
import shutil
from pathlib import Path

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.rendering.cli import main
from fyp_evaluation.rendering.config import RenderSettings
from fyp_evaluation.rendering.inputs import RenderCall, load_calls
from fyp_evaluation.rendering.render import (RENDER_MANIFEST, SSML_MANIFEST, TIMINGS, check_render, render_dir, ssml_differences, ssml_outputs,
                                             synthesize, write_render, write_ssml)
from fyp_evaluation.rendering.ssml import assert_faithful, make_ssml

ROOT = Path(__file__).resolve().parents[1]
RELEASE, SCHEMAS = ROOT / "datasets" / "advisory_execution_pilot_v1", ROOT / "schemas"
CALL = "CALL_001"


class FakePolly:
    def __init__(self):
        self.requests = []

    def synthesize_speech(self, **request):
        self.requests.append(request)
        return {"AudioStream": io.BytesIO(b"\x01\x00" * (8000 * (1 + len(self.requests) % 3)))}


def fake_encode(pcm, destination):
    destination.write_bytes(pcm[:8])


def small_call():
    turns = ({"turn_id": 0, "speaker": "EMPLOYEE", "text": "Hello & welcome, the amount is 500k."},
             {"turn_id": 1, "speaker": "CLIENT", "text": "Thank you, my NRIC is S, 9, 1."},
             {"turn_id": 2, "speaker": "EMPLOYEE", "text": "Noted."})
    return RenderCall("CALL_999", turns, "a" * 64, "b" * 64)


def copy_release(destination):
    shutil.copytree(RELEASE, destination, ignore=shutil.ignore_patterns("*.lock", "renders"))
    return destination


# ---- SSML ---------------------------------------------------------------------------------------------------------------------------

def sub(alias, token):
    return f'<sub alias="{alias}">{token}</sub>'


@pytest.mark.parametrize("text, expected", [
    ("The amount is 500k, in yuan.", f"<speak>The amount is {sub('five hundred kay', '500k')}, in yuan.</speak>"),
    ("About 10m.", f"<speak>About {sub('ten em', '10m')}.</speak>"),
    ("50k and 75k", f"<speak>{sub('fifty kay', '50k')} and {sub('seventy-five kay', '75k')}</speak>"),
    ("My NRIC and FIN.", f"<speak>My {sub('en ar eye see', 'NRIC')} and {sub('eff eye en', 'FIN')}.</speak>"),
    ("US dollars via an ETF in SGD.", f"<speak>{sub('you ess', 'US')} dollars via an {sub('ee tee eff', 'ETF')} in {sub('ess gee dee', 'SGD')}.</speak>"),
    ("It is S six five three two five three nine A.", f"<speak>It is {sub('ess', 'S')} six five three two five three nine {sub('ay', 'A')}.</speak>"),
    # "oh", "double" and "triple" are kept exactly as written
    ("F nine nine three three triple two B", f"<speak>{sub('eff', 'F')} nine nine three three triple two {sub('bee', 'B')}</speak>"),
    ("T five seven two double eight oh nine D", f"<speak>{sub('tee', 'T')} five seven two double eight oh nine {sub('dee', 'D')}</speak>"),
    ("S, zero, one, two, three, four, five, six, X", f"<speak>{sub('ess', 'S')}, zero, one, two, three, four, five, six, {sub('ex', 'X')}</speak>"),
    ("Fish & chips <now>", "<speak>Fish &amp; chips &lt;now&gt;</speak>"),
    ("It is 5km away, call 10 m, FINAL, NRICs, a one two plan, I want one hundred A.", "<speak>It is 5km away, call 10 m, FINAL, NRICs, a one two plan, I want one hundred A.</speak>"),
])
def test_ssml_escapes_text_and_adds_only_the_documented_pronunciation_hints(text, expected):
    assert make_ssml(text) == expected
    assert "thousand" not in expected and "million" not in expected and "say-as" not in expected


def test_every_currency_code_the_release_allows_is_spelled_out():
    from fyp_evaluation.rendering.ssml import SPELLED
    allowed = json.loads((RELEASE / "normalization.json").read_text(encoding="utf-8"))["profiles"]["currency_iso_4217_alpha3"]["canonical"]["allowed_values"]
    assert set(allowed) <= set(SPELLED)


def test_every_spoken_identifier_of_the_release_gets_its_letters_spelled_out():
    import re
    _, calls, _ = load_calls(RELEASE, SCHEMAS)
    pattern = re.compile(r"(?:NRIC|FIN|identification number) is [STFGM] (?:\w+ )*[A-Z]\b")
    found = 0
    for call in calls.values():
        for turn in call.turns:
            if pattern.search(turn["text"]):
                found += 1
                assert make_ssml(turn["text"]).count("<sub alias=") >= 2, turn["text"]  # prefix and suffix letter
    assert found >= 45


def test_the_four_assigned_shorthand_amounts_keep_their_audible_suffix():
    """EC_FINANCIAL_SHORTHAND (edge_case_assignments.csv): the audio must say the shorthand, never expand it."""
    _, calls, _ = load_calls(RELEASE, SCHEMAS, ["CALL_010", "CALL_016", "CALL_032", "CALL_034"])
    expected = {"CALL_010": ("500k", "five hundred kay"), "CALL_016": ("10m", "ten em"), "CALL_032": ("50k", "fifty kay"),
                "CALL_034": ("75k", "seventy-five kay")}
    for call_id, (token, alias) in expected.items():
        spoken = [make_ssml(t["text"]) for t in calls[call_id].turns if token in t["text"]]
        assert spoken and all(sub(alias, token) in ssml for ssml in spoken), call_id
    every = " ".join(make_ssml(t["text"]) for call in calls.values() for t in call.turns)
    assert "thousand" not in every and "million" not in every


def test_ssml_must_carry_exactly_the_canonical_text():
    with pytest.raises(ValueError, match="exactly the canonical"):
        assert_faithful("hello", "<speak>hullo</speak>")
    with pytest.raises(ValueError, match="not valid XML"):
        assert_faithful("a\x00b", "<speak>a\x00b</speak>")


@pytest.mark.parametrize("ssml, why", [
    ('<speak><sub alias="banana">500k</sub></speak>', "not the one the pronunciation table prescribes"),  # right text, wrong speech
    ('<speak><sub alias="five hundred thousand">500k</sub></speak>', "not the one the pronunciation table prescribes"),
    ('<speak><sub alias="x">hello</sub></speak>', "not the one the pronunciation table prescribes"),
    ('<speak><sub>500k</sub></speak>', "not a permitted pronunciation hint"),
    ('<speak><sub alias="five hundred kay" role="x">500k</sub></speak>', "not a permitted pronunciation hint"),
    ('<speak><say-as interpret-as="characters">500k</say-as></speak>', "not a permitted pronunciation hint"),
    ('<speak><break time="1s"/>500k</speak>', "not a permitted pronunciation hint"),
    ('<speak><phoneme alphabet="ipa" ph="x">500k</phoneme></speak>', "not a permitted pronunciation hint"),
    ('<speak><prosody rate="slow">500k</prosody></speak>', "not a permitted pronunciation hint"),
    ('<speak volume="loud">500k</speak>', "plain <speak> element"),
    ('<speak><sub alias="five hundred kay"><b>500k</b></sub></speak>', "not a permitted pronunciation hint"),
])
def test_arbitrary_aliases_and_unsupported_tags_are_rejected(ssml, why):
    with pytest.raises(ValueError, match=why):
        assert_faithful("500k", ssml)


def test_the_closed_table_accepts_exactly_its_own_hints_and_refuses_unsupported_amounts():
    assert_faithful("S six A", f"<speak>{sub('ess', 'S')} six {sub('ay', 'A')}</speak>")
    with pytest.raises(ValueError, match="only 0-999"):
        make_ssml("a 5000k fee")
    assert make_ssml("a 2.5k fee") == f"<speak>a {sub('two point five kay', '2.5k')} fee</speak>"


def test_every_verified_turn_of_the_release_converts_to_faithful_ssml():
    _, calls, refused = load_calls(RELEASE, SCHEMAS)
    assert not refused and len(calls) == 50
    turns = 0
    for call in calls.values():
        for name, ssml in ssml_outputs(call, "x").items():
            if name != SSML_MANIFEST:
                turns += 1
        assert [t["turn_id"] for t in call.turns] == list(range(len(call.turns)))
    assert turns == sum(len(c.turns) for c in calls.values()) > 1000


# ---- writing, rendering and checking one call -----------------------------------------------------------------------------------------

def test_ssml_files_are_written_once_and_never_silently_replaced(tmp_path):
    call = small_call()
    assert len(write_ssml(tmp_path, call, "r1")) == 4  # three turns plus the manifest
    assert write_ssml(tmp_path, call, "r1") == []  # idempotent
    path = render_dir(tmp_path, call.call_id, "r1") / "segments" / "1.ssml"
    path.write_text("<speak>other</speak>", encoding="utf-8")
    with pytest.raises(FileExistsError, match="differs"):
        write_ssml(tmp_path, call, "r1")
    assert path.read_text(encoding="utf-8") == "<speak>other</speak>"
    assert write_ssml(tmp_path, call, "r1", overwrite=True) == ["segments/1.ssml"]


def test_polly_receives_the_ssml_on_disk_and_timings_cover_every_turn(tmp_path):
    call, settings, polly = small_call(), RenderSettings(), FakePolly()
    with pytest.raises(ValueError, match="run --ssml first"):
        synthesize(call, "r1", tmp_path, settings, polly)
    assert not polly.requests
    write_ssml(tmp_path, call, "r1")
    segments = synthesize(call, "r1", tmp_path, settings, polly)
    assert [r["TextType"] for r in polly.requests] == ["ssml"] * 3
    assert [r["VoiceId"] for r in polly.requests] == ["Jasmine", "Matthew", "Jasmine"]
    assert polly.requests[0]["Text"] == make_ssml(call.turns[0]["text"]) and polly.requests[0]["Engine"] == "neural"
    audio = write_render(tmp_path, call, "demo", "r1", settings, segments, fake_encode, rendered_at="2026-10-12T00:00:00Z")
    directory = audio.parent
    timings = parse_json_text((directory / TIMINGS).read_text(encoding="utf-8"))
    assert [row["turn_id"] for row in timings["turns"]] == [0, 1, 2]
    assert timings["turns"][1]["start_seconds"] == pytest.approx(timings["turns"][0]["end_seconds"] + 0.4)
    assert sorted(p.name for p in (directory / "segments").glob("*.mp3")) == ["0.mp3", "1.mp3", "2.mp3"]
    manifest = parse_json_text((directory / RENDER_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["dialogue_sha256"] == call.dialogue_sha256 and manifest["settings"]["employee_voice"] == "Jasmine"
    assert check_render(tmp_path, call, "r1", SCHEMAS) == ([], [])
    with pytest.raises(FileExistsError, match="already exists"):
        write_render(tmp_path, call, "demo", "r1", settings, segments, fake_encode, rendered_at="x")


def test_a_render_goes_stale_when_the_dialogue_ssml_or_timings_change(tmp_path):
    call, settings = small_call(), RenderSettings()
    write_ssml(tmp_path, call, "r1")
    write_render(tmp_path, call, "demo", "r1", settings, synthesize(call, "r1", tmp_path, settings, FakePolly()), fake_encode, rendered_at="t")
    directory = render_dir(tmp_path, call.call_id, "r1")
    changed = RenderCall(call.call_id, call.turns, "c" * 64, call.report_sha256)
    assert any("stale" in e for e in check_render(tmp_path, changed, "r1", SCHEMAS)[0])
    timings = json.loads((directory / TIMINGS).read_text(encoding="utf-8"))
    timings["turns"].pop()
    (directory / TIMINGS).write_text(json.dumps(timings), encoding="utf-8")
    errors = check_render(tmp_path, call, "r1", SCHEMAS)[0]
    assert any("does not match the render manifest" in e for e in errors) and any("every dialogue turn" in e for e in errors)
    (directory / "audio.mp3").write_bytes(b"tampered")
    assert any("audio.mp3 does not match" in e for e in check_render(tmp_path, call, "r1", SCHEMAS)[0])
    (directory / "segments" / "2.ssml").unlink()
    assert ssml_differences(tmp_path, call, "r1")[0] == ["segments/2.ssml"]


# ---- the gate and the command line ----------------------------------------------------------------------------------------------------------------

def test_preview_is_read_only_and_reports_eligibility_and_an_approximate_cost(capsys):
    before = sorted(p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob("*") if p.is_file())
    assert main(["--dataset-root", str(RELEASE), "--preview"]) == 0
    out = capsys.readouterr().out
    assert "50 eligible, 0 refused" in out and "an estimate, not a quote" in out
    assert before == sorted(p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob("*") if p.is_file())


def test_a_changed_dialogue_is_refused_and_nothing_is_sent_or_written(tmp_path, capsys):
    root = copy_release(tmp_path / "dataset")
    dialogue = root / "authoring" / "dialogues" / f"{CALL}.json"
    document = json.loads(dialogue.read_text(encoding="utf-8"))
    document["turns"][1]["text"] += " Extra words."
    dialogue.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _, calls, refused = load_calls(root, SCHEMAS, [CALL, "CALL_002"])
    assert CALL in refused and "CALL_002" in calls
    assert main(["--dataset-root", str(root), "--ssml", "--call-id", CALL]) == 0
    assert "NOT ELIGIBLE" in capsys.readouterr().out and not list(root.glob("calls/*/renders"))
    assert main(["--dataset-root", str(root), "--execute", "--call-id", CALL]) == 2
    assert "not eligible" in capsys.readouterr().err and not list(root.glob("calls/*/renders"))


def test_the_command_line_writes_ssml_then_checks_a_render(tmp_path, capsys):
    root = copy_release(tmp_path / "dataset")
    assert main(["--dataset-root", str(root), "--check", "--call-id", CALL]) == 1  # no SSML yet: --check no longer passes silently
    assert "SSML missing or differs" in capsys.readouterr().out
    assert main(["--dataset-root", str(root), "--ssml", "--call-id", CALL]) == 0
    assert "SSML files written" in capsys.readouterr().out
    assert main(["--dataset-root", str(root), "--check", "--call-id", CALL]) == 0
    assert f"{CALL}: SSML OK, no render yet" in capsys.readouterr().out
    turn_file = render_dir(root, CALL, "clean_voice_A") / "segments" / "1.ssml"
    good = turn_file.read_text(encoding="utf-8")
    turn_file.write_text('<speak><sub alias="banana">x</sub></speak>', encoding="utf-8")  # hand-edited SSML, even with no audio yet
    assert main(["--dataset-root", str(root), "--check", "--call-id", CALL]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "segments/1.ssml" in out
    turn_file.write_text(good, encoding="utf-8")
    manifest = render_dir(root, CALL, "clean_voice_A") / "ssml_manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("ssml-v3", "ssml-v2"), encoding="utf-8")  # an old pronunciation version
    assert main(["--dataset-root", str(root), "--check", "--call-id", CALL]) == 1
    assert "ssml_manifest.json" in capsys.readouterr().out
    assert main(["--dataset-root", str(root), "--ssml", "--overwrite", "--call-id", CALL]) == 0
    capsys.readouterr()
    _, calls, _ = load_calls(root, SCHEMAS, [CALL])
    call, settings = calls[CALL], RenderSettings()
    write_render(root, call, "advisory_execution_pilot_v1", "clean_voice_A", settings, synthesize(call, "clean_voice_A", root, settings, FakePolly()),
                 fake_encode, rendered_at="2026-10-12T00:00:00Z")
    assert main(["--dataset-root", str(root), "--check", "--call-id", CALL]) == 0
    assert f"{CALL}: OK" in capsys.readouterr().out
    assert main(["--dataset-root", str(root), "--execute", "--call-id", CALL]) == 2  # a render exists: never replaced silently
    assert "already exists" in capsys.readouterr().err


def test_usage_errors_are_rejected_before_anything_is_read(capsys):
    for argv in (["--preview", "--limit", "2"], ["--execute", "--overwrite", "--skip-existing"], ["--render-id", "1bad"]):
        with pytest.raises(SystemExit) as stopped:
            main(["--dataset-root", str(RELEASE), *argv])
        assert stopped.value.code == 2
    capsys.readouterr()


def test_a_re_render_of_a_shorter_dialogue_leaves_no_stale_turn_audio_or_ssml(tmp_path):
    long_call, settings = small_call(), RenderSettings()
    write_ssml(tmp_path, long_call, "r1")
    write_render(tmp_path, long_call, "demo", "r1", settings, synthesize(long_call, "r1", tmp_path, settings, FakePolly()), fake_encode, rendered_at="t")
    short_call = RenderCall(long_call.call_id, long_call.turns[:2], "d" * 64, long_call.report_sha256)
    write_ssml(tmp_path, short_call, "r1", overwrite=True)
    write_render(tmp_path, short_call, "demo", "r1", settings, synthesize(short_call, "r1", tmp_path, settings, FakePolly()), fake_encode,
                 overwrite=True, rendered_at="t2")
    directory = render_dir(tmp_path, long_call.call_id, "r1")
    assert sorted(p.name for p in (directory / "segments").iterdir()) == ["0.mp3", "0.ssml", "1.mp3", "1.ssml"]
    assert check_render(tmp_path, short_call, "r1", SCHEMAS) == ([], [])
