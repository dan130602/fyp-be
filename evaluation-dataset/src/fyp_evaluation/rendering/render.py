"""G8 render: SSML files, Polly synthesis, stitched audio and per-turn timings for one verified dialogue.

Nothing here reads or writes ground truth. A render is bound to the exact dialogue it was made from (`dialogue_sha256` in
`render_manifest.json`), so any later change to the G6 dialogue makes it stale; `check_render` reports that.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from jsonschema import Draft202012Validator

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.llm.fsutil import write_text_atomic
from fyp_evaluation.planning import outputs as g1

from .config import (APPROX_USD_PER_MILLION_CHARACTERS, BYTES_PER_SECOND, CALLS_DIR, PRONUNCIATION_VERSION, RENDERS_DIR, SAMPLE_RATE,
                     SCHEMA_VERSION, STAGE)
from .ssml import assert_faithful, make_ssml

SSML_MANIFEST, RENDER_MANIFEST, TIMINGS, AUDIO, SEGMENTS = "ssml_manifest.json", "render_manifest.json", "timings.json", "audio.mp3", "segments"


def render_dir(root, call_id, render_id):
    return Path(root) / CALLS_DIR / call_id / RENDERS_DIR / render_id


def billed_characters(call):
    """Polly bills the characters of the spoken text, not SSML tags; the canonical length is the estimate."""
    return sum(len(turn["text"]) for turn in call.turns)


def estimate_cost_usd(characters, engine):
    return round(characters * APPROX_USD_PER_MILLION_CHARACTERS[engine] / 1_000_000, 4)


def ssml_outputs(call, render_id):
    """{path relative to the render directory: text}: one `.ssml` per turn plus the manifest that binds them to the dialogue."""
    files, rows = {}, []
    for turn in call.turns:
        ssml = make_ssml(turn["text"])
        name = f"{SEGMENTS}/{turn['turn_id']}.ssml"
        files[name] = ssml
        rows.append({"turn_id": turn["turn_id"], "speaker": turn["speaker"], "ssml_file": name, "ssml_sha256": text_hash(ssml),
                     "characters": len(turn["text"])})
    files[SSML_MANIFEST] = serialize_json({"schema_version": SCHEMA_VERSION, "stage": STAGE, "call_id": call.call_id, "render_id": render_id,
                                           "dialogue_sha256": call.dialogue_sha256, "text_type": "ssml",
                                           "pronunciation": PRONUNCIATION_VERSION, "turns": rows})
    return files


def _read(path):
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n") if path.is_file() else None


def ssml_differences(root, call, render_id):
    """Relative paths of SSML outputs that are missing or differ from a fresh build, and stray `.ssml` files."""
    directory = render_dir(root, call.call_id, render_id)
    expected = ssml_outputs(call, render_id)
    differing = [name for name, text in expected.items() if _read(directory / name) != text]
    stray = sorted(p.relative_to(directory).as_posix() for p in (directory / SEGMENTS).glob("*.ssml")
                   if p.relative_to(directory).as_posix() not in expected) if (directory / SEGMENTS).is_dir() else []
    return differing, stray


def write_ssml(root, call, render_id, *, overwrite=False):
    """Write the SSML outputs; refuse to replace differing existing files without `overwrite`. Returns the paths written."""
    directory = render_dir(root, call.call_id, render_id)
    differing, stray = ssml_differences(root, call, render_id)
    existing = [name for name in differing if (directory / name).is_file()] + stray
    if existing and not overwrite:
        raise FileExistsError(f"{call.call_id}: SSML differs from the verified dialogue ({existing[0]}); pass --overwrite to replace it")
    for name in stray:
        (directory / name).unlink()
    expected = ssml_outputs(call, render_id)
    for name in differing:
        write_text_atomic(directory / name, expected[name])
    return differing


def encode_mp3(ffmpeg, pcm, destination):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
               "-codec:a", "libmp3lame", "-b:a", "128k", str(destination)]
    result = subprocess.run(command, input=pcm, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode('utf-8', errors='replace')}")


def synthesize(call, render_id, root, settings, polly, *, progress=None):
    """[(turn_id, pcm)] from Polly. The text sent is the SSML file on disk, after it was checked against a fresh build."""
    directory = render_dir(root, call.call_id, render_id)
    differing, stray = ssml_differences(root, call, render_id)
    if differing or stray:
        raise ValueError(f"{call.call_id}: the SSML on disk does not match the verified dialogue; run --ssml first")
    segments = []
    for turn in call.turns:
        response = polly.synthesize_speech(Engine=settings.engine, VoiceId=settings.voice(turn["speaker"]), OutputFormat="pcm",
                                           SampleRate=str(SAMPLE_RATE), TextType="ssml",
                                           Text=(directory / SEGMENTS / f"{turn['turn_id']}.ssml").read_bytes().decode("utf-8").replace("\r\n", "\n"))
        try:
            pcm = response["AudioStream"].read()
        finally:
            response["AudioStream"].close()
        if not pcm or len(pcm) % 2:
            raise RuntimeError(f"Polly returned invalid PCM for {call.call_id} turn {turn['turn_id']}")
        segments.append((turn["turn_id"], pcm))
        if progress:
            progress(turn["turn_id"], len(call.turns), len(pcm) / BYTES_PER_SECOND)
    return segments


def write_render(root, call, dataset_id, render_id, settings, segments, encode, *, overwrite=False, rendered_at):
    """Stitch the segments (with silence between turns), derive timings and write audio, segments and manifests together."""
    directory = render_dir(root, call.call_id, render_id)
    if not overwrite and any((directory / name).exists() for name in (AUDIO, TIMINGS, RENDER_MANIFEST)):
        raise FileExistsError(f"Render already exists: {directory}; pass --overwrite to replace it")
    silence = b"\x00\x00" * (SAMPLE_RATE * settings.silence_ms // 1000)
    stitched, rows = bytearray(), []
    for position, (turn_id, pcm) in enumerate(segments):
        start = len(stitched) / BYTES_PER_SECOND
        stitched.extend(pcm)
        rows.append({"turn_id": turn_id, "start_seconds": round(start, 3), "end_seconds": round(len(stitched) / BYTES_PER_SECOND, 3)})
        if position != len(segments) - 1:
            stitched.extend(silence)
    timings = serialize_json({"schema_version": "0.1.0", "call_id": call.call_id, "render_id": render_id, "audio_file": AUDIO,
                              "duration_seconds": round(len(stitched) / BYTES_PER_SECOND, 3), "turns": rows})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SEGMENTS).mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="polly-", dir=directory) as name:
        temp = Path(name)
        encode(bytes(stitched), temp / AUDIO)
        for turn_id, pcm in segments:
            encode(pcm, temp / f"{turn_id}.mp3")
        audio_sha256 = g1.sha256((temp / AUDIO).read_bytes())
        manifest = serialize_json({
            "schema_version": SCHEMA_VERSION, "stage": STAGE, "dataset_id": dataset_id, "call_id": call.call_id, "render_id": render_id,
            "dialogue_sha256": call.dialogue_sha256, "g7_report_sha256": call.report_sha256,
            "ssml_manifest_sha256": text_hash(ssml_outputs(call, render_id)[SSML_MANIFEST]), "timings_sha256": text_hash(timings),
            "audio_sha256": audio_sha256, "settings": settings.as_record(), "billed_characters_estimate": billed_characters(call),
            "turns": len(call.turns), "rendered_at": rendered_at})
        for stale in (directory / SEGMENTS).glob("*.mp3"):  # a re-render of a changed (e.g. shorter) dialogue must not leave old turn audio behind
            if stale.name not in {f"{turn_id}.mp3" for turn_id, _ in segments}:
                stale.unlink()
        for turn_id, _ in segments:
            os.replace(temp / f"{turn_id}.mp3", directory / SEGMENTS / f"{turn_id}.mp3")
        os.replace(temp / AUDIO, directory / AUDIO)
        write_text_atomic(directory / TIMINGS, timings)
        write_text_atomic(directory / RENDER_MANIFEST, manifest)
    return directory / AUDIO


def check_ssml(root, call, render_id):
    """Errors for the SSML files of one call, valid before any audio exists; reads only.

    Every file must equal a fresh build from the CURRENT verified dialogue, and each on-disk turn file is additionally checked on its
    own: exact visible text and only closed-table pronunciation hints (so a hand-edited file cannot slip through on a hash alone).
    """
    directory = render_dir(root, call.call_id, render_id)
    differing, stray = ssml_differences(root, call, render_id)
    errors = [f"SSML missing or differs from the verified dialogue: {name}" for name in differing] + [f"stray SSML file: {s}" for s in stray]
    for turn in call.turns:
        text = _read(directory / SEGMENTS / f"{turn['turn_id']}.ssml")
        if text is not None:
            try:
                assert_faithful(turn["text"], text)
            except ValueError as error:
                errors.append(f"{SEGMENTS}/{turn['turn_id']}.ssml: {error}")
    return errors


def check_render(root, call, render_id, schema_dir):
    """(errors, warnings) for one render against the CURRENT verified dialogue; reads only.

    Audio is git-ignored, so a missing `audio.mp3` is a warning; a present one must match its recorded hash.
    """
    directory = render_dir(root, call.call_id, render_id)
    errors, warnings = check_ssml(root, call, render_id), []
    try:
        manifest = parse_json_text(_read(directory / RENDER_MANIFEST) or "")
        timings_text = _read(directory / TIMINGS) or ""
        timings = parse_json_text(timings_text)
    except ValueError:
        return errors + ["render_manifest.json or timings.json is missing or unreadable"], warnings
    if manifest.get("dialogue_sha256") != call.dialogue_sha256:
        errors.append("stale: the render was made from a different dialogue")
    if manifest.get("g7_report_sha256") != call.report_sha256:
        errors.append("the render records a different G7 report")
    if manifest.get("timings_sha256") != text_hash(timings_text):
        errors.append("timings.json does not match the render manifest")
    if manifest.get("ssml_manifest_sha256") != text_hash(ssml_outputs(call, render_id)[SSML_MANIFEST]):
        errors.append("the SSML manifest does not match the render manifest")
    schema = json.loads((Path(schema_dir) / "timings.schema.json").read_text(encoding="utf-8"))
    errors += [f"timings.json: {e.json_path}: {e.message}" for e in Draft202012Validator(schema).iter_errors(timings)]
    rows = timings.get("turns", [])
    if [row.get("turn_id") for row in rows] != [turn["turn_id"] for turn in call.turns]:
        errors.append("timings.json does not cover every dialogue turn exactly once, in order")
    previous = 0.0
    for row in rows:
        if row["start_seconds"] < previous or row["end_seconds"] <= row["start_seconds"]:
            errors.append(f"timings.json: turn {row['turn_id']} overlaps the previous turn or is empty")
        previous = row["end_seconds"]
    if rows and timings.get("duration_seconds", 0) < rows[-1]["end_seconds"]:
        errors.append("timings.json: duration is shorter than the last turn")
    audio = directory / AUDIO
    if not audio.is_file():
        warnings.append("audio.mp3 is not present (audio is git-ignored)")
    elif g1.sha256(audio.read_bytes()) != manifest.get("audio_sha256"):
        errors.append("audio.mp3 does not match the render manifest")
    for turn in call.turns:
        if (directory / SEGMENTS / f"{turn['turn_id']}.mp3").is_file() is False and audio.is_file():
            warnings.append(f"segments/{turn['turn_id']}.mp3 is not present")
            break
    return errors, warnings
