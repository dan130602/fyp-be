"""Render canonical call dialogues with Amazon Polly without changing gold data.

Preview by default. Pass --execute to make billable Polly requests.
Requires boto3 and imageio-ffmpeg for execution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
CALLS = ROOT / "evaluation_dataset" / "calls"
SAMPLE_RATE = 16000  # Polly PCM is signed, little-endian, 16-bit mono.
BYTES_PER_SECOND = SAMPLE_RATE * 2


def make_ssml(text: str) -> str:
    """Return the exact SSML sent to Polly for one canonical dialogue turn."""
    return f"<speak>{escape(text)}</speak>"


def load_turns(call_id: str) -> list[dict]:
    path = CALLS / call_id / "ground_truth.json"
    if not path.is_file():
        raise ValueError(f"Canonical ground truth is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = data.get("dialogue")
    if data.get("call_id") != call_id or not isinstance(turns, list) or not turns:
        raise ValueError(f"Invalid canonical dialogue in {path}")
    for index, turn in enumerate(turns):
        if (turn.get("turn_id") != index or turn.get("speaker") not in {"EMPLOYEE", "CLIENT"}
                or not isinstance(turn.get("text"), str) or not turn["text"].strip()):
            raise ValueError(f"Invalid turn {index} in {path}")
        if len(turn["text"]) > 3000:
            raise ValueError(f"Turn {index} exceeds Polly's billed-character limit")
    return turns


def encode_mp3(ffmpeg: str, pcm: bytes, destination: Path) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
               "-codec:a", "libmp3lame", "-b:a", "128k", str(destination)]
    result = subprocess.run(command, input=pcm, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode('utf-8', errors='replace')}")


def render_call(call_id: str, turns: list[dict], args: argparse.Namespace,
                polly: object, ffmpeg: str) -> Path:
    output_dir = CALLS / call_id / "renders" / args.render_id
    audio_path = output_dir / "audio.mp3"
    timings_path = output_dir / "timings.json"
    if not args.overwrite and (audio_path.exists() or timings_path.exists()):
        raise FileExistsError(f"Render already exists: {output_dir}; pass --overwrite to replace it")

    silence = b"\x00\x00" * (SAMPLE_RATE * args.silence_ms // 1000)
    voices = {"EMPLOYEE": args.employee_voice, "CLIENT": args.client_voice}
    all_pcm = bytearray()
    timing_rows = []
    segment_pcm = []
    for turn in turns:
        turn_id = turn["turn_id"]
        text = make_ssml(turn["text"])
        response = polly.synthesize_speech(
            Engine=args.engine, VoiceId=voices[turn["speaker"]],
            OutputFormat="pcm", SampleRate=str(SAMPLE_RATE),
            TextType="ssml", Text=text,
        )
        try:
            pcm = response["AudioStream"].read()
        finally:
            response["AudioStream"].close()
        if not pcm or len(pcm) % 2:
            raise RuntimeError(f"Polly returned invalid PCM for {call_id} turn {turn_id}")
        start = len(all_pcm) / BYTES_PER_SECOND
        all_pcm.extend(pcm)
        end = len(all_pcm) / BYTES_PER_SECOND
        timing_rows.append({"turn_id": turn_id, "start_seconds": round(start, 3),
                            "end_seconds": round(end, 3)})
        segment_pcm.append((turn_id, pcm))
        if turn_id != len(turns) - 1:
            all_pcm.extend(silence)
        print(f"  {call_id} turn {turn_id + 1}/{len(turns)}: {end - start:.2f}s", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="polly-", dir=output_dir) as temp_name:
        temp = Path(temp_name)
        encode_mp3(ffmpeg, bytes(all_pcm), temp / "audio.mp3")
        for turn_id, pcm in segment_pcm:
            encode_mp3(ffmpeg, pcm, temp / f"{turn_id}.mp3")
        timings = {"schema_version": "0.1.0", "call_id": call_id,
                   "render_id": args.render_id, "audio_file": "audio.mp3",
                   "duration_seconds": round(len(all_pcm) / BYTES_PER_SECOND, 3),
                   "turns": timing_rows}
        (temp / "timings.json").write_text(
            json.dumps(timings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for turn_id, _ in segment_pcm:
            os.replace(temp / f"{turn_id}.mp3", segment_dir / f"{turn_id}.mp3")
        os.replace(temp / "audio.mp3", audio_path)
        os.replace(temp / "timings.json", timings_path)
    return audio_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true", help="Render all canonical calls")
    choice.add_argument("--call", help="Render one call, for example CALL_001")
    parser.add_argument("--execute", action="store_true", help="Make billable Polly requests")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--engine", default="neural", choices=("neural", "standard"))
    parser.add_argument("--employee-voice", default="Jasmine")
    parser.add_argument("--client-voice", default="Matthew")
    parser.add_argument("--render-id", default="clean_voice_A")
    parser.add_argument("--silence-ms", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="When rendering all calls, leave finished renders untouched")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.render_id):
        parser.error("--render-id must start with a letter and contain only letters, digits, _ or -")
    if args.call and not re.fullmatch(r"CALL_[0-9]{3,}", args.call):
        parser.error("--call must be a canonical ID such as CALL_001")
    if not 0 <= args.silence_ms <= 5000:
        parser.error("--silence-ms must be between 0 and 5000")
    if args.overwrite and args.skip_existing:
        parser.error("--overwrite and --skip-existing cannot be used together")
    call_ids = [p.name for p in sorted(CALLS.glob("CALL_*")) if p.is_dir()] if args.all else [args.call]
    try:
        calls = [(call_id, load_turns(call_id)) for call_id in call_ids]
        if args.skip_existing:
            pending = []
            for call_id, turns in calls:
                output = CALLS / call_id / "renders" / args.render_id
                if (output / "audio.mp3").exists() and (output / "timings.json").exists():
                    print(f"{call_id}: skipping existing render")
                else:
                    pending.append((call_id, turns))
            calls = pending
        for call_id, turns in calls:
            print(f"{call_id}: {len(turns)} turns -> {CALLS / call_id / 'renders' / args.render_id / 'audio.mp3'}")
        if not calls:
            print("No calls need rendering.")
            return 0
        if not args.execute:
            print("Preview only. Pass --execute to synthesize audio with Amazon Polly.")
            return 0
        if not args.overwrite:
            for call_id, _ in calls:
                output = CALLS / call_id / "renders" / args.render_id
                if (output / "audio.mp3").exists() or (output / "timings.json").exists():
                    raise FileExistsError(f"Render already exists: {output}; pass --overwrite to replace it")
        try:
            import boto3
            import imageio_ffmpeg
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as error:
            raise RuntimeError('Install rendering dependencies: python -m pip install -e ".[render]"') from error

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        polly = boto3.Session(region_name=args.region).client("polly")
        try:
            available = {voice["Id"] for page in polly.get_paginator("describe_voices").paginate(Engine=args.engine)
                         for voice in page["Voices"]}
            for voice in (args.employee_voice, args.client_voice):
                if voice not in available:
                    raise ValueError(f"Voice {voice!r} is unavailable for {args.engine} in {args.region}")
            for call_id, turns in calls:
                print(f"Rendering {call_id}...", flush=True)
                print(f"Saved {render_call(call_id, turns, args, polly, ffmpeg)}", flush=True)
        except (BotoCoreError, ClientError) as error:
            raise RuntimeError(f"AWS Polly request failed: {error}") from error
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Render failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
