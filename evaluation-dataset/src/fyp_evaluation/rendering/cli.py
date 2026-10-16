"""G8 command line: SSML conversion and Amazon Polly rendering of the G7-verified dialogues.

    --preview             (default) eligibility, characters and an approximate cost per call; writes nothing, no AWS
    --ssml [--overwrite]  write segments/<turn_id>.ssml and ssml_manifest.json; offline, no AWS
    --check               validate the SSML (even before any audio exists) and any render against the current verified dialogues; offline
    --execute             BILLABLE: send the SSML files to Polly, stitch the audio and write timings.json and render_manifest.json

Only calls whose G7 report is current, intact and VALID are processed; the others are listed with a reason and never sent.
AWS credentials come from the operating-system environment or the project-local .env (see credentials.py); they are never printed, logged or stored.
Exit codes: 0 success, 1 a check failed or a call could not be rendered, 2 input/usage/precondition errors.
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from fyp_evaluation.llm.errors import CredentialsError
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .config import DEFAULT_RENDER_ID, RenderSettings
from .credentials import polly_client
from .inputs import load_calls
from .render import (AUDIO, RENDER_MANIFEST, TIMINGS, billed_characters, check_render, check_ssml, encode_mp3, estimate_cost_usd, render_dir,
                     ssml_differences, synthesize, write_render, write_ssml)


def build_parser():
    defaults = RenderSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--ssml", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true", help="Make billable Amazon Polly requests")
    parser.add_argument("--call-id", action="append", dest="call_ids", metavar="CALL_ID", help="Restrict to a call (repeatable)")
    parser.add_argument("--render-id", default=DEFAULT_RENDER_ID)
    parser.add_argument("--overwrite", action="store_true", help="Replace differing SSML or an existing render")
    parser.add_argument("--skip-existing", action="store_true", help="With --execute: leave calls with a complete, current render alone")
    parser.add_argument("--limit", type=int, help="With --execute: render at most N calls")
    parser.add_argument("--region", default=defaults.region)
    parser.add_argument("--engine", default=defaults.engine, choices=("neural", "standard"))
    parser.add_argument("--employee-voice", default=defaults.employee_voice)
    parser.add_argument("--client-voice", default=defaults.client_voice)
    parser.add_argument("--silence-ms", type=int, default=defaults.silence_ms)
    return parser


def _say(text):
    print(text, flush=True)


def _complete(root, call, render_id, schema_dir):
    directory = render_dir(root, call.call_id, render_id)
    if not all((directory / name).is_file() for name in (AUDIO, TIMINGS, RENDER_MANIFEST)):
        return False
    return not check_render(root, call, render_id, schema_dir)[0]


def _preview(root, calls, refused, args, settings):
    total = 0
    for call_id, call in calls.items():
        differing, stray = ssml_differences(root, call, args.render_id)
        existing = (render_dir(root, call_id, args.render_id) / RENDER_MANIFEST).is_file()
        chars = billed_characters(call)
        total += chars
        _say(f"{call_id}: {len(call.turns)} turns, {chars} characters, ssml {'current' if not (differing or stray) else 'not written'}"
             f"{', render present' if existing else ''}")
    for call_id, reason in refused.items():
        _say(f"{call_id}: NOT ELIGIBLE - {reason}")
    _say(f"{len(calls)} eligible, {len(refused)} refused; {total} characters ~ ${estimate_cost_usd(total, settings.engine)} "
         f"at the approximate {settings.engine} list price (an estimate, not a quote).")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.render_id):
        parser.error("--render-id must start with a letter and contain only letters, digits, _ or -")
    if not 0 <= args.silence_ms <= 5000:
        parser.error("--silence-ms must be between 0 and 5000")
    if args.overwrite and args.skip_existing:
        parser.error("--overwrite and --skip-existing cannot be used together")
    if (args.skip_existing or args.limit is not None) and not args.execute:
        parser.error("--skip-existing and --limit apply to --execute")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    settings = RenderSettings(args.region, args.engine, args.employee_voice, args.client_voice, args.silence_ms)
    root = args.dataset_root
    try:
        runner, calls, refused = load_calls(root, args.schema_dir, args.call_ids)
    except (ValueError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.ssml:
        try:
            for call_id, call in calls.items():
                written = write_ssml(root, call, args.render_id, overwrite=args.overwrite)
                _say(f"{call_id}: {len(call.turns)} turns, {len(written)} SSML files written or updated")
        except (FileExistsError, ValueError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 2
        for call_id, reason in refused.items():
            _say(f"{call_id}: NOT ELIGIBLE - {reason}")
        return 0
    if args.check:
        failed = 0
        for call_id, call in calls.items():
            if (render_dir(root, call_id, args.render_id) / RENDER_MANIFEST).is_file():
                errors, warnings = check_render(root, call, args.render_id, args.schema_dir)  # includes the SSML checks
                label = "OK"
            else:
                errors, warnings, label = check_ssml(root, call, args.render_id), [], "SSML OK, no render yet"
            failed += bool(errors)
            _say(f"{call_id}: {label if not errors else 'FAILED'}" + "".join(f"\n  error: {e}" for e in errors)
                 + "".join(f"\n  note: {w}" for w in warnings))
        for call_id, reason in refused.items():
            _say(f"{call_id}: NOT ELIGIBLE - {reason}")
        return 1 if failed else 0
    if not args.execute:
        _preview(root, calls, refused, args, settings)
        return 0
    return _execute(root, runner, calls, refused, args, settings)


def _execute(root, runner, calls, refused, args, settings):
    if refused:
        for call_id, reason in refused.items():
            print(f"{call_id}: NOT ELIGIBLE - {reason}", file=sys.stderr)
        print("Refusing to render while selected calls are not eligible; narrow the selection with --call-id.", file=sys.stderr)
        return 2
    pending = []
    for call_id, call in calls.items():
        directory = render_dir(args.dataset_root, call_id, args.render_id)
        if args.skip_existing and _complete(args.dataset_root, call, args.render_id, args.schema_dir):
            _say(f"{call_id}: skipping the existing current render")
        elif not args.overwrite and any((directory / name).exists() for name in (AUDIO, TIMINGS, RENDER_MANIFEST)):
            print(f"{call_id}: a render already exists at {directory}; pass --overwrite to replace it or --skip-existing", file=sys.stderr)
            return 2
        else:
            pending.append(call)
    pending = pending[:args.limit] if args.limit else pending
    if not pending:
        _say("No calls need rendering.")
        return 0
    try:
        import boto3
        import imageio_ffmpeg
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        print('Install the rendering dependencies first: python -m pip install -e ".[render]"', file=sys.stderr)
        return 2
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    characters = sum(billed_characters(call) for call in pending)
    _say(f"Rendering {len(pending)} calls, {characters} characters (~${estimate_cost_usd(characters, settings.engine)} estimated).")
    done = []
    try:
        polly = polly_client(settings.region)
        available = {v["Id"] for page in polly.get_paginator("describe_voices").paginate(Engine=settings.engine) for v in page["Voices"]}
        for voice in (settings.employee_voice, settings.client_voice):
            if voice not in available:
                print(f"Voice {voice!r} is unavailable for {settings.engine} in {settings.region}", file=sys.stderr)
                return 2
        for call in pending:
            write_ssml(args.dataset_root, call, args.render_id, overwrite=args.overwrite)
            _say(f"{call.call_id}: {len(call.turns)} turns...")
            segments = synthesize(call, args.render_id, args.dataset_root, settings, polly,
                                  progress=lambda i, n, seconds, c=call: _say(f"  {c.call_id} turn {i + 1}/{n}: {seconds:.2f}s"))
            path = write_render(args.dataset_root, call, runner.inputs.dataset_id, args.render_id, settings, segments,
                                lambda pcm, destination: encode_mp3(ffmpeg, pcm, destination), overwrite=args.overwrite,
                                rendered_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            done.append(call.call_id)
            _say(f"{call.call_id}: saved {path}")
    except CredentialsError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (BotoCoreError, ClientError) as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
        print(f"Amazon Polly request failed after {len(done)} finished calls ({', '.join(done) or 'none'}): {type(error).__name__} {code}".rstrip(), file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Render stopped after {len(done)} finished calls ({', '.join(done) or 'none'}): {error}", file=sys.stderr)
        return 1
    return 0
