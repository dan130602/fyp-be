"""CLI: build or check transcript.json / speaker_roles.json from ground truth.

Deterministic and local only: no LLM, Polly, AWS Transcribe, Bedrock or other
network call. Exit codes follow the existing `fyp-validate` convention: 0 on
success; 1 when `--check` loads existing files fine but finds them
cross-file inconsistent with their ground truth; 2 for input/usage errors
(missing files, unreadable JSON) and for a build that cannot produce a valid
pair (mirroring `compile-ground-truth`'s exception-based failure mode).
"""

import argparse
import json
import sys
from pathlib import Path

from fyp_evaluation.validation.schemas import load_json, validate
from fyp_evaluation.validation.transcript import validate_transcript_pair

from .converter import build_transcript_artifacts

DEFAULT_CALLS_DIR = Path("evaluation_dataset/calls")
DEFAULT_SCHEMA_DIR = Path("schemas")
OUTPUT_NAMES = ("transcript.json", "speaker_roles.json")


def _discover_calls(calls_dir):
    return sorted(path.parent.name for path in Path(calls_dir).glob("*/ground_truth.json"))


def _serialize(document):
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write_atomic(path, serialized):
    """Write via a same-directory temp file plus rename, refusing symlinks."""
    if path.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {path}")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.is_symlink():
        raise ValueError(f"Refusing to use occupied temporary path: {tmp}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(serialized, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _build_call(call_id, calls_dir, schema_dir):
    """Load, validate and convert one call. Never writes; raises ValueError,
    naming the call and the failing field/document, on any problem."""
    call_dir = Path(calls_dir) / call_id
    gt_path = call_dir / "ground_truth.json"
    if not gt_path.exists():
        raise ValueError(f"{call_id}: no ground_truth.json at {gt_path}")
    ground_truth = load_json(gt_path)
    if ground_truth.get("call_id") != call_id:
        raise ValueError(
            f"{call_id}: ground_truth.call_id {ground_truth.get('call_id')!r} does not match its directory"
        )
    if errors := validate(ground_truth, "ground_truth", schema_dir):
        raise ValueError(f"{call_id}: invalid ground_truth.json:\n  " + "\n  ".join(errors))
    transcript, speaker_roles = build_transcript_artifacts(ground_truth, schema_dir=schema_dir)
    if errors := validate_transcript_pair(ground_truth, transcript, speaker_roles, schema_dir):
        raise ValueError(f"{call_id}: generated pair failed cross-file validation:\n  " + "\n  ".join(errors))
    return ground_truth, transcript, speaker_roles


def _check_call(call_id, calls_dir, schema_dir):
    """Load existing outputs and cross-validate them. Returns an error list
    (empty means valid); raises ValueError only for missing/usage problems."""
    call_dir = Path(calls_dir) / call_id
    gt_path = call_dir / "ground_truth.json"
    if not gt_path.exists():
        raise ValueError(f"{call_id}: no ground_truth.json at {gt_path}")
    if missing := [name for name in OUTPUT_NAMES if not (call_dir / name).exists()]:
        raise ValueError(f"{call_id}: missing {', '.join(missing)}")
    ground_truth = load_json(gt_path)
    transcript = load_json(call_dir / "transcript.json")
    speaker_roles = load_json(call_dir / "speaker_roles.json")
    return validate_transcript_pair(ground_truth, transcript, speaker_roles, schema_dir)


def _build_command(call_ids, calls_dir, schema_dir, overwrite):
    # Build and validate every selected call before writing anything, so one
    # malformed call cannot leave the dataset with a partially updated,
    # mismatched transcript/speaker-role pair for another call.
    built = {}
    failures = []
    for call_id in call_ids:
        try:
            built[call_id] = _build_call(call_id, calls_dir, schema_dir)
        except ValueError as error:
            failures.append(str(error))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2

    targets = []
    for call_id in call_ids:
        call_dir = Path(calls_dir) / call_id
        _, transcript, speaker_roles = built[call_id]
        for name, document in zip(OUTPUT_NAMES, (transcript, speaker_roles)):
            path = call_dir / name
            if path.is_symlink():
                print(f"Refusing to write through symlink: {path}", file=sys.stderr)
                return 2
            if path.exists() and not overwrite:
                print(f"Refusing to overwrite {path}; pass --overwrite explicitly", file=sys.stderr)
                return 2
            targets.append((call_id, path, document))

    for call_id in call_ids:
        written_now = []
        try:
            for target_call_id, path, document in targets:
                if target_call_id != call_id:
                    continue
                pre_existing = path.exists()
                _write_atomic(path, _serialize(document))
                if not pre_existing:
                    written_now.append(path)
        except OSError as error:
            # Best-effort rollback of files this call newly created, so a
            # failed second write never leaves a mismatched pair for this
            # call. A file that existed before this run and was overwritten
            # cannot be restored without a separate backup, which this local,
            # deterministic converter does not keep.
            for path in written_now:
                path.unlink(missing_ok=True)
            print(f"{call_id}: {error}", file=sys.stderr)
            return 2

    for call_id in call_ids:
        turns = len(built[call_id][0]["dialogue"])
        print(f"{call_id}: wrote {', '.join(OUTPUT_NAMES)} ({turns} turns, {turns} audio segments)")
    return 0


def _check_command(call_ids, calls_dir, schema_dir):
    # Check every selected call before deciding an exit code, so one missing
    # or invalid call never silently hides another call's result. A call
    # this can't even load (missing files, unreadable JSON) is a usage
    # problem (exit 2); a call that loads but disagrees with its ground
    # truth is a validation failure (exit 1), matching the rest of this
    # project's CLI convention.
    usage_failures = []
    content_failures = []
    for call_id in call_ids:
        try:
            errors = _check_call(call_id, calls_dir, schema_dir)
        except ValueError as error:
            usage_failures.append(str(error))
            continue
        if errors:
            content_failures.append(f"{call_id}:\n  " + "\n  ".join(errors))
        else:
            print(f"{call_id}: valid")
    if usage_failures:
        print("\n".join(usage_failures), file=sys.stderr)
        return 2
    if content_failures:
        print("\n".join(content_failures), file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--calls-dir", type=Path, default=DEFAULT_CALLS_DIR)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--call", metavar="CALL_ID", help="Convert/check a single call")
    selection.add_argument("--all", action="store_true", help="Convert/check every discovered call")
    parser.add_argument("--check", action="store_true",
                        help="Read-only: validate existing transcript.json/speaker_roles.json; write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace existing transcript.json/speaker_roles.json")
    args = parser.parse_args(argv)

    if args.all:
        call_ids = _discover_calls(args.calls_dir)
        if not call_ids:
            print(f"No calls with ground_truth.json found under {args.calls_dir}", file=sys.stderr)
            return 2
    else:
        call_ids = [args.call]

    if args.check:
        return _check_command(call_ids, args.calls_dir, args.schema_dir)
    return _build_command(call_ids, args.calls_dir, args.schema_dir, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
