"""G8 command line: compile, check or validate the canonical per-call ground truth of a release (offline, deterministic).

    --write [--overwrite]   gate on a complete, valid, current G6; write calls/CALL_XXX/{ground_truth,transcript,speaker_roles}.json,
                            authoring/ground_truth_record.json and the call_ids of dataset_manifest.json
    --check                 rebuild everything in memory and compare byte for byte (writes nothing)

Exit codes: 0 success, 1 outputs differ or are missing, 2 input/precondition errors.
"""

import argparse
import sys
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.llm.fsutil import write_text_atomic
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .compile import CALL_FILES, CALLS_DIR, DIALOGUE_DIR, RECORD, REPORT_DIR, build_call_files, build_record

MANIFEST = "dataset_manifest.json"


def _text(path):
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def gate(root, schema_dir):
    """The validated G6 inputs; refuses unless G6 is complete, valid and every dialogue is current."""
    from fyp_evaluation.dialogues.run import DialogueRunner

    runner = DialogueRunner(root, schema_dir=schema_dir)
    if errors := runner.validate(require_complete=True):
        raise ValueError("G6 dialogues are not complete and valid:\n" + "\n".join(errors[:20]))
    stale = {c: s for c, s in runner.dependency_status().items() if s["status"] != "current"}
    if stale:
        raise ValueError(f"G6 dialogues are not current: {stale}")
    return runner.inputs


def expected_outputs(root, schema_dir):
    inputs = gate(root, schema_dir)
    files_by_call = {c: build_call_files(inputs, c, _text(root / DIALOGUE_DIR / f"{c}.json"), _text(root / REPORT_DIR / f"{c}.json"), schema_dir)
                     for c in inputs.call_ids}
    outputs = {f"{CALLS_DIR}/{c}/{name}": text for c, files in files_by_call.items() for name, text in files.items()}
    outputs[RECORD] = serialize_json(build_record(inputs, files_by_call))
    manifest = parse_json_text(_text(root / MANIFEST))
    manifest["call_ids"] = list(inputs.call_ids)
    outputs[MANIFEST] = serialize_json(manifest)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="With --write: replace existing differing outputs")
    args = parser.parse_args(argv)
    root = args.dataset_root
    try:
        outputs = expected_outputs(root, args.schema_dir)
    except (ValueError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2
    differing = [p for p, text in outputs.items() if not (root / p).is_file() or _text(root / p) != text]
    if args.check:
        for path in differing:
            print(f"differs or missing: {path}", file=sys.stderr)
        if not differing:
            print(f"G8 outputs reproduce byte-for-byte ({len(outputs)} files).")
        return 1 if differing else 0
    existing = [p for p in differing if (root / p).is_file() and p != MANIFEST]
    if existing and not args.overwrite:
        print("refusing to overwrite differing outputs (use --overwrite):\n" + "\n".join(existing[:20]), file=sys.stderr)
        return 2
    for path in differing:
        write_text_atomic(root / path, outputs[path])
    calls = sum(1 for p in outputs if p.endswith(CALL_FILES[0]))
    print(f"G8 wrote {len(differing)} of {len(outputs)} files ({calls} calls; G7 PENDING, timings PLACEHOLDER_ZERO).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
