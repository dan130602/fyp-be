"""Collect Transcribe output for a dataset's calls through the unmodified production handler.

    python -B scripts/collect.py --dataset legacy_pilot_v1 --all --dry-run     # what would happen; no AWS access
    python -B scripts/collect.py --dataset legacy_pilot_v1 --calls CALL_001      # one call
    python -B scripts/collect.py --dataset legacy_pilot_v1 --all                 # every call (billable Transcribe jobs)

Output goes to runs/<run id>/ (see README). Exit code 0 only when every selected call's job completed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_eval import collect as engine
from transcribe_eval.config import FYP_BE_ROOT, ConfigError, load_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="dataset folder name under evaluation-dataset/datasets (default: DATASET_ROOT setting)")
    parser.add_argument("--render-id", default="clean_voice_A", help="a render folder name, or 'auto' = each call's own single render (mixed Polly mp3 / Gemini+OmniVoice wav release)")
    pick = parser.add_mutually_exclusive_group(required=True)
    pick.add_argument("--all", action="store_true")
    pick.add_argument("--split", choices=("DEVELOPMENT", "HELDOUT"), help="every call the frozen splits/split.v1.json assigns to this label")
    pick.add_argument("--calls", nargs="+", metavar="CALL_ID")
    parser.add_argument("--run-id")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--timeout-minutes", type=float, default=30)
    parser.add_argument("--keep-uploads", action="store_true", help="leave the run's temporary S3 copies in place")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; touch nothing")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Configuration problem: {error}")
        return 2
    dataset_root = FYP_BE_ROOT / "evaluation-dataset" / "datasets" / args.dataset if args.dataset else settings.dataset_root
    try:
        if args.split:
            assignments = json.loads((dataset_root / "splits" / "split.v1.json").read_text(encoding="utf-8"))["assignments"]
            call_ids = sorted(c for c, v in assignments.items() if v["split"] == args.split)
        else:
            call_ids = engine.discover_calls(dataset_root) if args.all else args.calls
        planned = engine.plan(dataset_root, args.render_id, call_ids, settings.bucket)
    except engine.CollectError as error:
        print(f"Cannot collect: {error}")
        return 2
    print(f"dataset {planned['dataset_id']}, render {args.render_id}, bucket {settings.bucket}, region {settings.region}")
    for row in planned["calls"]:
        print(f"  {row['call_id']}: {row['bytes']:,} bytes, {row['audio_seconds']}s, sha256 {row['sha256'][:12]} -> {row['master_key']}")
    print(f"  total audio about {planned['audio_minutes']} minutes")
    if args.dry_run:
        print("dry run: nothing uploaded, no jobs started")
        return 0

    import boto3

    from transcribe_eval.handler_loader import load_handler

    settings.apply_to_environ()
    handler = load_handler(settings.handler_path)
    s3 = boto3.Session(region_name=settings.region).client("s3")
    try:
        metadata = engine.collect(settings, handler, s3, dataset_root=dataset_root, render_id=args.render_id, call_ids=call_ids,
                                  run_id=args.run_id, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_minutes * 60,
                                  keep_uploads=args.keep_uploads)
    except Exception as error:  # noqa: BLE001 - metadata.json already records it
        print(f"collect failed: {type(error).__name__}: {error}")
        return 1
    print(f"\nrun {metadata['run_id']}: {metadata['status']}")
    for entry in metadata["calls"]:
        print(f"  {entry['call_id']}: {entry['status']}" + (f" ({entry['failure_reason']})" if entry["failure_reason"] else ""))
    print(f"saved to {engine.RUNS_ROOT / metadata['run_id']}")
    return 0 if metadata["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
