"""One command for a whole evaluation: transcribe, score, (compare), export. Stops at the first step that fails. Never publishes.

    python -B scripts/run_eval.py --run-id dev-clean-2                                               # the original recordings
    python -B scripts/run_eval.py --run-id dev-v2-1 --audio-condition environment_v2 --clean-run dev175-multilang-1
    python -B scripts/run_eval.py --run-id dev-v2-1 --audio-condition environment_v2 --clean-run dev175-multilang-1 --skip-clean-controls
    python -B scripts/run_eval.py ... --dry-run          # print the steps; the collect step runs with --dry-run (no AWS, no cost)

A noisy run is exported under its own dataset card ("... (noisy and channel-degraded)") beside the clean baseline.
--skip-clean-controls does not send the clean (C1) calls again: their earlier transcripts from --clean-run are added (about US$4 cheaper).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transcribe_eval.pipeline import DEFAULT_DATASET, PipelineError, final_run_id, plan_steps, publish_hint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="a new id for the transcribed run (never reuse one)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--audio-condition", choices=("clean", "environment_v2"), default="clean")
    parser.add_argument("--clean-run", help="an earlier scored clean run: the paired comparison's baseline (needed for environment_v2)")
    parser.add_argument("--skip-clean-controls", action="store_true")
    parser.add_argument("--calls", nargs="+", metavar="CALL_ID", help="only these calls (a canary); default is every call")
    parser.add_argument("--timeout-minutes", type=float, default=120)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        steps = plan_steps(args.run_id, dataset=args.dataset, audio_condition=args.audio_condition, skip_clean_controls=args.skip_clean_controls,
                           clean_run=args.clean_run, calls=args.calls, timeout_minutes=args.timeout_minutes, compare=not args.no_compare, export=not args.no_export)
    except PipelineError as error:
        print(f"Cannot plan this run: {error}")
        return 2
    if args.dry_run:
        command = steps[0][1]
        print("dry run: only the collect step is run, with --dry-run (nothing is uploaded). The steps would be:", flush=True)
        for number, (what, line) in enumerate(steps, 1):
            print(f"  {number}. {what}: {' '.join(line[2:])}", flush=True)
        return subprocess.run(command + ["--dry-run"], cwd=ROOT).returncode
    for number, (what, line) in enumerate(steps, 1):
        print(f"\n[{number}/{len(steps)}] {what}\n    {' '.join(line[2:])}", flush=True)
        code = subprocess.run(line, cwd=ROOT).returncode
        if code != 0:
            print(f"\nStopped: step {number} ({what}) failed with exit code {code}. Nothing after it was run.")
            return code
    print(f"\nDone. Results: runs/{final_run_id(args.run_id, args.audio_condition, args.skip_clean_controls)}/")
    print(publish_hint(final_run_id(args.run_id, args.audio_condition, args.skip_clean_controls)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
