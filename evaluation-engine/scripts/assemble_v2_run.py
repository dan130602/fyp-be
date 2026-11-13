"""Assemble one complete environment_v2 run: the V2 run's new transcripts plus the clean run's transcripts for the skipped clean calls.

    python -B scripts/assemble_v2_run.py --v2-run dev-v2-baseline-1 --clean-run dev175-multilang-1 --out dev-v2-baseline-1-full
    python -B scripts/score.py --run dev-v2-baseline-1-full

Offline; no AWS. Refuses unless the handler file, TRANSCRIBE_* settings and Transcribe job settings of the two runs match.
The new run keeps the V2 run's real dates; copied transcripts keep their original job records and are tagged reused_from_run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_eval.assemble import assemble
from transcribe_eval.collect import RUNS_ROOT
from transcribe_eval.compare import CompareError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-run", required=True)
    parser.add_argument("--clean-run", required=True)
    parser.add_argument("--out", required=True, help="new run id (must not exist)")
    args = parser.parse_args(argv)
    try:
        result = assemble(RUNS_ROOT, args.v2_run, args.clean_run, args.out)
    except CompareError as error:
        print(f"Cannot assemble: {error}")
        return 2
    print(f"assembled run {result['run_id']}: {result['calls']} calls = {result['new_calls']} newly transcribed + "
          f"{result['reused_clean_calls']} clean transcripts reused from {args.clean_run}")
    for text in result["warnings"]:
        print(f"  warning: {text}")
    print(f"next: python -B scripts/score.py --run {result['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
