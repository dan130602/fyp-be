"""Export a scored run to the fyp-ai dashboard as a local bundle (view only). No commit or clean tree required; no MP3s are copied.

    python -B scripts/export_to_fyp_ai.py --run 20261017T021321Z-65063c3e
    python -B scripts/export_to_fyp_ai.py --run <id> --destination ../../fyp-ai/evaluation/stt_runs --overwrite

To play audio next to errors, set STT_AUDIO_ROOT for the dashboard to fyp-be/evaluation-dataset/datasets (see fyp-ai evaluation/README.md).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_eval.collect import RUNS_ROOT
from transcribe_eval.export import DEFAULT_DESTINATION, ExportError, export_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        bundle = export_run(RUNS_ROOT / args.run, args.destination, overwrite=args.overwrite)
    except (ExportError, FileNotFoundError) as error:
        print(f"Export failed: {error}")
        return 2
    print(f"exported {args.run} to {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
