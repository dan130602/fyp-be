"""Run G6: generate each call's dialogue around its accepted G4 evidence plan (model-assisted, Batch only; see README).

    python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run
    python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --submit --call-id CALL_027
    python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --collect
    python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --status | --validate | --check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.dialogues.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
