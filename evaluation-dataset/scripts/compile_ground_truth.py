"""Run G9: compile the canonical calls/CALL_XXX/{ground_truth,transcript,speaker_roles}.json of a release (offline; see README).

    python scripts/compile_ground_truth.py --dataset-root datasets/advisory_execution_pilot_v1 --write
    python scripts/compile_ground_truth.py --dataset-root datasets/advisory_execution_pilot_v1 --check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.ground_truths.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
