"""Run G1 deterministic dataset planning for one release.

    python scripts/plan_dataset.py --dataset-root datasets/advisory_execution_pilot_v1
    python scripts/plan_dataset.py --dataset-root datasets/advisory_execution_pilot_v1 --check
    python scripts/plan_dataset.py --dataset-root datasets/advisory_execution_pilot_v1 --validate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.planning.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
