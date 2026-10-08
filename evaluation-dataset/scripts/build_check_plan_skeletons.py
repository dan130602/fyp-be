"""Run G2B: construct the deterministic check_plan_skeleton.json files for one release.

    python scripts/build_check_plan_skeletons.py --dataset-root datasets/advisory_execution_pilot_v1 --overwrite
    python scripts/build_check_plan_skeletons.py --dataset-root datasets/advisory_execution_pilot_v1 --validate
    python scripts/build_check_plan_skeletons.py --dataset-root datasets/advisory_execution_pilot_v1 --check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.check_plan_skeletons.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
