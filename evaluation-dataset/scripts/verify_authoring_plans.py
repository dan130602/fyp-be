"""Run G5: verify each call's authoring plan (G3 call config + G4 check plan) against the frozen gold (Gemini; see README).

    python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run
    python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --run --call-id CALL_021
    python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --run
    python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.plan_verifications.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
