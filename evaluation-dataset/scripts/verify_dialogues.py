"""Run G7: verify each call's accepted G6 dialogue against the frozen gold and assigned edge cases (Gemini; see README).

    python scripts/verify_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run
    python scripts/verify_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --run --call-id CALL_016
    python scripts/verify_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.dialogue_verifications.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
