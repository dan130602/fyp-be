"""Run G8: SSML conversion and Amazon Polly rendering of the G7-verified dialogues of a release (see README).

    python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --preview
    python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --ssml
    python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --check
    python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --execute --call-id CALL_001   # billable
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.rendering.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
