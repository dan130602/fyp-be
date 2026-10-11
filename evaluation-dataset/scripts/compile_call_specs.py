"""Run G2A: compile the immutable per-call call_spec.json files for one release.

    python scripts/compile_call_specs.py --dataset-root datasets/advisory_execution_pilot_v1 --overwrite
    python scripts/compile_call_specs.py --dataset-root datasets/advisory_execution_pilot_v1 --validate
    python scripts/compile_call_specs.py --dataset-root datasets/advisory_execution_pilot_v1 --check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.call_specs.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
