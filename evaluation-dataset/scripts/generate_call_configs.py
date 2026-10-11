"""Run G3: generate the call_config.json scenario files for one release (model-assisted; see README).

    python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run
    python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --sync --call-id CALL_001
    python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --submit
    python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --collect
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.call_configs.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
