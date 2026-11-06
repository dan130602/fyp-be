"""Build or verify the frozen 175/75 DEVELOPMENT/HELDOUT split of a release (offline; see MULTILINGUAL.md).

    python scripts/build_split.py --dataset-root datasets/advisory_execution_multilingual_v1 --write-registry   (local-only evidence)
    python scripts/build_split.py --dataset-root datasets/advisory_execution_multilingual_v1 --write
    python scripts/build_split.py --dataset-root datasets/advisory_execution_multilingual_v1 --check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.splits.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
