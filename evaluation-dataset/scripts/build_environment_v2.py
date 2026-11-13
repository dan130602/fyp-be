"""Build or check the environment_v2 audio variants (offline; see MULTILINGUAL.md).

    python scripts/build_environment_v2.py init
    python scripts/build_environment_v2.py assign --write       (once)   |   assign --check
    python scripts/build_environment_v2.py preview              (listening gate; DEVELOPMENT clips only)
    python scripts/build_environment_v2.py freeze               (after listening; pins the assignments hash)
    python scripts/build_environment_v2.py build --split DEVELOPMENT   |   build --split HELDOUT
    python scripts/build_environment_v2.py verify
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.augmentation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
