"""Export the prepared dataset; creates destination folders automatically."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.exporting import main

if __name__ == "__main__":
    raise SystemExit(main())
