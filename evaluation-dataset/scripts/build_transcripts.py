#!/usr/bin/env python
"""Thin wrapper: build or check transcript.json / speaker_roles.json.

    python scripts/build_transcripts.py --call CALL_001
    python scripts/build_transcripts.py --all
    python scripts/build_transcripts.py --all --check
    python scripts/build_transcripts.py --call CALL_001 --overwrite

See `fyp_evaluation.transcripts.cli` for the reusable implementation and
`fyp_evaluation.transcripts.converter` for the pure conversion logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fyp_evaluation.transcripts.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
