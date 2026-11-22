"""Frozen copies of the experiment runtime modules (overlap, retry_plan, retry_merge, formatting), byte-identical to the git blobs listed in SOURCES.json.

They import each other as top-level modules (`import overlap as o` in retry_merge.py), exactly as in the experiments, so this directory is put on sys.path once.
Never edit these files: a different strategy version is a new export (tools/export_vendored.py) followed by a replay parity run.
"""
import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent


def ensure_on_path():
    entry = str(DIRECTORY)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return DIRECTORY


ensure_on_path()
