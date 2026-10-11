"""Atomic text writes with the G1/G2A/G2B safety contract (never follow a symlink, exclusive temporary file)."""

import os
import time
from pathlib import Path

REPLACE_ATTEMPTS = 6  # transient Windows sharing violations (OneDrive/antivirus briefly holding a new file)
REPLACE_FIRST_DELAY = 0.05


def temporary_path(path):
    path = Path(path)
    return path.with_name(path.name + ".tmp")


def write_text_atomic(path, text):
    write_bytes_atomic(path, text.encode("utf-8"))


def write_bytes_atomic(path, data):
    """Write exact bytes (no newline translation), atomically."""
    path = Path(path)
    temporary = temporary_path(path)
    for candidate in (path, temporary, path.parent):
        if candidate.is_symlink():
            raise ValueError(f"Refusing symlink path: {candidate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()  # orphan from an interrupted run; it is a regular file because symlinks are refused above
    with temporary.open("xb") as stream:  # exclusive: never opens an existing path
        stream.write(data)
    _replace(temporary, path)


def _replace(source, target):
    """os.replace with a short bounded retry: on Windows a sync client can hold a just-written file for a moment."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_FIRST_DELAY * 2 ** attempt)
