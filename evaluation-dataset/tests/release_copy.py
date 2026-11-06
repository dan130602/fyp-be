"""Shared, lightweight release copying for the G3-G7 test fixtures.

The committed pilot release is about 630 MB, of which almost everything is rendered audio (`calls/<id>/renders/**`:
mp3, ssml and render manifests). The G3-G7 tests never read audio, yet every isolated copy used to include it, so one
full test run wrote tens of gigabytes. Copies now leave `renders` out unless a test asks for them with `keep_renders=True`.
Production dataset generation and audio behaviour are unchanged; this only affects test fixtures.
"""

import shutil
from pathlib import Path


def is_render_path(relative):
    """True for `calls/<call id>/renders` and everything below it (`relative` is a POSIX path inside the release)."""
    parts = relative.split("/")
    return len(parts) >= 3 and parts[0] == "calls" and parts[2] == "renders"


def copy_tree(source, destination, *, skip_prefixes=(), skip_suffixes=(), keep_renders=False):
    """Copy file contents only (shutil.copytree would copy a OneDrive ReadOnly directory attribute).

    Skips paths that start with a prefix or end with a suffix, and the rendered audio unless `keep_renders`."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if relative.startswith(tuple(skip_prefixes)) or relative.endswith(tuple(skip_suffixes)):
            continue
        if not keep_renders and is_render_path(relative):
            continue
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return destination
