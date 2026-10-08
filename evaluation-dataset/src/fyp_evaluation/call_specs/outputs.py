"""G2A output handling: generate in memory, read disk, write safely, compare.

Everything is built and validated in memory before any file is touched. Files
are replaced one at a time through a temporary name (each replacement is
atomic); `--overwrite` then removes stale extra files from the generated
directory, and `--check` reports any missing, changed or extra file.
"""

import os
from pathlib import Path

from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .compile import RECORD, SPEC_DIR, build_record, build_specs, load_g1_sources, serialize_json, spec_path
from .validate import validate_spec_set


def generate(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Compile and fully validate in memory; return (sources, {path: text}, summary). Writes nothing."""
    src = load_g1_sources(root, schema_dir)
    specs = build_specs(src)
    texts = {spec_path(spec["call_id"]): serialize_json(spec) for spec in specs}
    record = build_record(src.dataset_id, src.hashes, {s["call_id"]: texts[spec_path(s["call_id"])] for s in specs}, specs)
    texts[RECORD] = serialize_json(record)
    if errors := validate_spec_set(src, texts, schema_dir):
        raise ValueError("Compiled call specs failed validation (generator defect):\n" + "\n".join(errors))
    return src, texts, record["summary"]


def read_disk(root):
    """{relative path: text} for the generated directory and record; None marks a non-regular or unreadable file."""
    root = Path(root)
    directory = root / SPEC_DIR
    texts = {}
    if directory.is_symlink():
        raise ValueError(f"Refusing symlink directory: {directory}")

    def read(path):
        if path.is_symlink() or not path.is_file():
            return None
        try:
            return path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            return None

    if directory.is_dir():
        for entry in sorted(directory.iterdir()):
            texts[f"{SPEC_DIR}/{entry.name}"] = read(entry)
    for relative in (RECORD, RECORD + ".tmp"):  # an orphaned record temporary is an interrupted run, not ignorable
        path = root / relative
        if path.exists() or path.is_symlink():
            texts[relative] = read(path)
    return texts


def _temporary(path):
    return path.with_name(path.name + ".tmp")


def _write_atomic(path, text):
    """Write via a temporary name that is never followed if it is a symlink, then replace atomically."""
    temporary = _temporary(path)
    if temporary.is_symlink():
        raise ValueError(f"Refusing symlink temporary path: {temporary}")
    if temporary.exists():
        temporary.unlink()  # orphan from an interrupted run; the preflight guarantees a regular file
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:  # exclusive: never opens an existing path
        stream.write(text)
    os.replace(temporary, path)


def write_specs(root, texts, *, overwrite=False):
    """Write validated texts.

    Preflight first: no output or temporary-output path may be a symlink, and nothing is written
    unless every path is safe. Without overwrite any existing output or orphaned temporary file is
    refused; with overwrite orphaned regular temporary files are replaced and stale extra files in
    the generated directory are removed (unlink never follows a symlink).
    """
    root = Path(root)
    directory, record = root / SPEC_DIR, root / RECORD
    for path in (directory, record, _temporary(record)):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink output path: {path}")
    if _temporary(record).is_dir():
        raise ValueError(f"Unexpected directory at {_temporary(record)}; remove it manually")
    present = list(directory.iterdir()) if directory.is_dir() else []
    if any(entry.is_dir() or entry.is_symlink() for entry in present):
        raise ValueError(f"Unexpected subdirectory or symlink in {directory}; remove it manually")
    if (present or record.exists() or _temporary(record).exists()) and not overwrite:
        raise FileExistsError(f"Outputs or orphaned temporary files already exist under {directory}; use --overwrite explicitly")
    directory.mkdir(parents=True, exist_ok=True)
    for relative, text in texts.items():
        _write_atomic(root / relative, text)
    # Re-list rather than reuse the pre-write snapshot: a stale temporary file from an interrupted run
    # can share a name with this run's own temporary file.
    for entry in directory.iterdir():
        if f"{SPEC_DIR}/{entry.name}" not in texts:
            entry.unlink()
    return [root / relative for relative in texts]


def check_outputs(root, texts):
    """Compare regenerated texts with disk: {'missing': [...], 'changed': [...], 'extra': [...]}."""
    disk = read_disk(root)
    return {
        "missing": sorted(rel for rel in texts if rel not in disk),
        "changed": sorted(rel for rel, text in texts.items() if rel in disk and disk[rel] != text),
        "extra": sorted(rel for rel in disk if rel not in texts),
    }
