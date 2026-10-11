"""G2B output handling: generate in memory, read disk, write safely, compare.

Same safety contract as G2A: everything is built and validated in memory first; no output or
temporary-output path is ever followed if it is a symlink; temporary files are created
exclusively; an orphaned temporary file is detected by --validate/--check and replaced only
under --overwrite; stale extra files are removed from the generated directory.
"""

import os
from pathlib import Path

from fyp_evaluation.call_specs.compile import serialize_json
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .compile import RECORD, SPEC_DIR, build_record, build_skeletons, skeleton_path
from .inputs import load_inputs
from .validate import validate_skeleton_set


def generate(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Gate, compile and fully validate in memory; return (inputs, {path: text}, summary). Writes nothing."""
    inputs = load_inputs(root, schema_dir)
    skeletons = build_skeletons(inputs)
    skeleton_texts = {sk["call_id"]: serialize_json(sk) for sk in skeletons}
    record = build_record(inputs, skeletons, skeleton_texts)
    texts = {skeleton_path(call_id): text for call_id, text in skeleton_texts.items()}
    texts[RECORD] = serialize_json(record)
    if errors := validate_skeleton_set(inputs, texts, schema_dir):
        raise ValueError("Compiled skeletons failed validation (generator defect):\n" + "\n".join(errors))
    return inputs, texts, record["summary"]


def read_disk(root):
    """{relative path: text} for the generated directory, the record and its temporary file; None marks an unreadable path."""
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
    for relative in (RECORD, RECORD + ".tmp"):
        path = root / relative
        if path.exists() or path.is_symlink():
            texts[relative] = read(path)
    return texts


def _temporary(path):
    return path.with_name(path.name + ".tmp")


def _write_atomic(path, text):
    temporary = _temporary(path)
    if temporary.is_symlink():
        raise ValueError(f"Refusing symlink temporary path: {temporary}")
    if temporary.exists():
        temporary.unlink()  # orphan from an interrupted run; the preflight guarantees a regular file
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:  # exclusive: never opens an existing path
        stream.write(text)
    os.replace(temporary, path)


def write_skeletons(root, texts, *, overwrite=False):
    """Write validated texts after a full preflight (symlinks, directories, existing outputs, orphans)."""
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
    for entry in directory.iterdir():  # re-list: a stale temporary file can share a name with this run's own
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
