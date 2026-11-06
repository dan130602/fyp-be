"""Export validated JSON data (committed by default; --allow-uncommitted for a provisional export); no Git mutations or model calls."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .validation.schemas import load_json, validate_answers, validate_checklist
from .validation.transcript import validate_transcript_pair

FORMAT = "fyp-nlp-dataset-v1"
CALL_FILES = ("ground_truth.json", "transcript.json", "speaker_roles.json")
DEFAULT_DATASET_ROOT = Path("datasets/legacy_pilot_v1")
DATASET_MANIFEST_VERSION = "0.1.0"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def serialize(document):
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def no_links(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Refusing symlink/junction: {part}")


def resolve_dataset_root(project, dataset_root=DEFAULT_DATASET_ROOT):
    project = Path(project).resolve()
    dataset = Path(dataset_root)
    return (dataset if dataset.is_absolute() else project / dataset).resolve()


G9_RECORD = "authoring/ground_truth_record.json"


def is_g9_release(dataset):
    """A release whose calls/ were compiled by G9 (`fyp_evaluation.ground_truths`); it owns its normalization.json."""
    return (Path(dataset) / G9_RECORD).is_file()


def _g9_errors(dataset, schemas, paths):
    """G9 releases: every exported call file must equal a fresh G9 compilation from the CURRENT accepted inputs.

    The compiler itself refuses unless G6 (and so G0-G5) is complete, valid and current, and it re-checks every evidence span
    against the plan, so an exported snapshot can never be stale or hand-edited.
    """
    from .ground_truths.cli import expected_outputs
    from .ground_truths.timings import recorded_render_id

    # a FINAL record (audio-derived timings) names its G8 render; the in-memory recompile then uses the same render and the same gates
    expected = expected_outputs(dataset, schemas, recorded_render_id(dataset))
    errors = []
    for name, path in paths.items():
        if name.startswith("calls/") and path.read_bytes().decode("utf-8").replace("\r\n", "\n") != expected.get(name):
            errors.append(f"{name}: differs from a fresh G9 compilation; run scripts/compile_ground_truth.py --write first")
    return errors


SPLIT_FILES = ("splits/split.v1.json", "splits/development_exposure.v1.json", "splits/split_report.v1.json")


def split_selection(dataset, schemas, label):
    """The call IDs a frozen split assigns to `label`, after the full offline verification of the split (stale, altered or missing
    assignments, fingerprints, quotas and locks). The manifest is the sole authority; nothing here draws or edits assignments."""
    from .splits import SplitError
    from .splits.build import manifest_hash, split_calls, verify

    errors = verify(dataset, schemas)
    if errors:
        raise ValueError("The split is missing, altered or stale: " + "; ".join(errors[:5]))
    manifest = load_json(dataset / "splits" / "split.v1.json")
    try:
        call_ids = split_calls(manifest, label)
    except SplitError as error:
        raise ValueError(str(error)) from error
    return {"label": label, "call_ids": call_ids, "split_version": manifest["split_version"], "split_manifest_sha256": manifest_hash(dataset),
            "dataset_calls": len(manifest["assignments"]), "totals": manifest["totals"], "seed": manifest["seed"]}


def prepare_snapshot(project, dataset_root=DEFAULT_DATASET_ROOT, *, allow_uncommitted=False, split=None):
    """Validate the prepared source and return deterministic snapshot bytes.

    By default every input must match committed HEAD. `allow_uncommitted=True` skips that comparison (for a provisional
    export); the manifest then records `source.uncommitted: true`, because `source.git_commit` no longer identifies the data.
    `split` ("DEVELOPMENT" or "HELDOUT") exports only the calls the frozen split assigns to it, in the same v1 layout; the manifest then
    carries a `split` object (label, split manifest hash). Without it the export is byte-identical to before.
    """
    project = Path(project).resolve()
    repo = Path(git(project, "rev-parse", "--show-toplevel").decode().strip())
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    dataset = resolve_dataset_root(project, dataset_root)
    schemas = project / "schemas"
    g9 = is_g9_release(dataset)
    final_render = None
    if g9:
        from .ground_truths.timings import recorded_render_id

        final_render = recorded_render_id(dataset)  # None: PROVISIONAL placeholder times; else the render whose times were stitched in
    calls = sorted(p for p in (dataset / "calls").iterdir() if p.is_dir())
    if not calls:
        raise ValueError("No call directories found")
    source_manifest_path = dataset / "dataset_manifest.json"
    source_manifest = load_json(source_manifest_path)
    required_manifest_keys = {"schema_version", "dataset_id", "checklist_id", "call_ids"}
    if set(source_manifest) != required_manifest_keys:
        raise ValueError("dataset_manifest.json must contain exactly schema_version, dataset_id, checklist_id and call_ids")
    if source_manifest["schema_version"] != DATASET_MANIFEST_VERSION:
        raise ValueError(f"Unsupported dataset manifest version: {source_manifest['schema_version']}")
    for key in ("dataset_id", "checklist_id"):
        if not isinstance(source_manifest[key], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", source_manifest[key]):
            raise ValueError(f"dataset_manifest.json {key} must be a lowercase identifier")
    discovered_call_ids = [call.name for call in calls]
    if source_manifest["call_ids"] != discovered_call_ids:
        raise ValueError("dataset_manifest.json call_ids must exactly match sorted call directories")
    selection = split_selection(dataset, schemas, split) if split else None
    if selection:
        if selection["dataset_calls"] != len(discovered_call_ids):
            raise ValueError("The split assignments do not cover exactly the calls of the release")
        wanted = set(selection["call_ids"])
        calls = [call for call in calls if call.name in wanted]
        discovered_call_ids = [call.name for call in calls]
        if discovered_call_ids != selection["call_ids"]:
            raise ValueError("The split names calls that are not compiled in this release")

    paths = {"checklist.json": dataset / "checklist.json"}
    for call in calls:
        if not re.fullmatch(r"CALL_[0-9]{3,}", call.name):
            raise ValueError(f"Unexpected call directory: {call.name}")
        for name in CALL_FILES:
            paths[f"calls/{call.name}/{name}"] = call / name

    # Validate the exact local bytes, and require them to match HEAD. This also
    # detects ignored/untracked inputs and works with Git's CRLF checkout rules.
    normalization_path = dataset / "normalization.json" if g9 else project / "configs/normalization.json"
    inputs = [source_manifest_path, *paths.values(), normalization_path, *schemas.glob("*.json")]
    if selection:
        inputs += [dataset / name for name in SPLIT_FILES]
    for path in inputs:
        no_links(path)
        if allow_uncommitted:
            continue
        relative = path.relative_to(repo).as_posix()
        committed = git(repo, "show", f"{commit}:{relative}")
        local = path.read_bytes()
        if local.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise ValueError(f"Source differs from committed data; commit it first (or use --allow-uncommitted): {relative}")

    checklist = load_json(paths["checklist.json"])
    if g9:
        errors = _g9_errors(dataset, schemas, paths)
    else:
        normalization = load_json(normalization_path)
        errors = validate_checklist(checklist, normalization, schemas)
    for call in calls:
        gt, transcript, roles = [load_json(call / name) for name in CALL_FILES]
        if gt.get("call_id") != call.name:
            errors.append(f"{call.name}: ground truth call_id differs from directory")
        if not g9:
            errors += [f"{call.name}: {e}" for e in validate_answers(gt, "ground_truth", checklist, normalization, schemas)]
            errors += [f"{call.name}: {e}" for e in validate_transcript_pair(gt, transcript, roles, schemas)]
        placeholder = all(s["start_time"] == s["end_time"] == "0.0" for s in transcript["audio_segments"])
        if not final_render and not placeholder:
            errors.append(f"{call.name}: v1 export supports placeholder_zero timings only")
        if final_render and placeholder:  # already guaranteed by the recompile; kept as an explicit guard against a half-written release
            errors.append(f"{call.name}: the record says audio-derived timings but the transcript still has placeholder times")
    if errors:
        raise ValueError("\n".join(errors))
    # Canonical JSON serialization makes hashes independent of OS line endings.
    files = {name: serialize(load_json(path)) for name, path in sorted(paths.items())}
    manifest = {
        "format": FORMAT,
        "source": {
            "git_commit": commit,
            "dataset_path": dataset.relative_to(repo).as_posix(),
            "dataset_id": source_manifest["dataset_id"],
            "checklist_id": source_manifest["checklist_id"],
        },
        "call_ids": discovered_call_ids,
        "timing_status": "audio_derived" if final_render else "placeholder_zero",
        "files": {name: digest(data) for name, data in files.items()},
    }
    if allow_uncommitted:  # added only when used, so committed exports stay byte-identical to before
        manifest["source"]["uncommitted"] = True
    if selection:  # the loader ignores extra manifest keys; dataset_sha256 still covers exactly the exported files
        manifest["split"] = {"label": selection["label"], "split_version": selection["split_version"],
                             "split_manifest_sha256": selection["split_manifest_sha256"], "dataset_calls": selection["dataset_calls"],
                             "calls": len(discovered_call_ids), "totals": selection["totals"], "seed": selection["seed"]}
    if g9:  # what has not happened yet (G7 dialogue verification, audio-derived timings), from the compiled ground truth
        manifest["dataset_status"] = load_json(calls[0] / "ground_truth.json")["status"]
    manifest["dataset_sha256"] = digest(serialize(manifest["files"]))
    files["manifest.json"] = serialize(manifest)
    return files


def verify_snapshot(destination):
    """Verify inventory and hashes without trusting manifest paths for writes."""
    destination = Path(destination).absolute()
    no_links(destination)
    manifest = load_json(destination / "manifest.json")
    if manifest.get("format") != FORMAT or not isinstance(manifest.get("files"), dict):
        raise ValueError("Destination is not a supported managed snapshot")
    expected = set(manifest["files"]) | {"manifest.json"}
    allowed_dirs = {"calls"}
    for name in expected:
        if name not in {"checklist.json", "manifest.json"} and not re.fullmatch(
            r"calls/CALL_[0-9]{3,}/(?:ground_truth|transcript|speaker_roles)\.json", name
        ):
            raise ValueError(f"Unexpected manifest path: {name}")
        allowed_dirs.update(p.as_posix() for p in Path(name).parents if p.as_posix() != ".")
    actual = set()
    for path in destination.rglob("*"):
        no_links(path)
        name = path.relative_to(destination).as_posix()
        if path.is_dir():
            if name not in allowed_dirs:
                raise ValueError(f"Unrelated directory in destination: {name}")
        else:
            actual.add(name)
    if actual != expected:
        raise ValueError("Destination inventory changed; refusing to overwrite unrelated/missing files")
    for name, expected_hash in manifest["files"].items():
        if digest(serialize(load_json(destination / name))) != expected_hash:
            raise ValueError(f"Snapshot modified: {name}")
    if digest(serialize(manifest["files"])) != manifest.get("dataset_sha256"):
        raise ValueError("Manifest dataset hash mismatch")
    return manifest


def export_snapshot(project, destination, *, dataset_root=DEFAULT_DATASET_ROOT, overwrite=False, allow_uncommitted=False, split=None):
    destination = Path(destination).absolute()
    no_links(destination)
    destination = destination.resolve()
    project = Path(project).resolve()
    if destination == project or project in destination.parents or destination in project.parents:
        raise ValueError("Destination must be outside the dataset project")
    files = prepare_snapshot(project, dataset_root, allow_uncommitted=allow_uncommitted, split=split)  # Complete validation before any writes.
    if destination.exists():
        if not overwrite:
            raise ValueError("Destination exists; use --overwrite for a verified snapshot update")
        verify_snapshot(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Both directories live on the same filesystem for rename/rollback.
    temporary = Path(tempfile.mkdtemp(prefix=".dataset-export-", dir=destination.parent))
    backup = temporary / "previous"
    installed = False
    try:
        stage = temporary / "snapshot"
        stage.mkdir()
        for name, data in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        verify_snapshot(stage)
        if destination.exists():
            verify_snapshot(destination)
            destination.rename(backup)
        try:
            stage.rename(destination)
            installed = True
        except OSError:
            if backup.exists():
                backup.rename(destination)
            raise
    finally:
        # If rollback itself fails, retain the backup for manual recovery.
        if installed or not backup.exists():
            import shutil
            shutil.rmtree(temporary)
        else:
            print(f"Recovery backup retained at {backup}", file=sys.stderr)
    return json.loads(files["manifest.json"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Source dataset release, relative to --project-dir unless absolute",
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify destination only; write nothing")
    parser.add_argument("--allow-uncommitted", action="store_true",
                        help="Export without requiring the source to match committed HEAD (manifest records source.uncommitted)")
    parser.add_argument("--split", choices=("DEVELOPMENT", "HELDOUT"),
                        help="Export only the calls the frozen splits/split.v1.json assigns to this label (same v1 layout)")
    args = parser.parse_args(argv)
    try:
        manifest = verify_snapshot(args.destination) if args.check else export_snapshot(
            args.project_dir,
            args.destination,
            dataset_root=args.dataset_root,
            overwrite=args.overwrite,
            allow_uncommitted=args.allow_uncommitted,
            split=args.split,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Export error: {error}", file=sys.stderr)
        return 2
    print(f"{'Verified' if args.check else 'Exported'} {len(manifest['call_ids'])} calls: {args.destination}")
    print(f"Source commit: {manifest['source']['git_commit']}" + (" (plus UNCOMMITTED changes)" if manifest["source"].get("uncommitted") else ""))
    if manifest.get("dataset_status"):
        print(f"Dataset status: {manifest['dataset_status']}")
    if manifest.get("split"):
        print(f"Split: {manifest['split']['label']} ({manifest['split']['calls']} of {manifest['split']['dataset_calls']} calls), "
              f"split manifest {manifest['split']['split_manifest_sha256']}")
    print(f"Dataset SHA256: {manifest['dataset_sha256']}")
    return 0
