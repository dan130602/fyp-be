"""Identity checks that run before any inference: the model directory against its manifest, the vendored sources against SOURCES.json, the installed MERaLiON package against its manifest.

Nothing here imports torch, meralion_3_asr or vllm (the installed package is located and hashed, never imported). A manifest whose status is not "complete" is refused: unknown
information stays unknown and cannot pass a check.
"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from meralion_worker import vendored

CHUNK = 8 * 1024 * 1024
PACKAGE_MANIFEST_ID = "meralion-3-asr-0.1.0"


class CheckError(Exception):
    """An identity check failed (exit code 3)."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def canonical_digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CheckError(f"file not found: {path}")
    except json.JSONDecodeError as error:
        raise CheckError(f"not valid JSON: {path}: {error}")


def load_model_manifest(manifest_id, config_root):
    return load_json(Path(config_root) / "models" / f"{manifest_id}.json")


def check_model(model_dir, manifest):
    """Hash every manifest file in model_dir. Returns {"files_checked", "seconds", "manifest_sha256"}; raises CheckError otherwise."""
    started = time.monotonic()
    if manifest.get("status") != "complete":
        raise CheckError(f"model manifest {manifest.get('id')!r} is {manifest.get('status')!r}, not 'complete': {manifest.get('status_note', 'identity not established')}")
    if manifest.get("kind") != "complete_checkpoint":
        raise CheckError(f"model manifest kind must be 'complete_checkpoint', not {manifest.get('kind')!r}")
    directory = Path(model_dir)
    if not directory.is_dir():
        raise CheckError(f"model directory not found: {directory}")
    if (directory / "adapter_config.json").exists():
        raise CheckError("adapter_config.json found: v0 supports complete checkpoints only, never an adapter")
    files = manifest.get("files") or {}
    if not files:
        raise CheckError("model manifest lists no files")
    unlisted = sorted(p.name for p in directory.glob("*.safetensors") if p.name not in files)
    if unlisted:
        raise CheckError(f"safetensors files not in the manifest: {unlisted}")
    for name, expected in sorted(files.items()):
        path = directory / name
        if not path.is_file():
            raise CheckError(f"model file missing: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise CheckError(f"model file size differs: {name} ({path.stat().st_size} != {expected['bytes']})")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise CheckError(f"model file hash differs: {name}")
    return {"files_checked": len(files), "seconds": round(time.monotonic() - started, 3), "manifest_sha256": canonical_digest(manifest)}


def verify_vendored(directory=None):
    """Every file in SOURCES.json must exist and match its recorded git blob id, sha256 and size."""
    directory = Path(directory) if directory else vendored.DIRECTORY
    sources = load_json(directory / "SOURCES.json")
    checked = {}
    for name, record in sorted(sources["files"].items()):
        path = directory / name
        if not path.is_file():
            raise CheckError(f"vendored file missing: {name}")
        data = path.read_bytes()
        if blob_id(data) != record["git_blob"]:
            raise CheckError(f"vendored file differs from its git blob: {name}")
        if hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["bytes"]:
            raise CheckError(f"vendored file differs from its recorded sha256 or size: {name}")
        checked[name] = record["git_blob"]
    return {"commits": {name: sources["files"][name]["commit"] for name in checked}, "files": checked, "sources_sha256": canonical_digest(sources)}


def installed_package_dir():
    """Locate the installed meralion_3_asr package without importing it."""
    spec = importlib.util.find_spec("meralion_3_asr")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def verify_package(manifest, package_dir=None):
    """Every file of the installed package must match the manifest, with no extra .py files. A pending manifest is refused."""
    if manifest.get("status") != "complete":
        raise CheckError(f"package manifest {manifest.get('id')!r} is {manifest.get('status')!r}, not 'complete': {manifest.get('status_note', 'provenance not established')}")
    if manifest.get("provenance_verdict") not in ("identical_to_pypi", "differs_from_pypi", "not_on_pypi"):
        raise CheckError(f"package provenance verdict is {manifest.get('provenance_verdict')!r}")
    directory = Path(package_dir) if package_dir else installed_package_dir()
    if directory is None or not directory.is_dir():
        raise CheckError("meralion_3_asr is not installed")
    expected = manifest.get("files") or {}
    if not expected:
        raise CheckError("package manifest lists no files")
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    extra = sorted(name for name in present - set(expected) if name.endswith(".py"))
    if extra:
        raise CheckError(f"installed package has files not in the manifest: {extra}")
    for name, digest in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            raise CheckError(f"installed package file missing: {name}")
        if sha256_file(path) != digest:
            raise CheckError(f"installed package file differs from the manifest: {name}")
    return {"files_checked": len(expected), "provenance_verdict": manifest["provenance_verdict"], "manifest_sha256": canonical_digest(manifest)}
