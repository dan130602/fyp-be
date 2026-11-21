"""The runtime a decoder runs on, as part of the invocation identity (Stage 3 follow-up).

A state started with one runtime must not be resumed with another: package versions and any tracked compatibility patch change what the decoder is. `runtime_identity()` records
  * the installed versions of the packages that decide decoding (None when a package is not installed), and
  * the tracked patches: for each entry of `patches/manifest.json` the target file, the manifest hashes and the sha256 of the file that is INSTALLED, with a state of
    "original" (equals the manifest's original hash), "patched" (equals the manifest's patched hash), "unknown" (anything else: not a state the manifest describes) or "not_installed" (the target package
    is not in this environment, e.g. the Transformers environment: the patch does not apply there).
With no manifest there are no patches and the list is empty. Importing this module loads none of the packages.
"""
import hashlib
import importlib.util
import json
from importlib import metadata
from pathlib import Path

PACKAGES = ("vllm", "torch", "transformers", "meralion-3-asr")
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "patches" / "manifest.json"


def package_versions(packages=PACKAGES):
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _installed_file(target):
    """The installed file for a manifest target such as 'vllm/platforms/cuda.py' (found without importing the package), or None."""
    top, _, rest = target.partition("/")
    spec = importlib.util.find_spec(top)
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / rest
        if candidate.is_file():
            return candidate
    return None


def patches_identity(manifest_path=None):
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.is_file():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for entry in manifest.get("patches", []):
        installed = _installed_file(entry["target"])
        digest = hashlib.sha256(installed.read_bytes()).hexdigest() if installed else None
        state = "not_installed" if installed is None else "original" if digest == entry["original_sha256"] else "patched" if digest == entry["patched_sha256"] else "unknown"
        found.append({"target": entry["target"], "target_version": entry.get("target_version"), "original_sha256": entry["original_sha256"], "patched_sha256": entry["patched_sha256"],
                      "patch_sha256": entry["patch_sha256"], "installed_sha256": digest, "state": state})
    return found


def runtime_identity(*, packages=PACKAGES, manifest_path=None):
    return {"packages": package_versions(packages), "patches": patches_identity(manifest_path)}
