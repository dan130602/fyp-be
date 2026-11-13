"""Which audio a collect run sends for each call: the original clean recording, or the environment_v2 variant assigned to the call.

`clean` (default) is the behaviour the engine always had. `environment_v2` reads the frozen assignment manifest
(`audio_augmentations/environment_v2/assignments.json`, built by evaluation-dataset) and resolves every call explicitly:

  * condition C1 -> the call's own clean render (the G9-pinned file named by the manifest), never a copy;
  * any other condition -> `calls/<call>/audio_variants/environment_v2/audio.wav`.

There is no folder scanning and no fallback: a missing variant, a variant that disagrees with its `augmentation.json`, a manifest that
does not list the call, or a config that is not FROZEN stops the run before anything is uploaded.
"""

import hashlib
import json
from pathlib import Path

CLEAN = "clean"
ENVIRONMENT_V2 = "environment_v2"
AUDIO_CONDITIONS = (CLEAN, ENVIRONMENT_V2)
CLEAN_CONDITION = "C1"
V2_FOLDER = Path("audio_augmentations") / "environment_v2"
_CACHE = {}


class AudioConditionError(RuntimeError):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest(dataset_root):
    """(assignments document, sha256 of the file, sha256 of config.json) after checking that the config is FROZEN."""
    folder = Path(dataset_root) / V2_FOLDER
    path = folder / "assignments.json"
    if not path.is_file():
        raise AudioConditionError(f"no environment_v2 assignments at {path}")
    config_path = folder / "config.json"
    if not config_path.is_file():
        raise AudioConditionError(f"no environment_v2 config at {config_path}")
    key = (str(path), path.stat().st_mtime_ns, config_path.stat().st_mtime_ns)    # either file changing invalidates the check
    if key not in _CACHE:
        raw = path.read_bytes()
        document = json.loads(raw)
        config = _read(config_path)
        if config.get("status") != "FROZEN":
            raise AudioConditionError("environment_v2 config is not FROZEN; freeze it before evaluating")
        pinned = (document.get("subset") or {}).get("source_assignments_sha256") or hashlib.sha256(raw).hexdigest()
        if config.get("frozen_assignments_sha256") != pinned:
            raise AudioConditionError("assignments.json does not match the hash pinned when the config was frozen")
        _CACHE[key] = (document, hashlib.sha256(raw).hexdigest(), hashlib.sha256(config_path.read_bytes()).hexdigest())
    return _CACHE[key]


def row_of(dataset_root, call_id):
    document = manifest(dataset_root)[0]
    row = document["calls"].get(call_id)
    if row is None:
        raise AudioConditionError(f"{call_id}: not listed in the environment_v2 assignments")
    return row


def condition_of(dataset_root, call_id):
    return row_of(dataset_root, call_id)["condition"]


def select_calls(dataset_root, call_ids, skip_clean_controls):
    """(kept call ids, skipped clean-control ids). Without the flag nothing is skipped."""
    if not skip_clean_controls:
        return list(call_ids), []
    skipped = [c for c in call_ids if condition_of(dataset_root, c) == CLEAN_CONDITION]
    return [c for c in call_ids if c not in skipped], skipped


def resolve(dataset_root, call_id):
    """{path, condition, variant, expected_sha256, seconds}: the exact file to send for this call, or AudioConditionError."""
    root = Path(dataset_root)
    row = row_of(root, call_id)
    condition = row["condition"]
    if condition == CLEAN_CONDITION:
        path = root / row["source"]["path"]
        if not path.is_file():
            raise AudioConditionError(f"{call_id}: clean source audio missing at {path}")
        return {"path": path, "condition": condition, "variant": False, "expected_sha256": None, "seconds": row["duration_seconds"]}
    path = root / row["variant_path"]
    meta_path = path.with_name("augmentation.json")
    if not path.is_file() or not meta_path.is_file():
        raise AudioConditionError(f"{call_id}: {condition} variant missing at {path.parent}; build it, the engine never falls back to clean")
    meta = _read(meta_path)
    if meta["assignment"]["condition"] != condition:
        raise AudioConditionError(f"{call_id}: assigned {condition} but the variant on disk was built as {meta['assignment']['condition']}")
    return {"path": path, "condition": condition, "variant": True, "expected_sha256": meta["output"]["sha256"],
            "seconds": meta["output"]["duration_seconds"]}
