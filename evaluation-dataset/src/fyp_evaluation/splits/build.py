"""Build and verify the frozen split manifest and its balance report from committed inputs only."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from fyp_evaluation.planning.outputs import sha256

from . import ALGORITHM, DEVELOPMENT, HELDOUT, HELDOUT_QUOTAS, MANIFEST_FILE, REGISTRY_FILE, REPORT_FILE, SEED, SPLIT_DIR, SplitError
from .allocate import HELDOUT_FRACTION, LONG_TARGET_PER_LANGUAGE, allocate, validate_assignment
from .features import GROUP_WEIGHT, LANGUAGES, dump, feature_definitions, fingerprints, labels_of, load_features
from .registry import USER_GIVEN, validate_registry

RARE_AT_MOST = 3  # a label on this many calls or fewer is "rare" and listed individually


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_id_of(dataset_root):
    return _read(Path(dataset_root) / "dataset_manifest.json")["dataset_id"]


def _schema(schema_dir, name):
    return _read(Path(schema_dir) / "splits" / f"{name}.schema.json")


def schema_errors(document, schema_dir, name):
    validator = Draft202012Validator(_schema(schema_dir, name))
    return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(document)]


def _quotas(features):
    totals = {lang: sum(1 for f in features.values() if f["language"] == lang) for lang in LANGUAGES}
    return {"heldout": dict(HELDOUT_QUOTAS), "development": {l: totals[l] - HELDOUT_QUOTAS[l] for l in LANGUAGES}, "language_totals": totals}


def build_documents(dataset_root, registry, *, seed=SEED, quotas=None, groups=()):
    """(manifest, report) for the release. Pure function of the committed inputs and the registry."""
    root = Path(dataset_root)
    features = load_features(root)
    quotas = quotas or dict(HELDOUT_QUOTAS)
    locked = {e["call_id"] for e in registry["locks"]}
    errors = validate_registry(registry, features)
    if errors:
        raise SplitError("; ".join(errors))
    assignment, meta = allocate(features, locked, quotas, seed=seed, groups=groups)
    model = meta.pop("model")
    problems = validate_assignment(features, assignment, locked, quotas, groups)
    if problems:
        raise SplitError("; ".join(problems))
    totals = {s: sum(1 for v in assignment.values() if v == s) for s in (DEVELOPMENT, HELDOUT)}
    registry_hash = sha256(dump(registry).encode("utf-8"))
    manifest = {
        "schema_version": "1.0", "split_version": "split.v1", "dataset_id": dataset_id_of(root),
        "method": "retrospective: drawn after generation; calls used for prompt tuning, qualification, diagnosis or targeted debugging "
                  "stay in DEVELOPMENT. Not a pre-registered hold-out.",
        "seed": seed,
        "algorithm": {**ALGORITHM, "tie_break": "sha256('<seed>:<first call id of the unit>')", "units": "whole calls (no genuine groups found)",
                      "objective": "sum over labels of weight * ((heldout_count - target) / sqrt(n * 0.3 * 0.7))^2, labels counted per distinct call",
                      "group_weights": GROUP_WEIGHT, "heldout_fraction": HELDOUT_FRACTION,
                      "priorities": ["SHORT/LONG overall and within language", "edge-case types, families and language variants",
                                     "per-question C01-C12 FAIL/PASS", "mixed-language style", "secondary transaction categories"],
                      "long_target_per_language": LONG_TARGET_PER_LANGUAGE,
                      "stages": ["development locks and exact language quotas", "greedy construction (rare labels dominate)",
                                 "best-improvement equal-size swaps inside one language"]},
        "quotas": _quotas(features), "totals": totals, "feature_definitions": feature_definitions(),
        "inputs": {**fingerprints(root), "registry_sha256": registry_hash},
        "groups": [sorted(g) for g in groups], "development_locks": sorted(locked),
        "assignments": {c: {"split": assignment[c], "language": features[c]["language"],
                            "basis": "DEVELOPMENT_LOCK" if c in locked else "ALLOCATED"} for c in sorted(features)},
    }
    return manifest, _report(features, assignment, locked, model, meta, manifest)


def _report(features, assignment, locked, model, meta, manifest):
    held = [c for c in features if assignment[c] == HELDOUT]
    rows, rare, group_dev = [], [], {}
    label_calls, label_group = {}, {}
    for call, feature in sorted(features.items()):
        for label, group in labels_of(feature).items():
            label_calls.setdefault(label, []).append(call)
            label_group[label] = group
    for label in sorted(model.total):
        calls = label_calls[label]
        n = len(calls)
        h = sum(1 for c in calls if assignment[c] == HELDOUT)
        locked_calls = sum(1 for c in calls if c in locked)
        target = model.target[label]
        group = label_group[label]
        row = {"label": label, "group": group, "calls": n, "heldout": h, "development": n - h, "target_heldout": round(target, 2),
               "deviation": round(h - target, 2), "locked_calls": locked_calls, "unlocked_calls": n - locked_calls}
        rows.append(row)
        group_dev.setdefault(group, []).append(abs(row["deviation"]))
        if n <= RARE_AT_MOST:
            rare.append({**row, "call_ids": calls, "assignments": {c: assignment[c] for c in calls},
                         "note": "a label on one call can only appear in one subset" if n == 1 else "too few calls to appear proportionally in both subsets"})
    cannot_reach = [r for r in rows if r["target_heldout"] >= 1 and r["unlocked_calls"] < round(r["target_heldout"])]
    by_language = {}
    for lang in LANGUAGES:
        calls = [c for c, f in features.items() if f["language"] == lang]
        by_language[lang] = {"development": sum(1 for c in calls if assignment[c] == DEVELOPMENT), "heldout": sum(1 for c in calls if assignment[c] == HELDOUT),
                             "long_heldout": sum(1 for c in calls if assignment[c] == HELDOUT and features[c]["duration"] == "LONG"),
                             "long_total": sum(1 for c in calls if features[c]["duration"] == "LONG"),
                             "development_locks": sum(1 for c in calls if c in locked)}
    return {
        "schema_version": "1.0", "split_version": "split.v1", "dataset_id": manifest["dataset_id"], "manifest_inputs": manifest["inputs"],
        "counts": {"calls": len(features), "development": len(features) - len(held), "heldout": len(held), "development_locks": len(locked),
                   "user_given_locks": len(USER_GIVEN), "audit_derived_locks": len(locked) - len(USER_GIVEN)},
        "by_language": by_language, "search": {"swaps": meta["swaps"], "objective": meta["objective"], "units": meta["units"]},
        "balance": rows, "max_abs_deviation_by_group": {g: max(v) for g, v in sorted(group_dev.items())},
        "rare_labels": rare, "labels_unreachable_with_locks": cannot_reach,
        "long_heldout_total": sum(v["long_heldout"] for v in by_language.values()),
        "limitations": [
            "heuristic search; no global optimality is claimed",
            "labels are not crossed (no full cross-product of dimensions); only marginal counts are balanced",
            "single-call and very small labels cannot appear in both subsets and are listed under rare_labels",
            "the split is retrospective: held-out calls were generated under the same procedure that was tuned on the locked calls, and later "
            "prompt, schema and setting revisions (G3 v3, G4 v3/v4, G7 v3, output limits) were derived from specific calls; no stronger holdout "
            "independence is claimed than that history supports",
            "the same verifier families (Codex G5, Gemini or Claude-assisted G7) reviewed development and held-out calls",
        ],
    }


def _text(value):
    return dump(value)


def write_documents(dataset_root, registry, schema_dir):
    manifest, report = build_documents(dataset_root, registry)
    for name, doc in (("split_manifest", manifest), ("split_report", report)):
        errors = schema_errors(doc, schema_dir, name)
        if errors:
            raise SplitError(f"{name} violates its schema: {errors[:3]}")
    folder = Path(dataset_root) / SPLIT_DIR
    folder.mkdir(exist_ok=True)
    (folder / MANIFEST_FILE).write_text(_text(manifest), encoding="utf-8", newline="\n")
    (folder / REPORT_FILE).write_text(_text(report), encoding="utf-8", newline="\n")
    return manifest, report


def verify(dataset_root, schema_dir):
    """Errors for the committed split files; needs only committed inputs (no ignored authoring files)."""
    root, folder = Path(dataset_root), Path(dataset_root) / SPLIT_DIR
    errors = []
    try:
        registry = _read(folder / REGISTRY_FILE)
        manifest = _read(folder / MANIFEST_FILE)
        report = _read(folder / REPORT_FILE)
    except OSError as error:
        return [f"missing split file: {error}"]
    errors += schema_errors(registry, schema_dir, "development_exposure")
    errors += schema_errors(manifest, schema_dir, "split_manifest")
    errors += schema_errors(report, schema_dir, "split_report")
    if errors:
        return errors
    features = load_features(root)
    errors += validate_registry(registry, features)
    current = {**fingerprints(root), "registry_sha256": sha256(dump(registry).encode("utf-8"))}
    stale = sorted(k for k in current if manifest["inputs"].get(k) != current[k])
    if stale:
        return errors + [f"stale fingerprints (inputs changed since the split was frozen): {stale}"]
    locked = {e["call_id"] for e in registry["locks"]}
    assignment = {c: a["split"] for c, a in manifest["assignments"].items()}
    errors += validate_assignment(features, assignment, locked, manifest["quotas"]["heldout"], manifest["groups"])
    if sorted(manifest["development_locks"]) != sorted(locked):
        errors.append("development_locks differs from the registry")
    rebuilt_manifest, rebuilt_report = build_documents(root, registry, seed=manifest["seed"], quotas=manifest["quotas"]["heldout"], groups=manifest["groups"])
    for name, text, rebuilt in ((MANIFEST_FILE, (folder / MANIFEST_FILE).read_text(encoding="utf-8"), rebuilt_manifest),
                                (REPORT_FILE, (folder / REPORT_FILE).read_text(encoding="utf-8"), rebuilt_report)):
        if text.replace("\r\n", "\n") != _text(rebuilt):
            errors.append(f"{name} is not reproduced by the deterministic algorithm")
    return errors


def manifest_hash(dataset_root):
    return sha256((Path(dataset_root) / SPLIT_DIR / MANIFEST_FILE).read_bytes())


def split_calls(manifest, label):
    """Sorted call IDs assigned to `label` by a frozen manifest."""
    if label not in (DEVELOPMENT, HELDOUT):
        raise SplitError(f"unknown split label {label!r}")
    return sorted(c for c, a in manifest["assignments"].items() if a["split"] == label)
