"""The frozen 175/75 split: hard constraints, determinism, locks, grouping, staleness and rare-label reporting. Offline."""
import json
import random
import shutil
from pathlib import Path

import pytest

from fyp_evaluation.splits import DEVELOPMENT, HELDOUT, HELDOUT_QUOTAS, SEED, SplitError
from fyp_evaluation.splits.allocate import allocate, validate_assignment
from fyp_evaluation.splits.build import build_documents, split_calls, verify
from fyp_evaluation.splits.features import load_features
from fyp_evaluation.splits.registry import USER_GIVEN, validate_registry

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "datasets" / "advisory_execution_multilingual_v1"
SCHEMAS = ROOT / "schemas"
SPLITS = RELEASE / "splits"


def _json(name):
    return json.loads((SPLITS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def features():
    return load_features(RELEASE)


@pytest.fixture(scope="module")
def manifest():
    return _json("split.v1.json")


@pytest.fixture(scope="module")
def registry():
    return _json("development_exposure.v1.json")


def test_committed_split_verifies_end_to_end():
    assert verify(RELEASE, SCHEMAS) == []


def test_exact_coverage_counts_and_language_quotas(features, manifest):
    assert sorted(manifest["assignments"]) == sorted(features) and len(features) == 250
    assert len(split_calls(manifest, DEVELOPMENT)) == 175 and len(split_calls(manifest, HELDOUT)) == 75
    assert set(split_calls(manifest, DEVELOPMENT)).isdisjoint(split_calls(manifest, HELDOUT))
    for language, held in HELDOUT_QUOTAS.items():
        assert sum(1 for c, a in manifest["assignments"].items() if a["language"] == language and a["split"] == HELDOUT) == held
    assert manifest["quotas"]["development"] == {"ENGLISH": 52, "MANDARIN": 53, "EN_ZH_MIXED": 35, "SINGLISH": 35}


def test_every_development_lock_stays_in_development(manifest, registry):
    locked = {e["call_id"] for e in registry["locks"]}
    assert len(locked) == 41 and set(USER_GIVEN) <= locked and len(USER_GIVEN) == 33
    assert all(manifest["assignments"][c]["split"] == DEVELOPMENT for c in locked)
    assert sorted(manifest["development_locks"]) == sorted(locked)


def test_long_calls_are_balanced_two_per_language(features, manifest):
    for language in HELDOUT_QUOTAS:
        long_held = [c for c, a in manifest["assignments"].items() if a["language"] == language and a["split"] == HELDOUT and features[c]["duration"] == "LONG"]
        assert len(long_held) == 2


def test_labels_come_only_from_frozen_specs(features):
    sample = next(iter(features.values()))
    assert set(sample) == {"call_id", "language", "duration", "mixing_style", "edge_case_ids", "families", "variants", "failing", "secondary", "assignments"}
    assert all(q.startswith("C") for f in features.values() for q in f["failing"])  # only C01-C12 can FAIL


def test_deterministic_for_shuffled_input_order(features, registry):
    locked = {e["call_id"] for e in registry["locks"]}
    first, _ = allocate(features, locked, HELDOUT_QUOTAS, seed=SEED)
    items = list(features.items())
    random.Random(7).shuffle(items)
    second, _ = allocate(dict(items), locked, HELDOUT_QUOTAS, seed=SEED)
    assert first == second
    other, _ = allocate(features, locked, HELDOUT_QUOTAS, seed=SEED + 1)
    assert validate_assignment(features, other, locked, HELDOUT_QUOTAS) == []  # a different seed is still feasible, only the ties move
    a, b = build_documents(RELEASE, registry), build_documents(RELEASE, registry)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_infeasible_constraints_fail_loudly(features):
    english = [c for c, f in features.items() if f["language"] == "ENGLISH"]
    with pytest.raises(SplitError, match="unlocked calls remain"):
        allocate(features, set(english[:60]), HELDOUT_QUOTAS, seed=SEED)
    with pytest.raises(SplitError, match="not in the release"):
        allocate(features, {"CALL_999"}, HELDOUT_QUOTAS, seed=SEED)


def test_removing_a_user_given_lock_is_refused(registry):
    broken = json.loads(json.dumps(registry))
    broken["locks"] = [e for e in broken["locks"] if e["call_id"] != "CALL_002"]
    assert any("CALL_002" in e and "removed" in e for e in validate_registry(broken))
    demoted = json.loads(json.dumps(registry))
    next(e for e in demoted["locks"] if e["call_id"] == "CALL_003")["source"] = "audit_derived"
    assert validate_registry(demoted)


def _tiny(n_per_language=6):
    features = {}
    for language in ("ENGLISH", "MANDARIN"):
        for i in range(n_per_language):
            call = f"CALL_{len(features) + 1:03d}"
            features[call] = {"call_id": call, "language": language, "duration": "LONG" if i == 0 else "SHORT", "mixing_style": "NONE",
                              "edge_case_ids": [f"EC_{i % 3}"], "families": ["F"], "variants": [], "failing": ["C03"] if i % 2 else [],
                              "secondary": {"prod": "P", "cur": "SGD", "dir": "BUY", "ord": "LIMIT"}, "assignments": 1}
    return features


def test_a_genuine_group_stays_together_and_bad_groups_are_refused():
    features = _tiny()
    quotas = {"ENGLISH": 2, "MANDARIN": 2}
    group = ("CALL_002", "CALL_003")
    assignment, _ = allocate(features, set(), quotas, seed=SEED, groups=[group])
    assert assignment["CALL_002"] == assignment["CALL_003"]
    assert validate_assignment(features, assignment, set(), quotas, [group]) == []
    with pytest.raises(SplitError, match="more than one language"):
        allocate(features, set(), quotas, seed=SEED, groups=[("CALL_001", "CALL_007")])
    with pytest.raises(SplitError, match="partly locked"):
        allocate(features, {"CALL_002"}, quotas, seed=SEED, groups=[group])
    bad = {c: DEVELOPMENT for c in features}
    bad["CALL_002"] = HELDOUT
    assert any("split across" in e for e in validate_assignment(features, bad, set(), {"ENGLISH": 1, "MANDARIN": 0}, [group]))


def test_stale_fingerprints_are_detected(tmp_path):
    copy = tmp_path / "release"
    for name in ("authoring/call_specs", "splits"):
        shutil.copytree(RELEASE / name, copy / name)
    for name in ("authoring/call_manifest.csv", "authoring/edge_case_assignments.csv", "authoring/planning_record.json",
                 "authoring/call_spec_record.json", "dataset_manifest.json"):
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RELEASE / name, copy / name)
    assert verify(copy, SCHEMAS) == []
    spec = copy / "authoring/call_specs/CALL_001.json"
    spec.write_text(spec.read_text(encoding="utf-8").replace('"SHORT"', '"SHORT" ', 1), encoding="utf-8")
    assert any("stale fingerprints" in e for e in verify(copy, SCHEMAS))


def test_tampered_assignment_is_refused(tmp_path):
    copy = tmp_path / "release"
    for name in ("authoring/call_specs", "splits"):
        shutil.copytree(RELEASE / name, copy / name)
    for name in ("authoring/call_manifest.csv", "authoring/edge_case_assignments.csv", "authoring/planning_record.json",
                 "authoring/call_spec_record.json", "dataset_manifest.json"):
        shutil.copy2(RELEASE / name, copy / name)
    path = copy / "splits/split.v1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    ca = next(c for c, a in document["assignments"].items() if a["split"] == HELDOUT and a["language"] == "ENGLISH")
    cb = next(c for c, a in document["assignments"].items() if a["split"] == DEVELOPMENT and a["language"] == "ENGLISH" and a["basis"] == "ALLOCATED")
    document["assignments"][ca]["split"], document["assignments"][cb]["split"] = DEVELOPMENT, HELDOUT  # a valid-looking swap
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    assert any("not reproduced" in e for e in verify(copy, SCHEMAS))


def test_rare_labels_and_limitations_are_reported():
    report = _json("split_report.v1.json")
    rare = report["rare_labels"]
    assert rare and all(len(r["call_ids"]) <= 3 and r["note"] for r in rare)
    assert any(r["calls"] == 1 and "only appear in one subset" in r["note"] for r in rare)
    assert any("no global optimality" in s for s in report["limitations"]) and any("retrospective" in s for s in report["limitations"])
    assert report["counts"] == {"calls": 250, "development": 175, "heldout": 75, "development_locks": 41, "user_given_locks": 33, "audit_derived_locks": 8}
