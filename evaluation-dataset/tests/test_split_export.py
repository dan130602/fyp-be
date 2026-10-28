"""Split-aware export: v1 layout kept, calls selected only from the frozen split manifest, stale/altered splits refused. Offline."""
import json
import shutil
from pathlib import Path

import pytest

from fyp_evaluation import exporting
from fyp_evaluation.splits import DEVELOPMENT, HELDOUT
from fyp_evaluation.splits.build import manifest_hash
from test_dataset_export import source  # noqa: F401  (artificial committed two-call release fixture)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "datasets" / "advisory_execution_multilingual_v1"
SCHEMAS = ROOT / "schemas"
FAKE = {"label": HELDOUT, "call_ids": ["CALL_002"], "split_version": "split.v1", "split_manifest_sha256": "0" * 64, "dataset_calls": 2,
        "totals": {"DEVELOPMENT": 1, "HELDOUT": 1}, "seed": 1}


def test_unsplit_export_is_unchanged_and_carries_no_split_key(source, tmp_path):  # noqa: F811
    manifest = exporting.export_snapshot(source, tmp_path / "all")
    assert "split" not in manifest and manifest["call_ids"] == ["CALL_001", "CALL_002"]


def test_split_export_keeps_the_v1_layout_for_the_selected_calls_only(source, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(exporting, "split_selection", lambda dataset, schemas, label: FAKE)
    destination = tmp_path / "heldout"
    manifest = exporting.export_snapshot(source, destination, split=HELDOUT, allow_uncommitted=True)  # the fixture has no committed split files
    assert manifest["source"]["uncommitted"] is True
    assert manifest["format"] == "fyp-nlp-dataset-v1" and manifest["call_ids"] == ["CALL_002"]
    assert sorted(manifest["files"]) == ["calls/CALL_002/ground_truth.json", "calls/CALL_002/speaker_roles.json",
                                         "calls/CALL_002/transcript.json", "checklist.json"]
    assert manifest["split"]["label"] == HELDOUT and manifest["split"]["calls"] == 1 and manifest["split"]["dataset_calls"] == 2
    assert not (destination / "calls/CALL_001").exists()
    assert exporting.verify_snapshot(destination)["dataset_sha256"] == manifest["dataset_sha256"]


def test_a_split_export_requires_the_split_files_to_be_committed_unless_allowed(source, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(exporting, "split_selection", lambda dataset, schemas, label: FAKE)
    with pytest.raises(ValueError, match="splits/split.v1.json"):
        exporting.export_snapshot(source, tmp_path / "strict", split=HELDOUT)
    assert not (tmp_path / "strict").exists()


def test_a_selection_that_does_not_cover_the_release_or_names_missing_calls_is_refused(source, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(exporting, "split_selection", lambda d, s, label: {**FAKE, "dataset_calls": 3})
    with pytest.raises(ValueError, match="do not cover exactly"):
        exporting.export_snapshot(source, tmp_path / "a", split=HELDOUT)
    monkeypatch.setattr(exporting, "split_selection", lambda d, s, label: {**FAKE, "call_ids": ["CALL_002", "CALL_009"]})
    with pytest.raises(ValueError, match="not compiled"):
        exporting.export_snapshot(source, tmp_path / "b", split=HELDOUT)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_real_frozen_split_selects_disjoint_covering_subsets():
    development = exporting.split_selection(RELEASE, SCHEMAS, DEVELOPMENT)
    heldout = exporting.split_selection(RELEASE, SCHEMAS, HELDOUT)
    assert len(development["call_ids"]) == 175 and len(heldout["call_ids"]) == 75
    assert set(development["call_ids"]).isdisjoint(heldout["call_ids"])
    assert len(set(development["call_ids"]) | set(heldout["call_ids"])) == 250 == development["dataset_calls"]
    assert development["split_manifest_sha256"] == heldout["split_manifest_sha256"] == manifest_hash(RELEASE)


def test_an_altered_or_stale_split_is_refused_by_the_exporter(tmp_path):
    copy = tmp_path / "release"
    for name in ("authoring/call_specs", "splits"):
        shutil.copytree(RELEASE / name, copy / name)
    for name in ("authoring/call_manifest.csv", "authoring/edge_case_assignments.csv", "authoring/planning_record.json",
                 "authoring/call_spec_record.json", "dataset_manifest.json"):
        shutil.copy2(RELEASE / name, copy / name)
    assert exporting.split_selection(copy, SCHEMAS, HELDOUT)["call_ids"]
    path = copy / "splits/split.v1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["assignments"].pop("CALL_250")
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing, altered or stale"):
        exporting.split_selection(copy, SCHEMAS, HELDOUT)
    shutil.copy2(RELEASE / "splits/split.v1.json", path)
    spec = copy / "authoring/call_specs/CALL_010.json"
    spec.write_text(spec.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="missing, altered or stale"):
        exporting.split_selection(copy, SCHEMAS, HELDOUT)
