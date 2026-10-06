"""Artificial committed datasets exercise export and overwrite boundaries."""
import json
import subprocess
from pathlib import Path

import pytest

from fyp_evaluation import exporting
from fyp_evaluation.transcripts.converter import build_transcript_artifacts
from transcript_fixtures import minimal_ground_truth

PROJECT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("datasets/legacy_pilot_v1")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(exporting.serialize(value))


@pytest.fixture
def source(tmp_path):
    import shutil
    project = tmp_path / "source"
    shutil.copytree(PROJECT / "schemas", project / "schemas")
    shutil.copytree(PROJECT / "configs", project / "configs")
    write(project / DATASET_ROOT / "checklist.json", [
        {"id": "FIX01", "type": "COMPLIANCE", "question": "ARTIFICIAL marker?", "requirement_type": "OBLIGATION"}
    ])
    write(project / DATASET_ROOT / "dataset_manifest.json", {
        "schema_version": "0.1.0",
        "dataset_id": "artificial_pilot_v1",
        "checklist_id": "artificial_checklist_v1",
        "call_ids": ["CALL_001", "CALL_002"],
    })
    for cid in ("CALL_001", "CALL_002"):
        gt = minimal_ground_truth(cid, [("EMPLOYEE", "Artificial greeting."), ("CLIENT", "Artificial reply.")])
        transcript, roles = build_transcript_artifacts(gt, schema_dir=project / "schemas")
        for name, data in zip(exporting.CALL_FILES, (gt, transcript, roles)):
            write(project / DATASET_ROOT / "calls" / cid / name, data)
    def git(*args):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    git("init")
    git("add", ".")
    git("-c", "user.name=Export Test", "-c", "user.email=test@example.invalid", "commit", "-m", "Artificial fixture")
    return project


def test_export_deterministic_and_separate_inputs(source, tmp_path):
    destination = tmp_path / "ai/evaluation/datasets/pilot"
    manifest = exporting.export_snapshot(source, destination)
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    exporting.export_snapshot(source, destination, overwrite=True)
    after = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert before == after
    assert manifest["call_ids"] == ["CALL_001", "CALL_002"]
    assert manifest["source"]["dataset_id"] == "artificial_pilot_v1"
    assert len(before) == 8
    transcript = json.loads((destination / "calls/CALL_001/transcript.json").read_text())
    assert set(transcript) == {"transcripts", "audio_segments"}
    assert exporting.verify_snapshot(destination) == manifest


@pytest.mark.parametrize("damage", ["dirty", "missing", "mismatch"])
def test_invalid_source_writes_nothing(source, tmp_path, damage):
    path = source / DATASET_ROOT / "calls/CALL_002/transcript.json"
    if damage == "missing":
        path.unlink()
    else:
        data = json.loads(path.read_text())
        data["audio_segments"][0]["transcript"] = "Changed artificial text."
        write(path, data)
        if damage == "mismatch":
            subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "Mismatch"], check=True, capture_output=True)
    dest = tmp_path / "out"
    with pytest.raises((ValueError, OSError)):
        exporting.export_snapshot(source, dest)
    assert not dest.exists()


@pytest.mark.parametrize("damage", ["extra", "tamper", "directory"])
def test_protect_existing_destination(source, tmp_path, damage):
    dest = tmp_path / "out"
    exporting.export_snapshot(source, dest)
    if damage == "extra":
        (dest / "notes.txt").write_text("Keep me")
    elif damage == "directory":
        (dest / "unrelated").mkdir()
    else:
        (dest / "checklist.json").write_text("[]")
    before = {str(p): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        exporting.export_snapshot(source, dest, overwrite=True)
    assert before == {str(p): p.read_bytes() for p in dest.rglob("*") if p.is_file()}


def test_overwrite_must_be_explicit(source, tmp_path):
    dest = tmp_path / "out"
    exporting.export_snapshot(source, dest)
    with pytest.raises(ValueError, match="--overwrite"):
        exporting.export_snapshot(source, dest)


def test_source_manifest_must_match_calls(source, tmp_path):
    manifest_path = source / DATASET_ROOT / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["call_ids"] = ["CALL_001"]
    write(manifest_path, manifest)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "commit", "-m", "Bad manifest",
    ], check=True, capture_output=True)
    with pytest.raises(ValueError, match="call_ids"):
        exporting.export_snapshot(source, tmp_path / "out")


def test_install_failure_restores_existing_snapshot(source, tmp_path, monkeypatch):
    dest = tmp_path / "out"
    original = exporting.export_snapshot(source, dest)
    rename = Path.rename
    def fail_stage(path, target):
        if path.name == "snapshot":
            raise OSError("Artificial install failure")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_stage)
    with pytest.raises(OSError):
        exporting.export_snapshot(source, dest, overwrite=True)
    assert exporting.verify_snapshot(dest) == original


def test_cli_check_is_read_only(source, tmp_path):
    dest = tmp_path / "out"
    assert exporting.main(["--project-dir", str(source), "--destination", str(dest)]) == 0
    before = (dest / "manifest.json").read_bytes()
    assert exporting.main(["--destination", str(dest), "--check"]) == 0
    assert (dest / "manifest.json").read_bytes() == before


def test_windows_line_endings_verify(source, tmp_path):
    dest = tmp_path / "out"
    exporting.export_snapshot(source, dest)
    for path in dest.rglob("*.json"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    exporting.verify_snapshot(dest)


def test_relative_destination_outside_project(source, tmp_path, monkeypatch):
    monkeypatch.chdir(source)
    exporting.export_snapshot(source, Path('../ai/evaluation/datasets/pilot'))
    exporting.verify_snapshot(tmp_path / 'ai/evaluation/datasets/pilot')


def test_failed_rollback_preserves_backup(source, tmp_path, monkeypatch):
    dest = tmp_path / "out"
    exporting.export_snapshot(source, dest)
    rename = Path.rename
    def fail_install_and_restore(path, target):
        if path.name in {"snapshot", "previous"}:
            raise OSError("Artificial rename failure")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_install_and_restore)
    with pytest.raises(OSError):
        exporting.export_snapshot(source, dest, overwrite=True)
    backups = list(tmp_path.glob(".dataset-export-*/previous"))
    assert len(backups) == 1
    exporting.verify_snapshot(backups[0])
