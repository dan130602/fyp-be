"""--audio-condition clean|environment_v2: explicit per-call audio, clean calls still point at the original, no silent fallback."""
import hashlib
import json

import pytest

from transcribe_eval import audio_condition as ac
from transcribe_eval import collect as engine

from test_collect import BUCKET, dataset, world  # noqa: F401  (fixtures)

VARIANT = "audio_variants/environment_v2"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data) if isinstance(data, bytes) else path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def v2(dataset):  # noqa: F811
    """toy_v1 with CALL_001 clean (C1), CALL_002 office noise (N1), CALL_003 telephone (D1), and a FROZEN environment_v2 manifest."""
    third = dataset / "calls" / "CALL_003" / "renders" / "clean_voice_A"
    _write(third / "audio.mp3", b"audio-3")
    _write(third / "timings.json", {"duration_seconds": 60})
    calls = {}
    for number, condition in ((1, "C1"), (2, "N1"), (3, "D1")):
        call = f"CALL_00{number}"
        row = {"condition": condition, "severity": None, "duration_seconds": 60.0, "split": "DEVELOPMENT", "language": "ENGLISH",
               "source": {"render_id": "clean_voice_A", "path": f"calls/{call}/renders/clean_voice_A/audio.mp3"}, "variant_path": None}
        if condition != "C1":
            data = b"variant-%d" % number
            folder = dataset / "calls" / call / VARIANT
            _write(folder / "audio.wav", data)
            _write(folder / "augmentation.json", {"assignment": {"condition": condition, "severity": None},
                                                  "output": {"sha256": hashlib.sha256(data).hexdigest(), "duration_seconds": 61.5}})
            row["variant_path"] = f"calls/{call}/{VARIANT}/audio.wav"
        calls[call] = row
    folder = dataset / "audio_augmentations" / "environment_v2"
    _write(folder / "assignments.json", {"calls": calls})
    digest = hashlib.sha256((folder / "assignments.json").read_bytes()).hexdigest()
    _write(folder / "config.json", {"status": "FROZEN", "frozen_assignments_sha256": digest})
    ac._CACHE.clear()
    return dataset


def plan_v2(root, calls=("CALL_001", "CALL_002", "CALL_003"), render="auto", **kwargs):
    return engine.plan(root, render, list(calls), BUCKET, "environment_v2", **kwargs)


def run_v2(world, root, tmp_path, calls=("CALL_001", "CALL_002", "CALL_003"), **kwargs):  # noqa: F811
    return engine.collect(world.settings, world.handler, world.s3, dataset_root=root, render_id="auto", call_ids=list(calls),
                          runs_root=tmp_path / "runs", sleep=lambda s: None, progress=lambda m: None, audio_condition="environment_v2", **kwargs)


def test_clean_calls_point_at_their_original_and_the_rest_at_the_variant(v2):
    rows = {r["call_id"]: r for r in plan_v2(v2)["calls"]}
    assert rows["CALL_001"]["master_key"] == "eval-audio/toy_v1/clean_voice_A/CALL_001.mp3" and rows["CALL_001"]["v2_condition"] == "C1"
    assert rows["CALL_002"]["master_key"] == "eval-audio/toy_v1/environment_v2/CALL_002.wav" and rows["CALL_002"]["v2_condition"] == "N1"
    assert rows["CALL_002"]["audio_seconds"] == 61.5 and rows["CALL_001"]["audio_seconds"] == 60.0


def test_clean_mode_is_unchanged_and_ignores_the_v2_folder(v2):
    planned = engine.plan(v2, "clean_voice_A", ["CALL_001", "CALL_002"], BUCKET)
    assert [r["master_key"] for r in planned["calls"]] == ["eval-audio/toy_v1/clean_voice_A/CALL_00%d.mp3" % n for n in (1, 2)]
    assert planned["audio_condition"] == "clean" and planned["skipped_clean_controls"] == []


def test_skip_clean_controls_drops_exactly_the_clean_calls_and_says_so(v2):
    planned = plan_v2(v2, skip_clean_controls=True)
    assert [r["call_id"] for r in planned["calls"]] == ["CALL_002", "CALL_003"] and planned["skipped_clean_controls"] == ["CALL_001"]
    with pytest.raises(engine.CollectError, match="only applies"):
        engine.plan(v2, "clean_voice_A", ["CALL_001"], BUCKET, "clean", skip_clean_controls=True)


def test_a_missing_variant_stops_before_any_upload_and_never_falls_back_to_clean(v2, world, tmp_path):  # noqa: F811
    world.build()
    (v2 / "calls" / "CALL_002" / VARIANT / "audio.wav").unlink()
    with pytest.raises(engine.CollectError, match="never falls back"):
        run_v2(world, v2, tmp_path)
    assert not world.s3.objects and not (tmp_path / "runs").exists()


def test_a_variant_changed_since_it_was_built_is_refused(v2):
    (v2 / "calls" / "CALL_002" / VARIANT / "audio.wav").write_bytes(b"tampered")
    with pytest.raises(engine.CollectError, match="sha256 mismatch"):
        plan_v2(v2)


def test_a_variant_built_as_another_condition_is_refused(v2):
    path = v2 / "calls" / "CALL_002" / VARIANT / "augmentation.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["assignment"]["condition"] = "D2"
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(engine.CollectError, match="assigned N1"):
        plan_v2(v2)


def test_an_unfrozen_config_a_changed_manifest_or_an_unlisted_call_is_refused(v2):
    folder = v2 / "audio_augmentations" / "environment_v2"
    with pytest.raises(engine.CollectError, match="not listed"):
        plan_v2(v2, calls=("CALL_009",))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    _write(folder / "config.json", dict(config, frozen_assignments_sha256="0" * 64))
    with pytest.raises(engine.CollectError, match="pinned"):
        plan_v2(v2)
    _write(folder / "config.json", dict(config, status="DRAFT"))
    with pytest.raises(engine.CollectError, match="not FROZEN"):
        plan_v2(v2)


def test_environment_v2_needs_render_id_auto(v2):
    with pytest.raises(engine.CollectError, match="--render-id auto"):
        plan_v2(v2, render="clean_voice_A")


def test_a_v2_run_uploads_the_right_files_and_records_what_it_did(v2, world, tmp_path):  # noqa: F811
    world.build()
    metadata = run_v2(world, v2, tmp_path, run_id="v2a", skip_clean_controls=True)
    assert metadata["audio_condition"] == "environment_v2" and metadata["skip_clean_controls"] is True
    assert metadata["skipped_clean_controls"] == ["CALL_001"] and len(metadata["assignments_sha256"]) == 64
    entries = {e["call_id"]: e for e in metadata["calls"]}
    assert sorted(entries) == ["CALL_002", "CALL_003"] and entries["CALL_002"]["v2_condition"] == "N1"
    assert entries["CALL_002"]["run_key"].endswith("/CALL_002.wav") and entries["CALL_002"]["audio_condition"] == "environment_v2"
    assert "eval-audio/toy_v1/environment_v2/CALL_002.wav" in world.s3.objects
    assert "eval-audio/toy_v1/clean_voice_A/CALL_001.mp3" not in world.s3.objects     # the skipped clean control was not even uploaded


def test_without_the_skip_flag_the_clean_call_is_sent_from_its_original_recording(v2, world, tmp_path):  # noqa: F811
    world.build()
    metadata = run_v2(world, v2, tmp_path, run_id="v2b")
    entries = {e["call_id"]: e for e in metadata["calls"]}
    assert entries["CALL_001"]["master_key"] == "eval-audio/toy_v1/clean_voice_A/CALL_001.mp3" and entries["CALL_001"]["v2_condition"] == "C1"
    assert entries["CALL_001"]["run_key"].endswith("/CALL_001.mp3")
    assert metadata["skipped_clean_controls"] == [] and len(metadata["calls"]) == 3


def test_a_shared_subset_manifest_resolves_when_it_names_the_full_manifest_the_config_pinned(v2):
    folder = v2 / "audio_augmentations" / "environment_v2"
    document = json.loads((folder / "assignments.json").read_text(encoding="utf-8"))
    document["subset"] = {"split": "DEVELOPMENT", "source_assignments_sha256": "f" * 64}
    _write(folder / "assignments.json", document)
    _write(folder / "config.json", {"status": "FROZEN", "frozen_assignments_sha256": "f" * 64})
    assert [r["v2_condition"] for r in plan_v2(v2)["calls"]] == ["C1", "N1", "D1"]
    _write(folder / "config.json", {"status": "FROZEN", "frozen_assignments_sha256": "e" * 64})
    with pytest.raises(engine.CollectError, match="pinned"):
        plan_v2(v2)
