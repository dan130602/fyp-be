"""Assembling one complete environment_v2 run from a skip-clean V2 run and the earlier clean run (offline, fake run folders)."""
import hashlib
import json

import pytest

from transcribe_eval import assemble as asm
from transcribe_eval import compare as cmp

from test_compare import CONDS, JOB, LANGS, row  # noqa: F401

ENV = {"TRANSCRIBE_LANGUAGE_OPTIONS": "en-US,zh-CN"}


def collected(root, run_id, calls, condition, skipped=(), handler="aaa", started="2026-11-06T00:00:00+00:00", env=None, job=None, status="COMPLETED"):
    folder = root / run_id
    (folder / "raw" / "transcripts").mkdir(parents=True)
    entries = []
    for call in calls:
        (folder / "raw" / "transcripts" / f"{call}.json").write_text(json.dumps({"results": {"transcripts": [{"transcript": f"{run_id} {call}"}]}}), encoding="utf-8")
        entries.append({"call_id": call, "status": status, "transcript_file": f"raw/transcripts/{call}.json", "audio_condition": condition,
                        "v2_condition": CONDS[call] if condition == "environment_v2" else None, "job": job or dict(JOB, CreationTime=f"{started}")})
    metadata = {"run_id": run_id, "started_at": started, "finished_at": started, "status": "completed", "dataset_id": "toy", "audio_condition": condition,
                "skip_clean_controls": bool(skipped), "skipped_clean_controls": list(skipped), "handler": {"sha256": handler},
                "handler_environment": env or ENV, "calls": entries}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    collected(root, "clean", ["CALL_001", "CALL_002", "CALL_003", "CALL_004"], "clean")
    collected(root, "v2", ["CALL_002", "CALL_003", "CALL_004"], "environment_v2", skipped=["CALL_001"], started="2026-11-13T09:00:00+00:00")
    return root


def test_the_assembled_run_has_every_call_keeps_the_real_date_and_tags_the_reused_transcripts(runs):
    before = {p: digest(p) for p in (runs / "clean").rglob("*") if p.is_file()}
    result = asm.assemble(runs, "v2", "clean", "full")
    assert result["calls"] == 4 and result["new_calls"] == 3 and result["reused_clean_calls"] == 1
    metadata = json.loads((runs / "full" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "full" and metadata["started_at"] == "2026-11-13T09:00:00+00:00"          # the V2 run's real date
    entries = {e["call_id"]: e for e in metadata["calls"]}
    assert sorted(entries) == ["CALL_001", "CALL_002", "CALL_003", "CALL_004"]
    assert entries["CALL_001"]["reused_from_run"] == "clean" and entries["CALL_001"]["v2_condition"] == "C1"
    assert entries["CALL_001"]["original_run_started_at"] == "2026-11-06T00:00:00+00:00"
    assert entries["CALL_001"]["job"]["CreationTime"] == "2026-11-06T00:00:00+00:00"                      # the original job record is not rewritten
    assert "reused_from_run" not in entries["CALL_002"] and metadata["reused_clean_controls"] == ["CALL_001"]
    assert metadata["skipped_clean_controls"] == [] and metadata["assembled_from"] == {"v2_run": "v2", "clean_run": "clean"}
    reused = json.loads((runs / "full" / "raw" / "transcripts" / "CALL_001.json").read_text(encoding="utf-8"))
    assert reused["results"]["transcripts"][0]["transcript"] == "clean CALL_001"                           # the clean run's text
    fresh = json.loads((runs / "full" / "raw" / "transcripts" / "CALL_002.json").read_text(encoding="utf-8"))
    assert fresh["results"]["transcripts"][0]["transcript"] == "v2 CALL_002"
    assert {p: digest(p) for p in (runs / "clean").rglob("*") if p.is_file()} == before                    # the source runs are not modified
    assert not (runs / "v2" / "raw" / "transcripts" / "CALL_001.json").exists()


@pytest.mark.parametrize("change", [{"env": {"TRANSCRIBE_LANGUAGE_OPTIONS": "en-US"}}, {"job": {"Settings": {"ShowSpeakerLabels": False}}},
                                    {"job": dict(JOB, VocabularyName="my-vocab")}])
def test_the_clean_transcripts_are_not_reused_when_the_transcription_settings_differ(tmp_path, change):
    root = tmp_path / "runs"
    collected(root, "clean", ["CALL_001", "CALL_002"], "clean")
    collected(root, "v2", ["CALL_002"], "environment_v2", skipped=["CALL_001"], **change)
    with pytest.raises(cmp.CompareError, match="re-transcribed"):
        asm.assemble(root, "v2", "clean", "full")
    assert not (root / "full").exists()


def test_it_refuses_an_existing_output_a_missing_clean_transcript_and_the_wrong_kind_of_run(runs):
    asm.assemble(runs, "v2", "clean", "full")
    with pytest.raises(cmp.CompareError, match="already exists"):
        asm.assemble(runs, "v2", "clean", "full")
    (runs / "clean" / "raw" / "transcripts" / "CALL_001.json").unlink()
    with pytest.raises(cmp.CompareError, match="no completed transcript for: CALL_001"):
        asm.assemble(runs, "v2", "clean", "other")
    with pytest.raises(cmp.CompareError, match="not an environment_v2 run"):
        asm.assemble(runs, "clean", "clean", "third")


def test_comparing_an_assembled_run_pairs_only_the_new_calls_and_counts_the_reused_ones_once(runs, tmp_path):
    asm.assemble(runs, "v2", "clean", "full")
    dataset = tmp_path / "toy"
    for call, language in LANGS.items():
        (dataset / "calls" / call).mkdir(parents=True)
        (dataset / "calls" / call / "ground_truth.json").write_text(json.dumps({"metadata": {"language_profile": language}}), encoding="utf-8")
    errors = {"clean": {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "full": {"CALL_001": 2, "CALL_002": 5, "CALL_003": 10, "CALL_004": 3}}
    for run_id, per_call in errors.items():
        (runs / run_id / "scores").mkdir()
        (runs / run_id / "scores" / "summary.json").write_text(json.dumps({"normalizer_version": "1.4.0"}), encoding="utf-8")
        (runs / run_id / "scores" / "calls.jsonl").write_text("\n".join(json.dumps(row(c, e)) for c, e in per_call.items()) + "\n", encoding="utf-8")
    result = cmp.compare(cmp.load_run(runs, "clean"), cmp.load_run(runs, "full"), dataset)
    assert "C1" not in result["paired_by_condition"]                      # a copied transcript is not a second measurement
    assert result["reused_clean_calls"] == 1 and result["v2_dataset_by_condition"]["C1"]["source"] == "reused_from_clean_run"
    assert result["paired_by_condition"]["N1"]["wer"]["delta_pp"] == 4.5


def test_a_changed_handler_file_with_identical_job_parameters_is_allowed_but_warned(tmp_path):
    root = tmp_path / "runs"
    collected(root, "clean", ["CALL_001", "CALL_002"], "clean")
    collected(root, "v2", ["CALL_002"], "environment_v2", skipped=["CALL_001"], handler="bbb")
    result = asm.assemble(root, "v2", "clean", "full")
    assert result["reused_clean_calls"] == 1 and any("handler file differs" in w for w in result["warnings"])
