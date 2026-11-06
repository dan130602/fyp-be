"""The export bundle: self-contained, free of machine paths and audio, and consistent with the scores it came from."""
import json

import pytest

from transcribe_eval.export import ExportError, export_run, frequent_errors
from transcribe_eval.score import score_run

from test_score import ADVISORY, items_from, load_turns


def scored_run(tmp_path, wrong_first_word=True):
    turns = load_turns(ADVISORY, "CALL_001")
    raw = []
    for index, t in enumerate(turns):
        words = t["text"].split()
        if wrong_first_word and index == 0:
            words[0] = "zzz"
        raw += items_from(words, speaker="spk_0" if t["speaker"] == "EMPLOYEE" else "spk_1")
    run = tmp_path / "runs" / "run1"
    (run / "raw" / "transcripts").mkdir(parents=True)
    (run / "raw" / "transcripts" / "CALL_001.json").write_text(json.dumps({"results": {"items": raw, "transcripts": [{"transcript": "x"}]}}),
                                                                encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps({
        "run_id": "run1", "dataset_id": ADVISORY.name, "render_id": "clean_voice_A", "started_at": "2026-10-17T00:00:00Z",
        "handler": {"git_commit": "abc", "uncommitted_changes": False}, "handler_environment": {}, "region": "ap-southeast-1",
        "calls": [{"call_id": "CALL_001", "status": "COMPLETED", "transcript_file": "raw/transcripts/CALL_001.json", "audio_sha256": "f" * 64,
                   "job": {"LanguageCode": "en-US", "Settings": {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 2}}}]}), encoding="utf-8")
    score_run(run, dataset_root=ADVISORY)
    return run


def test_an_unscored_run_is_refused(tmp_path):
    run = tmp_path / "runs" / "run1"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"dataset_id": "x", "calls": []}), encoding="utf-8")
    with pytest.raises(ExportError, match="not scored"):
        export_run(run, tmp_path / "out")


def test_the_bundle_is_self_contained_and_consistent_with_the_scores(tmp_path):
    bundle = export_run(scored_run(tmp_path), tmp_path / "out", dataset_root=ADVISORY)
    run = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
    call = json.loads((bundle / "calls" / "CALL_001.json").read_text(encoding="utf-8"))
    assert run["summary"]["normalised_wer"]["errors"] == 1 and run["calls"][0]["wer"]["substitutions"] == 1
    assert call["audio"] == {"dataset_id": ADVISORY.name, "call_id": "CALL_001", "render_id": "clean_voice_A", "audio_sha256": "f" * 64}
    assert len(call["reference_fingerprints"]["ground_truth_sha256"]) == 64 and call["turns"][0]["start"] is not None
    assert len(run["dataset_call_ids"]) == 50 and run["transcribe_job"]["language_code"] == "en-US"
    assert sum(u["substitutions"] for u in call["units"]) == run["calls"][0]["wer"]["substitutions"]
    errors = json.loads((bundle / "errors.json").read_text(encoding="utf-8"))
    assert [(e["op"], e["hyp"], e["count"]) for e in errors] == [("substitute", "zzz", 1)]
    assert errors[0]["occurrences"][0]["call_id"] == "CALL_001" and errors[0]["examples"]


def test_no_machine_paths_audio_files_or_secrets_are_written(tmp_path):
    bundle = export_run(scored_run(tmp_path), tmp_path / "out", dataset_root=ADVISORY)
    assert not list(bundle.rglob("*.mp3"))
    for path in bundle.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text and "\\Users\\" not in text and "/Users/" not in text and "C:\\" not in text, path.name
        assert "AWS_SECRET" not in text and "AKIA" not in text


def test_overwrite_is_explicit_and_a_dirty_tree_does_not_block(tmp_path):
    run = scored_run(tmp_path)
    export_run(run, tmp_path / "out", dataset_root=ADVISORY)
    with pytest.raises(ExportError, match="already exists"):
        export_run(run, tmp_path / "out", dataset_root=ADVISORY)
    export_run(run, tmp_path / "out", overwrite=True, dataset_root=ADVISORY)  # no commit or clean tree needed


def test_frequent_errors_put_entity_errors_first_and_count_every_occurrence():
    unit = lambda uid, part: {"id": uid, "turn_id": 0, "ref_text": "a", "hyp_text": "b", "parts": [part]}
    rows = frequent_errors({"CALL_001": [unit(0, {"ref": "x", "hyp": "y", "op": "substitute", "entity": False}),
                                         unit(1, {"ref": "p", "hyp": "q", "op": "substitute", "entity": True}),
                                         unit(2, {"ref": "x", "hyp": "y", "op": "substitute", "entity": False})]})
    assert [(r["ref"], r["count"], r["entity_count"]) for r in rows] == [("p", 1, 1), ("x", 2, 0)]
    assert [o["unit"] for o in rows[1]["occurrences"]] == [0, 2]
