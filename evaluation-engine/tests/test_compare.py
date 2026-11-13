"""Paired clean-versus-environment_v2 comparison on fake scored runs (offline)."""
import json

import pytest

from transcribe_eval import compare as cmp

JOB = {"Settings": {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 2}, "LanguageOptions": ["en-US", "zh-CN"], "IdentifyMultipleLanguages": True}
LANGS = {"CALL_001": "ENGLISH", "CALL_002": "ENGLISH", "CALL_003": "MANDARIN", "CALL_004": "MANDARIN"}
CONDS = {"CALL_001": "C1", "CALL_002": "N1", "CALL_003": "N1", "CALL_004": "D1"}


def row(call, errors, entity_errors=1, correct=98):
    return {"call_id": call, "wer": {"errors": errors, "reference_words": 100}, "entity_wer": {"errors": entity_errors, "reference_words": 10},
            "speaker": {"correct_items": correct, "eligible_items": 100}}


def make_run(root, run_id, errors, condition, skipped=(), handler="aaa", started="2026-11-10T00:00:00+00:00", normaliser="1.4.0", env=None, job=None):
    folder = root / run_id
    (folder / "scores").mkdir(parents=True)
    calls = [{"call_id": c, "status": "COMPLETED", "v2_condition": CONDS[c] if condition == "environment_v2" else None, "job": job or JOB} for c in errors]
    metadata = {"run_id": run_id, "started_at": started, "dataset_id": "toy", "audio_condition": condition,
                "skipped_clean_controls": list(skipped), "handler": {"sha256": handler},
                "handler_environment": env or {"TRANSCRIBE_LANGUAGE_OPTIONS": "en-US,zh-CN"}, "calls": calls}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "scores" / "summary.json").write_text(json.dumps({"normalizer_version": normaliser}), encoding="utf-8")
    (folder / "scores" / "calls.jsonl").write_text("\n".join(json.dumps(row(c, e)) for c, e in errors.items()) + "\n", encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    dataset = tmp_path / "toy"
    for call, language in LANGS.items():
        (dataset / "calls" / call).mkdir(parents=True)
        (dataset / "calls" / call / "ground_truth.json").write_text(json.dumps({"metadata": {"language_profile": language}}), encoding="utf-8")
    return tmp_path / "runs", dataset


def load(root, run_id):
    return cmp.load_run(root, run_id)


def test_paired_deltas_are_per_call_and_grouped_by_condition_and_language(world):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "clean")
    make_run(root, "v2", {"CALL_002": 5, "CALL_003": 10, "CALL_004": 3}, "environment_v2", skipped=["CALL_001"])
    result = cmp.compare(load(root, "clean"), load(root, "v2"), dataset)
    assert result["equivalent"] and result["reused_clean_calls"] == 1
    assert result["paired_by_condition"]["N1"]["wer"]["delta_pp"] == 4.5          # (15/200) - (6/200) = 4.5 points
    assert result["paired_by_condition"]["N1"]["calls"] == 2 and result["paired_by_condition"]["D1"]["wer"]["delta_pp"] == 0.0
    assert result["paired_by_condition_language"]["N1|ENGLISH"]["wer"]["delta_pp"] == 3.0
    assert result["paired_by_condition_language"]["N1|MANDARIN"]["wer"]["delta_pp"] == 6.0
    assert result["paired_by_condition"]["N1"]["wer"]["ci95_pp"] is not None
    assert result["v2_dataset_by_condition"]["C1"]["source"] == "reused_from_clean_run" and result["v2_dataset_by_condition"]["C1"]["wer"] == 2.0


def test_the_comparison_is_reproducible(world):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "clean")
    make_run(root, "v2", {"CALL_002": 5, "CALL_003": 10, "CALL_004": 3}, "environment_v2", skipped=["CALL_001"])
    first = cmp.compare(load(root, "clean"), load(root, "v2"), dataset)
    assert first == cmp.compare(load(root, "clean"), load(root, "v2"), dataset)


@pytest.mark.parametrize("change,reason", [({"env": {"TRANSCRIBE_LANGUAGE_OPTIONS": "en-US"}}, "TRANSCRIBE_"),
                                           ({"normaliser": "1.5.0"}, "normaliser version differs"),
                                           ({"job": {"Settings": {"ShowSpeakerLabels": False}}}, "job parameters differ"),
                                           ({"job": dict(JOB, VocabularyName="my-vocab")}, "job parameters differ")])
def test_clean_controls_are_not_reused_when_the_settings_differ(world, change, reason):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "clean")
    make_run(root, "v2", {"CALL_002": 5, "CALL_003": 10, "CALL_004": 3}, "environment_v2", skipped=["CALL_001"], **change)
    with pytest.raises(cmp.CompareError, match="re-transcribed.*" + reason):
        cmp.compare(load(root, "clean"), load(root, "v2"), dataset)


def test_different_settings_without_skipped_controls_still_compare_but_say_not_equivalent(world):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "clean")
    make_run(root, "v2", {"CALL_001": 2, "CALL_002": 5, "CALL_003": 10, "CALL_004": 3}, "environment_v2", job=dict(JOB, VocabularyName="my-vocab"))
    result = cmp.compare(load(root, "clean"), load(root, "v2"), dataset)
    assert not result["equivalent"] and result["reused_clean_calls"] == 0
    assert result["paired_by_condition"]["C1"]["wer"]["delta_pp"] == 0.0          # re-transcribed controls give the run-to-run noise floor


def test_runs_far_apart_in_time_get_a_warning(world):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "clean", started="2026-09-01T00:00:00+00:00")
    make_run(root, "v2", {"CALL_002": 5, "CALL_003": 10, "CALL_004": 3}, "environment_v2", skipped=["CALL_001"])
    assert any("days apart" in w for w in cmp.compare(load(root, "clean"), load(root, "v2"), dataset)["warnings"])


def test_a_clean_run_cannot_be_passed_as_the_v2_run_and_calls_missing_from_the_clean_run_are_listed(world):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2}, "clean")
    make_run(root, "other", {"CALL_001": 2}, "clean")
    with pytest.raises(cmp.CompareError, match="not an environment_v2 run"):
        cmp.compare(load(root, "clean"), load(root, "other"), dataset)
    make_run(root, "v2", {"CALL_002": 5, "CALL_003": 10}, "environment_v2")
    assert cmp.compare(load(root, "clean"), load(root, "v2"), dataset)["calls_missing_in_clean_run"] == ["CALL_003"]


def test_a_changed_handler_file_alone_only_warns_because_the_recorded_job_parameters_are_what_matter(world):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "clean")
    make_run(root, "v2", {"CALL_002": 5, "CALL_003": 10, "CALL_004": 3}, "environment_v2", skipped=["CALL_001"], handler="bbb")
    result = cmp.compare(load(root, "clean"), load(root, "v2"), dataset)
    assert result["equivalent"] and result["reused_clean_calls"] == 1
    assert any("handler file differs" in w for w in result["warnings"])


def test_a_different_output_bucket_does_not_block_reuse(world):
    root, dataset = world
    make_run(root, "clean", {"CALL_001": 2, "CALL_002": 2, "CALL_003": 4, "CALL_004": 3}, "clean",
             env={"TRANSCRIBE_LANGUAGE_OPTIONS": "en-US,zh-CN", "TRANSCRIBE_OUTPUT_BUCKET": "old-bucket"})
    make_run(root, "v2", {"CALL_002": 5, "CALL_003": 10, "CALL_004": 3}, "environment_v2", skipped=["CALL_001"],
             env={"TRANSCRIBE_LANGUAGE_OPTIONS": "en-US,zh-CN", "TRANSCRIBE_OUTPUT_BUCKET": "new-bucket"})
    assert cmp.compare(load(root, "clean"), load(root, "v2"), dataset)["equivalent"]
