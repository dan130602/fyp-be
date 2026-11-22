"""Regression tests for the pre-launch repairs: exact-35 evaluation, epoch completion/resume binding, checkpoint selection, critical-insertion detection."""
import hashlib
import json

import pytest

import sft_core as C
import sft_data as D
import sft_eval as E
import sft_select as S
import sft_train as T
import test_sft as H
import test_sft_run as R


@pytest.fixture(autouse=True)
def small_speech_token(monkeypatch):
    monkeypatch.setattr(C, "SPEECH_TOKEN_ID", H.SPEECH)


# ---------------------------------------------------------------- evaluation must cover exactly the frozen validation calls
def write_docs(folder, ids, empty=()):
    folder.mkdir(parents=True, exist_ok=True)
    for i in ids:
        (folder / f"{i}.json").write_text(json.dumps({"jobName": i, "meralion": {"transcript_raw": "" if i in empty else "hello", "trace": {"windows": []}}}))


def test_evaluation_refuses_missing_extra_duplicate_failed_or_empty(tmp_path):
    ids = [f"CALL_{i:03d}" for i in range(35)]
    write_docs(tmp_path / "ok", ids)
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"done": 35, "failed": 0, "ok": True}))
    assert E.check_docs(tmp_path / "ok", ids, summary) == ids
    write_docs(tmp_path / "missing", ids[:-1])
    with pytest.raises(E.EvaluationIncomplete, match="missing calls"):
        E.check_docs(tmp_path / "missing", ids)
    write_docs(tmp_path / "extra", ids + ["CALL_999"])
    with pytest.raises(E.EvaluationIncomplete, match="outside the frozen"):
        E.check_docs(tmp_path / "extra", ids)
    write_docs(tmp_path / "empty", ids, empty=[ids[3]])
    with pytest.raises(E.EvaluationIncomplete, match="empty transcript"):
        E.check_docs(tmp_path / "empty", ids)
    summary.write_text(json.dumps({"done": 34, "failed": 1, "ok": False}))
    with pytest.raises(E.EvaluationIncomplete, match="worker summary"):
        E.check_docs(tmp_path / "ok", ids, summary)
    with pytest.raises(E.EvaluationIncomplete, match="duplicates"):
        E.check_docs(tmp_path / "ok", ids[:-1] + [ids[0]])


def test_formatter_crosscheck_mismatch_is_fatal(tmp_path, monkeypatch):
    ids = ["CALL_001"]
    write_docs(tmp_path / "d", ids)
    valid = tmp_path / "v.json"
    valid.write_text(json.dumps({"call_ids": ids}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"per_call": {"CALL_001": {"language": "ENGLISH"}}}))
    monkeypatch.setattr(E, "make_predictions", lambda docs, out, worker: ({}, ["CALL_001"]))
    with pytest.raises(E.EvaluationIncomplete, match="differs from the vendored formatters"):
        E.main(["score", "--docs", str(tmp_path / "d"), "--exp", "X", "--ckpt", "c", "--out", str(tmp_path / "o"), "--split", str(split), "--valid", str(valid)])


def test_decode_health_and_critical_insertion_detection(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CALL_001.json").write_text(json.dumps({"meralion": {"trace": {"windows": [{"start_sample": 0, "stop_reason": "end_token", "tokens": 5}, {"start_sample": 1, "stop_reason": "max_new_tokens", "tokens": 512}]}}}))
    health = E.decode_health(docs)
    assert health["abnormal_stops"] == 1 and health["token_limit_stops"] == 1 and health["windows"] == 2
    for text, expected in (("I do not agree", True), ("transfer 50,000 now", True), ("同意", True), ("the weather is nice", False), ("knowledge", False)):
        assert bool(E.CRITICAL.search(text)) == expected, text
    units = tmp_path / "run" / "scores" / "units"
    base_units = tmp_path / "base" / "scores" / "units"
    for folder, rows in ((units, [{"turn_id": 1, "ref_text": "fine", "hyp_text": "not fine", "insertions": 1}, {"turn_id": 2, "ref_text": "a", "hyp_text": "so a", "insertions": 1}]),
                         (base_units, [{"turn_id": 2, "ref_text": "a", "hyp_text": "so a", "insertions": 1}])):
        folder.mkdir(parents=True)
        (folder / "CALL_001.json").write_text(json.dumps({"units": rows}))
    report = E.critical_insertions(tmp_path / "run", tmp_path / "base", {"CALL_001": "ENGLISH"}, tmp_path / "ci.csv")
    assert report == {"candidate_units": 1, "baseline_units": 0, "new_units": 1, "new_inserted_words": 1, "resolved_vs_baseline": 0}


# ---------------------------------------------------------------- epoch completion and resume binding
def args_for(tmp_path, data, batch, run_id, extra=()):
    return R.args_for(tmp_path, data, batch, run_id, k=4, extra=extra)


def test_checkpoint_file_alone_does_not_complete_an_epoch_and_resume_repairs_it(tmp_path, monkeypatch):
    data, batch, n, m = R.make_package(tmp_path)
    run = tmp_path / "runs" / "X"
    # the original job crashes AFTER the epoch-1 checkpoint is written but BEFORE validation/state/marker (one validation per epoch: the crash is injected at the epoch-end validation)
    original = T.evaluate_ce
    monkeypatch.setattr(T, "evaluate_ce", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        T.run(args_for(tmp_path, data, batch, "X", ("--stop-after-epoch", "1", "--skip-baseline-validation", "--validate-per-epoch", "1")), R.stack())
    monkeypatch.setattr(T, "evaluate_ce", original)
    assert (run / "checkpoints" / "epoch-1.safetensors").exists() and not (run / "checkpoints" / "epoch-1.complete.json").exists()
    rows_before = (run / "train_metrics.jsonl").read_text().count("\n")
    T.run(args_for(tmp_path, data, batch, "X", ("--stop-after-epoch", "1", "--skip-baseline-validation", "--validate-per-epoch", "1", "--resume")), R.stack())
    marker = json.loads((run / "checkpoints" / "epoch-1.complete.json").read_text())
    assert marker["epoch"] == 1
    updates = [json.loads(l)["update"] for l in (run / "train_metrics.jsonl").read_text().splitlines()]
    assert updates == sorted(set(updates)) and rows_before >= len(updates) > 0                      # no duplicated rows after the repeat
    # an uninterrupted run reaches the same weights
    T.run(args_for(tmp_path, data, batch, "Y", ("--stop-after-epoch", "1", "--skip-baseline-validation", "--validate-per-epoch", "1")), R.stack())
    assert R.digest_of(tmp_path, "X", 1) == R.digest_of(tmp_path, "Y", 1)


def test_state_past_the_requested_epoch_only_rewrites_the_marker(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    run = tmp_path / "runs" / "Z"
    T.run(args_for(tmp_path, data, batch, "Z", ("--stop-after-epoch", "1", "--skip-baseline-validation", "--validate-per-epoch", "1")), R.stack())
    (run / "checkpoints" / "epoch-1.complete.json").unlink()                                        # crash between state save and marker
    T.run(args_for(tmp_path, data, batch, "Z", ("--stop-after-epoch", "1", "--skip-baseline-validation", "--validate-per-epoch", "1", "--resume")), R.stack())
    assert (run / "checkpoints" / "epoch-1.complete.json").exists() and not (run / "checkpoints" / "epoch-2.safetensors").exists()


def test_resume_is_bound_to_the_original_data_manifest_and_initialisation(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    T.run(args_for(tmp_path, data, batch, "B", ("--stop-after-epoch", "1", "--skip-baseline-validation", "--validate-per-epoch", "1")), R.stack())
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["note"] = "edited after the run started"
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="data manifest"):
        T.run(args_for(tmp_path, data, batch, "B", ("--stop-after-epoch", "2", "--skip-baseline-validation", "--validate-per-epoch", "1", "--resume")), R.stack())


# ---------------------------------------------------------------- selection
def metrics(wer_err=500, ent_err=190, lang_err=125, events=10, fixed=0, regress=0, new_ins=0, limit=0, abnormal=0):
    scope = {"ALL": {"wer": {"errors": wer_err, "reference_words": 45000}, "entity": {"errors": ent_err, "reference_words": 3800}}}
    for lang in S.LANGS:
        scope[lang] = {"wer": {"errors": lang_err, "reference_words": 11000}, "entity": {"errors": 40, "reference_words": 900}}
    return {"decode_health": {"token_limit_stops": limit, "abnormal_stops": abnormal},
            "stages": {"raw": {"by_scope": scope, "entity_groups": {"event:VALUE_CORRECTION": {"errors": events}, "value:T01": {"errors": 50}},
                               "errors_vs_baseline": {"fixed": fixed, "regression": regress}, "critical_insertions": {"new_units": new_ins}}}}


def test_selection_follows_the_frozen_criteria_not_the_last_epoch():
    base = metrics()
    cands = {"epoch-1": metrics(ent_err=165, fixed=40, regress=10), "epoch-2": metrics(ent_err=160, lang_err=150, fixed=45, regress=8), "epoch-3": metrics(ent_err=150, new_ins=2, fixed=50, regress=5)}
    result = S.select(base, cands, {"epoch-1": 0.2, "epoch-2": 0.19, "epoch-3": 0.18})
    assert result["table"]["epoch-2"]["status"] == "REJECTED"                                       # language guard: 150 vs 125 errors on 11,000 words: +0.23 pp > min(0.20, 15% of 1.14)
    assert result["table"]["epoch-3"]["status"] == "PENDING_CRITICAL_INSERTION_REVIEW"
    assert result["selected"] == "epoch-1" and result["table"]["epoch-1"]["improved"]
    reviewed = S.select(base, cands, {}, reviewed=["epoch-3"])
    assert reviewed["selected"] == "epoch-3"


def test_selection_keeps_the_original_when_nothing_qualifies_and_ties_prefer_lower_wer():
    base = metrics()
    assert S.select(base, {"epoch-1": metrics(events=11), "epoch-2": metrics(limit=1)}, {})["selected"] is None
    tie = S.select(base, {"epoch-1": metrics(ent_err=170, wer_err=505), "epoch-2": metrics(ent_err=173, wer_err=490)}, {})
    assert tie["selected"] == "epoch-2"                                                             # within 5 entity errors: lower WER wins
    weak = S.select(base, {"epoch-1": metrics(ent_err=185)}, {})
    assert weak["selected"] == "epoch-1" and not weak["table"]["epoch-1"]["improved"] and "NOT met" in weak["decision"]


def test_pack_eval_archives_then_removes_only_the_named_folders(tmp_path):
    import tarfile
    import pack_eval
    runs, out = tmp_path / "runs", tmp_path / "out"
    for stage in ("raw", "final"):
        d = runs / f"sft-E-epoch-1-{stage}-1-prelim-1" / "scores"
        d.mkdir(parents=True)
        (d / "summary.json").write_text("{}")
    other = runs / "sft-keep-prelim-1"
    other.mkdir()
    (out / "worker" / "out").mkdir(parents=True)
    (out / "worker" / "out" / "CALL_001.json").write_text("{}")
    (out / "predictions_raw").mkdir()
    (out / "metrics.json").write_text("{}")
    (out / "engine_runs.json").write_text(json.dumps({s: str(runs / f"sft-E-epoch-1-{s}-1-prelim-1") for s in ("raw", "final")}))
    with pytest.raises(SystemExit):
        pack_eval.pack(tmp_path / "nowhere", runs)
    pack_eval.pack(out, runs)
    assert other.is_dir() and not (out / "worker").exists() and not (out / "predictions_raw").exists()
    assert sorted(p.name for p in runs.iterdir()) == ["sft-keep-prelim-1"]
    with tarfile.open(out / "engine_runs.tar.gz") as tar:
        assert any(m.name.endswith("scores/summary.json") for m in tar.getmembers())
    with tarfile.open(out / "worker.tar.gz") as tar:
        assert any(m.name.endswith("CALL_001.json") for m in tar.getmembers())
