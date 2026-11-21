"""tools/full_report.py and tools/repeat_compare.py on synthetic runs (no git-ignored data)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


full_report = load_tool("full_report")
repeat_compare = load_tool("repeat_compare")
CALLS = ["c1", "c2", "c3"]
LANGUAGE = {"c1": "ENGLISH", "c2": "ENGLISH", "c3": "MANDARIN"}


def document(call, raw, final, windows, retries):
    return {"jobName": call, "results": {"transcripts": [{"transcript": final}]},
            "meralion": {"transcript_raw": raw, "audio": {"seconds": 60.0}, "timing": {"model_load_s": 30.0},
                         "trace": {"windows": [{"start_sample": s, "end_sample": e, "text": t, "tokens": 5, "stop_reason": "end_token", "decode_seconds": 0.1, "native": {}} for (s, e), t in windows.items()]},
                         "retries": {"decodes": [{"retry_id": f"{call}_r{i:03d}", "window": list(w), "result": {"text": t, "tokens": 5, "stop_reason": "end_token", "decode_seconds": 0.1, "native": {}}} for i, (w, t) in enumerate(retries.items())]}}}


def write_docs(folder, docs):
    folder.mkdir(parents=True)
    for call, doc in docs.items():
        (folder / f"{call}.json").write_text(json.dumps(doc), encoding="utf-8")
    return folder


def write_scores(runs, run_id, errors, entity=2):
    base = runs / run_id / "scores"
    base.mkdir(parents=True)
    rows = [{"call_id": c, "wer": {"errors": errors[c], "reference_words": 1000}, "entity_wer": {"errors": entity, "reference_words": 100}} for c in CALLS]
    (base / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def two_runs(tmp_path, differ=False):
    def docs(variant):
        out = {}
        for call in CALLS:
            windows = {(0, 480000): f"text {call}", (240000, 720000): f"second {call}"}
            if variant and call == "c2":
                windows[(0, 480000)] = "different first window"
            retries = {(240000, 720000): f"retry {call}"}
            if variant and call == "c3":
                retries = {(240000, 720000): "another retry", (0, 480000): "extra retry"}
            out[call] = document(call, f"raw {call}" + (" x" if variant and call == "c2" else ""), f"final {call}", windows, retries)
        return out
    return write_docs(tmp_path / "a", docs(False)), write_docs(tmp_path / "b", docs(differ))


def test_identical_runs_report_no_difference_and_say_that_equal_scores_prove_nothing(tmp_path):
    a, b = two_runs(tmp_path)
    runs = tmp_path / "runs"
    for run in ("ra", "rb"):
        write_scores(runs, run, {"c1": 3, "c2": 4, "c3": 5})
    out = tmp_path / "repeat.json"
    assert repeat_compare.main(["--docs-a", str(a), "--docs-b", str(b), "--runs", str(runs), "--score", "raw=ra,rb", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["transcripts"]["raw_differing"] == 0 and report["first_pass_windows"]["differing_text"] == 0 and report["retry_plan"]["planned_by_a_only"] == 0
    assert report["scores"]["raw"]["wer"]["difference_b_minus_a"] == 0 and "do not establish guaranteed determinism" in report["note"]
    with pytest.raises(SystemExit, match="already exists"):
        repeat_compare.main(["--docs-a", str(a), "--docs-b", str(b), "--runs", str(runs), "--out", str(out)])


def test_every_kind_of_difference_is_found_separately_even_when_the_aggregate_error_counts_are_equal(tmp_path):
    a, b = two_runs(tmp_path, differ=True)
    runs = tmp_path / "runs"
    write_scores(runs, "ra", {"c1": 3, "c2": 4, "c3": 5})
    write_scores(runs, "rb", {"c1": 3, "c2": 5, "c3": 4})                                   # the totals are equal; two calls changed
    report = repeat_compare.build(type("A", (), dict(docs_a=a, docs_b=b, label_a="r1", label_b="r2", runs=runs, score=["raw=ra,rb"], config_compare=[]))())
    assert report["scores"]["raw"]["wer"]["difference_b_minus_a"] == 0 and report["scores"]["raw"]["wer"]["calls_with_a_different_count"] == 2
    assert report["transcripts"]["raw_differing_calls"] == ["c2"] and report["transcripts"]["identical_final"] == 3
    assert report["first_pass_windows"]["differing_calls"] == ["c2"] and report["first_pass_windows"]["differing_text"] == 1
    assert report["retry_plan"]["planned_by_b_only"] == 1 and report["retry_plan"]["b_only"] == [{"call": "c3", "window": [0, 480000]}] and report["retry_plan"]["planned_by_a_only"] == 0
    assert report["retry_outputs"]["differing_text_on_common_windows"] == 1


def test_a_recording_held_by_one_run_only_is_reported(tmp_path):
    a, b = two_runs(tmp_path)
    (b / "c3.json").unlink()
    report = repeat_compare.compare_documents(acceptance_scan(a), acceptance_scan(b))
    assert report["recordings"]["only_a"] == ["c3"] and report["recordings"]["shared"] == 2


def acceptance_scan(folder):
    import acceptance
    return acceptance.scan_documents(folder)[0]


def test_the_report_gives_integer_counts_for_all_the_pilot_and_the_remaining_recordings_and_each_repetition_separately(tmp_path):
    a, b = two_runs(tmp_path)
    runs = tmp_path / "runs"
    for run, errors in (("base-raw", {"c1": 5, "c2": 5, "c3": 5}), ("base-final", {"c1": 4, "c2": 4, "c3": 4}), ("r1-raw", {"c1": 4, "c2": 6, "c3": 5}), ("r1-final", {"c1": 3, "c2": 4, "c3": 4}),
                        ("r2-raw", {"c1": 5, "c2": 5, "c3": 5}), ("r2-final", {"c1": 4, "c2": 4, "c3": 4})):
        write_scores(runs, run, errors)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": c, "input": "/x"} for c in CALLS]}), encoding="utf-8")
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"per_call": {"c1": {}, "c2": {}}}), encoding="utf-8")
    languages = tmp_path / "languages.json"
    languages.write_text(json.dumps({c: {"language": l} for c, l in LANGUAGE.items()}), encoding="utf-8")
    compare = tmp_path / "compare.json"
    compare.write_text(json.dumps({"candidates": {"x": {"paired_wer": {"calls_better": 1}, "large_omission_events": {"new": [], "recovered": []}, "critical_flags": [{"flag": "dropped"}], "by_language": {}}}}), encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"verdict": "ACCEPTED", "failed_items": [], "not_established": ["same-hardware comparison"],
                                    "quality_details": {"new_large_omissions": {"raw": 0, "final": 0}, "new_loops": 0, "candidate_stop_reasons": {"end_token": 10}}}), encoding="utf-8")
    run_folder = tmp_path / "runfolder"
    run_folder.mkdir()
    (run_folder / "wall.json").write_text(json.dumps({"wall_seconds": 90, "exit_code": 0}), encoding="utf-8")
    (run_folder / "worker.log").write_text("INFO Model loading took 6.2 GiB memory\nnoise\nINFO Available KV cache memory: 20.1 GiB\n", encoding="utf-8")
    out = tmp_path / "report.json"
    argv = ["--jobs", str(jobs), "--pilot", str(pilot), "--languages", str(languages), "--runs", str(runs), "--out", str(out), "--baseline-raw", "base-raw", "--baseline-final", "base-final"]
    for name, raw, final, docs in (("r1", "r1-raw", "r1-final", a), ("r2", "r2-raw", "r2-final", b)):
        argv += ["--repetition", name, raw, final, str(docs), str(run_folder), str(decision), str(compare), str(compare)]
    assert full_report.main(argv) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert list(report["repetitions"]) == ["r1", "r2"] and report["groups"] == {"all": 3, "pilot_22_selection_influenced": 2, "remaining_not_in_pilot": 1, "language:ENGLISH": 2, "language:MANDARIN": 1}
    r1 = report["repetitions"]["r1"]
    assert r1["raw"]["all"]["wer"] == {"errors": 15, "reference_tokens": 3000, "percent": 0.5} and r1["raw"]["pilot_22_selection_influenced"]["wer"]["errors"] == 10 and r1["raw"]["remaining_not_in_pilot"]["wer"]["errors"] == 5
    assert r1["raw"]["language:ENGLISH"]["wer"]["errors"] == 10 and report["baseline"]["raw"]["all"]["wer"]["errors"] == 15
    paired = r1["raw_paired_wer_against_baseline"]["all"]
    assert paired["calls_better"] == 1 and paired["calls_worse"] == 1 and paired["error_change"] == 0 and paired["largest_regressions"] == [["c2", 1]] or paired["largest_regressions"] == [("c2", 1)]
    assert r1["facts"]["wall_seconds"] == 90 and r1["facts"]["audio_seconds_per_wall_second"] == 2.0 and len(r1["facts"]["engine_memory_lines"]) == 2 and r1["facts"]["decodes"] == 3 * 3
    assert r1["not_established"] == ["same-hardware comparison"] and "not independent validation" in report["pilot_note"]
    assert r1["raw_config_compare"]["critical_flags"] == 1
    with pytest.raises(SystemExit, match="already exists"):
        full_report.main(argv)


def test_languages_come_from_ground_truth_metadata_only(tmp_path):
    root = tmp_path / "dev"
    for call, language in LANGUAGE.items():
        (root / "calls" / call).mkdir(parents=True)
        (root / "calls" / call / "ground_truth.json").write_text(json.dumps({"metadata": {"language_profile": language}, "dialogue": ["secret reference"]}), encoding="utf-8")
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": c, "input": "/x"} for c in CALLS]}), encoding="utf-8")
    out = tmp_path / "languages.json"
    assert full_report.main(["--jobs", str(jobs), "--write-languages", str(out), "--dataset-root", str(root)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == {c: {"language": l} for c, l in LANGUAGE.items()} and "secret" not in out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        full_report.main(["--jobs", str(jobs), "--write-languages", str(out), "--dataset-root", str(root)])
