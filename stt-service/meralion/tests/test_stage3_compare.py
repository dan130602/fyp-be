"""to_predictions and compare_backends on synthetic documents (no git-ignored data)."""
import copy
import importlib.util
import sys
import json
import re
from pathlib import Path

import pytest

from harness import run, two_jobs

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


to_predictions = load_tool("to_predictions")
compare_backends = load_tool("compare_backends")
sys.path.insert(0, str(TOOLS))
import acceptance  # noqa: E402


@pytest.fixture
def docs(tmp_path):
    run(two_jobs(), tmp_path)
    folder = tmp_path / "run" / "out" / "transcripts"
    return {json.loads(p.read_text(encoding="utf-8"))["jobName"]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}, folder


# ---- to_predictions ------------------------------------------------------------------------------------------------------------------------------------------------------

def test_raw_and_final_predictions_carry_the_matching_text_and_one_shared_settings_block(docs, tmp_path):
    documents, folder = docs
    raw = to_predictions.convert(folder, "raw", tmp_path / "raw")
    final = to_predictions.convert(folder, "final", tmp_path / "final")
    assert raw == final == ["alpha", "beta"]
    rows = {v: {c: json.loads((tmp_path / v / f"{c}.json").read_text(encoding="utf-8")) for c in raw} for v in ("raw", "final")}
    for call in raw:
        assert rows["raw"][call]["transcript"] == documents[call]["meralion"]["transcript_raw"]
        assert rows["final"][call]["transcript"] == documents[call]["meralion"]["transcript_final"] == documents[call]["results"]["transcripts"][0]["transcript"]
        assert rows["raw"][call]["call_id"] == call and re.fullmatch(r"[0-9a-f]{64}", rows["raw"][call]["settings"]["audio_sha256"])
        assert rows["raw"][call]["audio_seconds"] == documents[call]["meralion"]["audio"]["seconds"] and "chunks" not in rows["raw"][call]
    stripped = [{k: v for k, v in rows["raw"][c]["settings"].items() if k != "audio_sha256"} for c in raw]
    assert stripped[0] == stripped[1]                                                       # one configuration per run, as the importer requires
    settings = rows["raw"]["alpha"]["settings"]
    assert settings["adapter"] is None and settings["training"] is False and settings["maximum_seconds"] == 30 and settings["transcript_variant"] == "raw"
    assert rows["raw"]["beta"]["transcript"] != rows["final"]["beta"]["transcript"] and "0.5 percent" in rows["final"]["beta"]["transcript"]


def test_the_output_folder_must_be_new_and_the_variant_known(docs, tmp_path):
    _, folder = docs
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        to_predictions.convert(folder, "raw", tmp_path / "taken")
    with pytest.raises(SystemExit, match="variant"):
        to_predictions.convert(folder, "other", tmp_path / "new")
    assert to_predictions.main(["--docs", str(folder), "--variant", "final", "--out", str(tmp_path / "ok")]) == 0


# ---- compare_backends ----------------------------------------------------------------------------------------------------------------------------------------------------

def vllm_like(documents):
    copied = copy.deepcopy(documents)
    for document in copied.values():
        for window in document["meralion"]["trace"]["windows"]:
            window["tokens"], window["stop_reason"], window["native"] = 10, "end_token", {"finish_reason": "stop", "stop_reason": None}
        for decode in document["meralion"]["retries"]["decodes"]:
            decode["result"]["tokens"], decode["result"]["stop_reason"], decode["result"]["native"] = 10, "end_token", {"finish_reason": "stop", "stop_reason": None}
    return copied


def test_identical_texts_give_clean_gates(docs):
    documents, _ = docs
    report = compare_backends.compare(documents, vllm_like(documents))
    assert report["decodes"]["differing_text"] == 0 and report["calls"]["shared"] == 2 and report["transcripts"]["raw_equal"] == 2 and report["transcripts"]["final_equal"] == 2
    assert all(report["gates"].values()) and {"all_calls_in_both", "no_new_loops", "no_max_new_tokens_window_in_vllm", "stop_reason_available_in_vllm", "no_aborted_or_unexplained_stop"} <= set(report["gates"])
    assert report["verdict"].startswith("NOT AN ACCEPTANCE VERDICT") and report["acceptance_evidence"] is False
    assert report["token_counts"]["note"].startswith("not comparable")


def test_a_differing_window_is_listed_by_bounds(docs):
    documents, _ = docs
    other = vllm_like(documents)
    other["alpha"]["meralion"]["trace"]["windows"][2]["text"] += " extra"
    report = compare_backends.compare(documents, other)
    assert report["decodes"]["differing_text"] == 1 and report["decodes"]["examples"][0]["call"] == "alpha" and report["decodes"]["examples"][0]["kind"] == "window"


def test_a_loop_only_vllm_has_is_a_new_loop_and_a_shared_loop_is_not(docs):
    documents, _ = docs
    other = vllm_like(documents)
    looping = "one two three four " * 4
    window = other["alpha"]["meralion"]["trace"]["windows"][1]
    window["text"] = looping
    report = compare_backends.compare(documents, other)
    assert report["loops"]["new_in_vllm"] == [{"kind": "window", "call": "alpha", "window": [window["start_sample"], window["end_sample"]]}] and report["gates"]["no_new_loops"] is False
    base = copy.deepcopy(documents)
    base["alpha"]["meralion"]["trace"]["windows"][1]["text"] = looping
    report = compare_backends.compare(base, other)
    assert report["loops"]["new_in_vllm"] == [] and report["loops"]["tf"] == 1 and report["loops"]["vllm"] == 1 and report["gates"]["no_new_loops"] is True


def test_stop_reasons_are_counted_and_the_unusual_ones_keep_their_native_values(docs):
    documents, _ = docs
    other = vllm_like(documents)
    windows = other["beta"]["meralion"]["trace"]["windows"]
    windows[0]["stop_reason"] = "max_new_tokens"
    windows[1].update(stop_reason="other", native={"finish_reason": "abort", "stop_reason": None})
    windows[2].update(stop_reason=None, native={"finish_reason": None, "stop_reason": None})
    report = compare_backends.compare(documents, other)
    counts = report["stop_reasons"]["vllm"]["counts"]
    assert counts["max_new_tokens"] == 1 and counts["other"] == 1 and counts["None"] == 1
    natives = {row["stop_reason"]: row["native"] for row in report["stop_reasons"]["vllm"]["not_end_token"]}
    assert natives["other"]["finish_reason"] == "abort"
    assert report["gates"]["no_max_new_tokens_window_in_vllm"] is False and report["gates"]["stop_reason_available_in_vllm"] is False


def test_calls_missing_on_one_side_fail_the_first_gate(docs):
    documents, _ = docs
    other = vllm_like(documents)
    del other["beta"]
    report = compare_backends.compare(documents, other)
    assert report["calls"]["only_tf"] == ["beta"] and report["gates"]["all_calls_in_both"] is False and report["decodes"]["only_tf"] > 0


def test_peak_gpu_memory_is_read_from_the_sampled_csv_and_speed_from_the_wall_file(docs, tmp_path):
    documents, _ = docs
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("timestamp, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
                   "2026/11/16 10:00:00.1, GPU, 3000 MiB, 24192 MiB, 10 %\n2026/11/16 10:00:05.1, GPU, 20500 MiB, 24192 MiB, 99 %\n", encoding="utf-8")
    run_dir = tmp_path / "rundir"
    run_dir.mkdir()
    (run_dir / "wall.json").write_text(json.dumps({"wall_seconds": 321, "exit_code": 0}), encoding="utf-8")
    report = compare_backends.compare(documents, vllm_like(documents), tf_run=run_dir, tf_gpu=gpu)
    tf = report["speed_memory"]["tf"]
    assert tf["gpu_memory"]["device_reported_peak_mib"] == 20500 and tf["gpu_memory"]["peak_mib"] is None and tf["gpu_memory"]["scope"] == "unavailable"          # a file with no scope record is not a slice measurement
    assert "NOT verified" in tf["gpu_memory"]["label"] and tf["invocation_wall_s"] == 321
    assert report["speed_memory"]["vllm"]["gpu_memory"]["device_reported_peak_mib"] is None and tf["decode_seconds_sum"] >= 0


def test_the_command_never_overwrites_a_report(docs, tmp_path):
    _, folder = docs
    out = tmp_path / "r.json"
    assert compare_backends.main(["--tf", str(folder), "--vllm", str(folder), "--out", str(out), "--report-only"]) == 1       # vllm telemetry is absent in these replay documents
    with pytest.raises(SystemExit, match="already exists"):
        compare_backends.main(["--tf", str(folder), "--vllm", str(folder), "--out", str(out), "--report-only"])


# ---- Stage 3 follow-up: independent evidence (Codex finding 1) --------------------------------------------------------------------------------------------------------------

@pytest.fixture
def two_runs(tmp_path):
    """Two faithful runs of the same two recordings (state and documents each), with vLLM-style telemetry on the second."""
    out = {}
    for label in ("tf", "vllm"):
        run(two_jobs(), tmp_path, base=label)
        folder = tmp_path / label / "out" / "transcripts"
        out[label] = {json.loads(p.read_text(encoding="utf-8"))["jobName"]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}
        out[label + "_state"], out[label + "_folder"] = tmp_path / label / "state", folder
    out["vllm"] = vllm_like(out["vllm"])
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "alpha", "input": "/x/alpha.wav"}, {"job_name": "beta", "input": "/x/beta.wav"}]}), encoding="utf-8")
    out["jobs"], out["tmp"] = jobs, tmp_path
    return out


def verify(runs, tf=None, vllm=None, **kwargs):
    kwargs.setdefault("expected_jobs", ["alpha", "beta"])
    return compare_backends.compare(tf if tf is not None else runs["tf"], vllm if vllm is not None else runs["vllm"], tf_state=runs["tf_state"], vllm_state=runs["vllm_state"], **kwargs)


def test_two_faithful_runs_with_their_evidence_are_accepted(two_runs):
    report = verify(two_runs)
    assert report["verdict"] == "ACCEPTED" and report["passed"] and report["independent_problems"] == {}


def test_both_sides_missing_the_same_recording_no_longer_leaves_the_gates_green(two_runs):
    """Codex finding 1: with both sides omitting the same pilot recording every gate used to stay true."""
    tf, vllm = copy.deepcopy(two_runs["tf"]), copy.deepcopy(two_runs["vllm"])
    del tf["beta"], vllm["beta"]
    report = verify(two_runs, tf, vllm)
    assert report["gates"]["all_calls_in_both"] is True                                           # the old gate is still blind to it ...
    assert not report["passed"] and report["gates"]["tf_membership_matches_the_job_manifest"] is False and report["gates"]["vllm_membership_matches_the_job_manifest"] is False


def test_a_window_missing_on_one_side_fails_that_side(two_runs):
    vllm = copy.deepcopy(two_runs["vllm"])
    del vllm["alpha"]["meralion"]["trace"]["windows"][3]
    report = verify(two_runs, vllm=vllm)
    assert not report["passed"] and report["gates"]["vllm_first_pass_windows_are_complete"] is False and report["gates"]["tf_first_pass_windows_are_complete"] is True


def test_a_retry_deleted_from_both_the_selected_count_and_the_decoded_list_is_caught_per_backend(two_runs):
    tf = copy.deepcopy(two_runs["tf"])
    tf["beta"]["meralion"]["retries"]["decodes"] = []
    tf["beta"]["meralion"]["retries"]["selected"] = 0
    report = verify(two_runs, tf=tf)
    assert not report["passed"] and report["gates"]["tf_persisted_plan_matches_the_documents"] is False and report["gates"]["vllm_persisted_plan_matches_the_documents"] is True


def test_different_retry_plans_across_backends_are_not_a_failure(tmp_path):
    """The backends may legitimately plan different retries (different first-pass text): each is checked against its OWN persisted plan, never against the other's."""
    import synthetic
    out = {}
    for label, world in (("tf", two_jobs()), ("vllm", synthetic.World({"alpha": dict(break_at=4), "beta": dict(break_at=2, replace_first="half a percent")}))):
        run(world, tmp_path, base=label)
        folder = tmp_path / label / "out" / "transcripts"
        out[label] = {json.loads(p.read_text(encoding="utf-8"))["jobName"]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}
    out["vllm"] = vllm_like(out["vllm"])
    tf_windows = [tuple(d["window"]) for d in out["tf"]["alpha"]["meralion"]["retries"]["decodes"]]
    vllm_windows = [tuple(d["window"]) for d in out["vllm"]["alpha"]["meralion"]["retries"]["decodes"]]
    assert tf_windows != vllm_windows                                                           # the two backends really planned different retries
    report = compare_backends.compare(out["tf"], out["vllm"], expected_jobs=["alpha", "beta"], tf_state=tmp_path / "tf" / "state", vllm_state=tmp_path / "vllm" / "state")
    assert report["independent_problems"] == {} and report["passed"] and report["verdict"] == "ACCEPTED"


def test_an_aborted_or_unexplained_stop_on_vllm_fails_the_gate(two_runs):
    vllm = copy.deepcopy(two_runs["vllm"])
    window = vllm["alpha"]["meralion"]["trace"]["windows"][0]
    window.update(stop_reason="other", native={"finish_reason": "abort", "stop_reason": None})
    report = verify(two_runs, vllm=vllm)
    assert not report["passed"] and report["gates"]["no_aborted_or_unexplained_stop"] is False


def test_a_duplicate_document_on_either_side_fails(two_runs):
    folder = two_runs["vllm_folder"]
    (folder / "beta-copy.json").write_text((folder / "beta.json").read_text(encoding="utf-8"), encoding="utf-8")
    docs, problems = acceptance.scan_documents(folder)
    assert any("duplicate" in p for p in problems)
    assert verify(two_runs, vllm=vllm_like(docs), vllm_problems=problems)["gates"]["vllm_documents_are_unique"] is False


def test_the_command_gates_on_the_verdict_and_needs_the_evidence(two_runs):
    out = two_runs["tmp"] / "report.json"
    base = ["--tf", str(two_runs["tf_folder"]), "--vllm", str(two_runs["vllm_folder"]), "--out", str(out)]
    assert compare_backends.main(base) == 2                                                    # no --jobs / --state: no verdict is produced
    full = base + ["--jobs", str(two_runs["jobs"]), "--state-tf", str(two_runs["tf_state"]), "--state-vllm", str(two_runs["vllm_state"])]
    assert compare_backends.main(full) == 1                                                    # the replay documents carry no vLLM telemetry, so the stop-reason gates fail
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "REJECTED"
