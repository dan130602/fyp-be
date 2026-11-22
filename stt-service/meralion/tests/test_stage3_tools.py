"""Stage 3 local tools: make_jobs and live_parity. Synthetic data only: no git-ignored artefact is read."""
import copy
import sys
import types
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import synthetic
from harness import run, two_jobs

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_jobs = load_tool("make_jobs")
live_parity = load_tool("live_parity")
sys.path.insert(0, str(TOOLS))
import acceptance  # noqa: E402


# ---- make_jobs -------------------------------------------------------------------------------------------------------------------------------------------------------------

def batch_folder(tmp_path, ids=("CALL_001", "CALL_002", "CALL_003"), tamper=None):
    batch = tmp_path / "batch"
    (batch / "audio").mkdir(parents=True)
    rows = []
    for call in ids:
        data = f"audio of {call}".encode()
        (batch / "audio" / f"{call}.wav").write_bytes(data)
        rows.append({"call_id": call, "audio": f"audio/{call}.wav", "audio_sha256": "0" * 64 if call == tamper else hashlib.sha256(data).hexdigest()})
    (batch / "calls.json").write_text(json.dumps({"calls": rows}), encoding="utf-8")
    return batch


def pilot_list(ids_seconds):
    return {"per_call": {call: {"audio_seconds": seconds} for call, seconds in ids_seconds.items()}}


def test_the_jobs_follow_the_pilot_order_and_the_shortest_call_is_the_c4_reference(tmp_path):
    batch = batch_folder(tmp_path)
    jobs, shortest, seconds = make_jobs.build_jobs(batch, pilot_list({"CALL_003": 300.0, "CALL_001": 180.5, "CALL_002": 900.0}))
    assert [job["job_name"] for job in jobs] == ["CALL_003", "CALL_001", "CALL_002"] and shortest == "CALL_001"
    assert all(Path(job["input"]).is_absolute() and Path(job["input"]).is_file() for job in jobs)


def test_a_shortest_tie_is_broken_by_call_id(tmp_path):
    _, shortest, _ = make_jobs.build_jobs(batch_folder(tmp_path), pilot_list({"CALL_002": 100.0, "CALL_001": 100.0}))
    assert shortest == "CALL_001"


def test_an_audio_file_that_does_not_match_the_manifest_hash_is_refused(tmp_path):
    with pytest.raises(make_jobs.JobsError, match="sha256"):
        make_jobs.build_jobs(batch_folder(tmp_path, tamper="CALL_002"), pilot_list({"CALL_001": 1.0, "CALL_002": 2.0}))


def test_an_unknown_call_and_a_missing_file_are_refused(tmp_path):
    batch = batch_folder(tmp_path)
    with pytest.raises(make_jobs.JobsError, match="not in"):
        make_jobs.build_jobs(batch, pilot_list({"CALL_999": 1.0}))
    (batch / "audio" / "CALL_001.wav").unlink()
    with pytest.raises(make_jobs.JobsError, match="missing audio"):
        make_jobs.build_jobs(batch, pilot_list({"CALL_001": 1.0}))


def test_the_command_writes_both_jobs_files(tmp_path, capsys):
    batch = batch_folder(tmp_path)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps(pilot_list({"CALL_001": 7200.0, "CALL_002": 3600.0})), encoding="utf-8")
    assert make_jobs.main(["--batch", str(batch), "--pilot", str(pilot), "--out", str(tmp_path / "out")]) == 0
    full = json.loads((tmp_path / "out" / "jobs-pilot22.json").read_text(encoding="utf-8"))["jobs"]
    one = json.loads((tmp_path / "out" / "jobs-c4.json").read_text(encoding="utf-8"))["jobs"]
    assert [j["job_name"] for j in full] == ["CALL_001", "CALL_002"] and [j["job_name"] for j in one] == ["CALL_002"]
    assert json.loads(capsys.readouterr().out)["total_audio_hours"] == 3.0
    assert make_jobs.main(["--batch", str(batch), "--pilot", str(pilot), "--out", str(tmp_path / "out2"), "--no-verify"]) == 0
    assert make_jobs.main(["--batch", str(tmp_path / "nope"), "--pilot", str(pilot), "--out", str(tmp_path / "out3")]) == 3


def test_all_calls_mode_keeps_the_manifest_order_checks_every_hash_and_writes_one_file(tmp_path):
    batch = batch_folder(tmp_path, ids=("CALL_009", "CALL_001", "CALL_005"))
    jobs = make_jobs.build_all_jobs(batch)
    assert [j["job_name"] for j in jobs] == ["CALL_009", "CALL_001", "CALL_005"] and all(Path(j["input"]).is_absolute() for j in jobs)
    with pytest.raises(make_jobs.JobsError, match="sha256"):
        make_jobs.build_all_jobs(batch_folder(tmp_path / "b", tamper="CALL_002"))
    out = tmp_path / "out"
    assert make_jobs.main(["--batch", str(batch), "--all-calls", "--out", str(out)]) == 0
    assert json.loads((out / "jobs-dev175.json").read_text(encoding="utf-8"))["jobs"][0]["job_name"] == "CALL_009" and sorted(p.name for p in out.iterdir()) == ["jobs-dev175.json"]
    assert make_jobs.main(["--batch", str(batch), "--out", str(out)]) == 3                      # neither --pilot nor --all-calls
    assert make_jobs.main(["--batch", str(batch), "--all-calls", "--pilot", str(out / "x.json"), "--out", str(out)]) == 3


def test_the_audit_compares_the_manifest_hashes_with_what_a_scored_run_recorded(tmp_path):
    batch = batch_folder(tmp_path)
    manifest = json.loads((batch / "calls.json").read_text(encoding="utf-8"))["calls"]
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for row in manifest:
        (predictions / f"{row['call_id']}.json").write_text(json.dumps({"call_id": row["call_id"], "settings": {"audio_sha256": row["audio_sha256"]}}), encoding="utf-8")
    assert make_jobs.audit_predictions(batch, predictions) == []
    assert make_jobs.main(["--batch", str(batch), "--audit-predictions", str(predictions)]) == 0
    (predictions / "CALL_002.json").write_text(json.dumps({"call_id": "CALL_002", "settings": {"audio_sha256": "1" * 64}}), encoding="utf-8")
    assert any("CALL_002" in p for p in make_jobs.audit_predictions(batch, predictions)) and make_jobs.main(["--batch", str(batch), "--audit-predictions", str(predictions)]) == 3
    (predictions / "CALL_002.json").unlink()
    assert any("differ" in p for p in make_jobs.audit_predictions(batch, predictions))


# ---- live_parity -----------------------------------------------------------------------------------------------------------------------------------------------------------

def saved_artefacts(tmp_path, docs):
    """Write the saved development artefacts a faithful live run would equal: windows, retries (keyed through a plan), the frozen plan, and both transcript folders."""
    runs = tmp_path / "saved"
    folders = {name: runs / "fg175-all" / name for name in ("overlap", "final_m2ax_seq_ap_lmonly_m5", "final_m2ax_seq_ap_lmonly_m5_fee2", "final_m2ax_seq_ap_lmonly_m5_fee2_amt1")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    (runs / "fg175-lm-retries-all").mkdir()
    selected = []
    for call, document in docs.items():
        block = document["meralion"]
        (folders["overlap"] / f"{call}.json").write_text(json.dumps({"call_id": call, "windows": [{k: w[k] for k in ("start_sample", "end_sample", "text")} for w in block["trace"]["windows"]]}), encoding="utf-8")
        for decode in block["retries"]["decodes"]:
            selected.append({"retry_id": decode["retry_id"], "call_id": call, "window": decode["window"]})
            (runs / "fg175-lm-retries-all" / f"{decode['retry_id']}.json").write_text(json.dumps({"text": decode["result"]["text"]}), encoding="utf-8")
        (folders["final_m2ax_seq_ap_lmonly_m5"] / f"{call}.json").write_text(json.dumps({"transcript": block["transcript_raw"]}), encoding="utf-8")
        for name in ("final_m2ax_seq_ap_lmonly_m5_fee2", "final_m2ax_seq_ap_lmonly_m5_fee2_amt1"):
            (folders[name] / f"{call}.json").write_text(json.dumps({"transcript": block["transcript_final"]}), encoding="utf-8")
    (runs / "fg175-lm-plan-all.json").write_text(json.dumps({"selected": selected}), encoding="utf-8")
    (runs / "fg175-lm-plan-lmonly.json").write_text(json.dumps({"selected": [{"call_id": g["call_id"], "window": g["window"]} for g in selected]}), encoding="utf-8")
    return runs


@pytest.fixture
def world(tmp_path):
    """A faithful synthetic run: documents, its persisted state, the job manifest and matching saved artefacts."""
    run(two_jobs(), tmp_path)
    folder = tmp_path / "run" / "out" / "transcripts"
    docs = {json.loads(p.read_text(encoding="utf-8"))["jobName"]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "alpha", "input": "/x/alpha.wav"}, {"job_name": "beta", "input": "/x/beta.wav"}]}), encoding="utf-8")
    return types.SimpleNamespace(docs=docs, runs=saved_artefacts(tmp_path, docs), folder=folder, state=tmp_path / "run" / "state", jobs=jobs, names=["alpha", "beta"], tmp=tmp_path)


def verify(world, docs=None, **kwargs):
    kwargs.setdefault("expected_jobs", world.names)
    kwargs.setdefault("state_root", world.state)
    return live_parity.compare(docs if docs is not None else world.docs, world.runs, **kwargs)


def test_a_faithful_run_with_its_evidence_is_accepted(world):
    report = verify(world)
    assert report["verdict"] == "ACCEPTED" and report["passed"] and report["calls"] == 2 and report["independent_problems"] == {}
    assert report["windows"]["total"] == 14 and report["retries"]["total"] == 2 and report["plan"]["equal"] and report["transcripts"]["comparable"] is True
    assert report["timing"]["decode_seconds_per_window"]["count"] == 16


def test_without_the_jobs_and_the_state_there_is_never_an_acceptance_verdict(world):
    report = live_parity.compare(world.docs, world.runs)
    assert report["passed"] and report["acceptance_evidence"] is False and report["verdict"].startswith("NOT AN ACCEPTANCE VERDICT")
    assert live_parity.compare(world.docs, world.runs, expected_jobs=world.names)["verdict"].startswith("NOT AN ACCEPTANCE VERDICT")
    assert live_parity.compare(world.docs, world.runs, state_root=world.state)["verdict"].startswith("NOT AN ACCEPTANCE VERDICT")


def test_one_missing_first_pass_window_fails_even_though_everything_present_matches(world):
    """Codex finding 1: 918 of 919 windows used to pass because only the windows that were present were compared."""
    docs = copy.deepcopy(world.docs)
    del docs["alpha"]["meralion"]["trace"]["windows"][3]
    report = verify(world, docs)
    assert report["windows"]["differing"] == 0                                                 # nothing present differs from the saved run ...
    assert not report["passed"] and report["checks"]["first_pass_windows_are_complete"] is False and report["verdict"] == "REJECTED"   # ... but the first pass is incomplete
    assert report["checks"]["persisted_plan_matches_the_documents"] is False


def test_a_recording_missing_from_the_outputs_or_not_in_the_manifest_fails(world):
    docs = copy.deepcopy(world.docs)
    del docs["beta"]
    assert verify(world, docs)["checks"]["membership_matches_the_job_manifest"] is False
    assert verify(world, expected_jobs=["alpha"])["checks"]["membership_matches_the_job_manifest"] is False        # beta is unexpected
    assert verify(world, expected_jobs=["alpha", "beta", "gamma"])["checks"]["membership_matches_the_job_manifest"] is False


def test_a_duplicate_document_is_found_before_any_dictionary_is_built(world):
    (world.folder / "beta-copy.json").write_text((world.folder / "beta.json").read_text(encoding="utf-8"), encoding="utf-8")
    docs, problems = acceptance.scan_documents(world.folder)
    assert len(docs) == 2 and any("duplicate output for 'beta'" in p for p in problems) and any("does not match jobName" in p for p in problems)
    assert verify(world, docs, document_problems=problems)["checks"]["documents_are_unique"] is False


def test_a_retry_deleted_from_both_the_selected_count_and_the_decoded_list_is_still_caught_by_the_persisted_plan(world):
    docs = copy.deepcopy(world.docs)
    docs["beta"]["meralion"]["retries"]["decodes"] = []
    docs["beta"]["meralion"]["retries"]["selected"] = 0
    report = verify(world, docs)
    assert not report["passed"] and report["checks"]["persisted_plan_matches_the_documents"] is False
    assert any("beta" in p and "persisted plan" in p for p in report["independent_problems"]["persisted_plan_matches_the_documents"])


def test_retry_ids_and_bounds_are_verified_against_the_backends_own_plan(world):
    for change in ("retry_id", "window"):
        docs = copy.deepcopy(world.docs)
        decode = docs["alpha"]["meralion"]["retries"]["decodes"][0]
        if change == "retry_id":
            decode["retry_id"] = "alpha_r099"
        else:
            decode["window"] = [1, 2]
        assert verify(world, docs)["checks"]["persisted_plan_matches_the_documents"] is False, change


def test_a_persisted_plan_that_the_selection_does_not_reproduce_is_caught(world):
    plan_path = world.state / "inv-1" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["selected"] = plan["selected"][:1]
    plan_path.write_bytes(json.dumps(plan).encode("utf-8"))
    report = verify(world)
    assert report["checks"]["plan_recomputes_from_the_persisted_first_pass"] is False and not report["passed"]


def test_the_state_must_exist(world, tmp_path):
    report = verify(world, state_root=tmp_path / "no-state")
    assert not report["passed"] and report["checks"]["persisted_plan_matches_the_documents"] is False


def test_a_differing_first_pass_text_against_the_saved_run_fails_unless_the_run_is_on_different_hardware(world):
    saved = world.runs / "fg175-all" / "overlap" / "alpha.json"
    data = json.loads(saved.read_text(encoding="utf-8"))
    data["windows"][2]["text"] += " other hardware"
    saved.write_text(json.dumps(data), encoding="utf-8")
    strict = verify(world)
    assert not strict["passed"] and strict["windows"]["first_pass_text_changes"] == 1 and strict["checks"]["first_pass_texts_equal_the_saved_run"] is False
    cross = verify(world, cross_hardware=True)
    assert cross["passed"] and cross["windows"]["first_pass_text_changes"] == 1 and cross["retries"]["retry_text_changes"] == 0 and cross["verdict"] == "ACCEPTED"   # reported separately, not hidden


def test_cross_hardware_never_waives_the_independent_checks(world):
    docs = copy.deepcopy(world.docs)
    del docs["alpha"]["meralion"]["trace"]["windows"][3]
    assert not verify(world, docs, cross_hardware=True)["passed"]


def test_a_retry_text_that_differs_is_counted_separately_from_first_pass_texts(world):
    saved = next((world.runs / "fg175-lm-retries-all").glob("*.json"))
    saved.write_text(json.dumps({"text": "different retry text"}), encoding="utf-8")
    report = verify(world)
    assert not report["passed"] and report["retries"]["retry_text_changes"] == 1 and report["windows"]["first_pass_text_changes"] == 0
    assert verify(world, cross_hardware=True)["passed"]


def test_a_plan_that_differs_from_the_frozen_one_fails_and_comparability_does_not_excuse_it(world):
    plan_path = world.runs / "fg175-lm-plan-lmonly.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["selected"].append({"call_id": "alpha", "window": [0, 480000]})
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    report = verify(world)
    assert report["transcripts"]["comparable"] is False                                         # the transcript comparison is informational ...
    assert report["checks"]["plan_equals_the_frozen_plan_or_the_budget_bound"] is False and not report["passed"]     # ... but the plan difference itself fails


def test_when_the_budget_bound_the_plan_difference_is_expected_but_the_recomputation_must_still_hold(world):
    docs = copy.deepcopy(world.docs)
    for document in docs.values():
        document["meralion"]["invocation"]["run_budget_bound"] = True
    report = verify(world, docs)
    assert report["transcripts"]["comparable"] is False and report["checks"]["plan_equals_the_frozen_plan_or_the_budget_bound"] is True


def test_a_transcript_that_differs_fails_when_the_comparison_is_comparable(world):
    docs = copy.deepcopy(world.docs)
    docs["beta"]["results"]["transcripts"][0]["transcript"] += " x"
    report = verify(world, docs)
    assert not report["passed"] and report["transcripts"]["differing"] == ["beta"]


def test_the_formatted_target_follows_the_documents_own_formatter(world):
    docs = copy.deepcopy(world.docs)
    for document in docs.values():
        document["meralion"]["formatter"]["name"] = "percentages-v2"
    (world.runs / "fg175-all" / "final_m2ax_seq_ap_lmonly_m5_fee2" / "beta.json").write_text(json.dumps({"transcript": "something else"}), encoding="utf-8")
    assert verify(world, docs)["transcripts"]["differing"] == ["beta"]


def renamed(docs):
    """The same documents with every retry id changed (as a standalone run numbers them differently)."""
    copied = copy.deepcopy(docs)
    for document in copied.values():
        for decode in document["meralion"]["retries"]["decodes"]:
            decode["retry_id"] = decode["retry_id"].replace("_r000", "_r025")
    return copied


def test_a_standalone_run_equals_the_same_recording_in_the_reference_run_regardless_of_retry_ids(world):
    assert acceptance.compare_same_gpu(renamed(world.docs), world.docs) == []


@pytest.mark.parametrize("what", ["window text", "retry bounds", "retry text", "trigger", "transcript"])
def test_any_real_difference_between_standalone_and_reference_is_a_problem(world, what):
    alone = renamed(world.docs)
    block = alone["alpha"]["meralion"]
    if what == "window text":
        block["trace"]["windows"][1]["text"] += " x"
    elif what == "retry bounds":
        block["retries"]["decodes"][0]["window"] = [1, 2]
    elif what == "retry text":
        block["retries"]["decodes"][0]["result"]["text"] += " x"
    elif what == "trigger":
        block["retries"]["outcomes"][0]["triggers"] = [[9, ["other"], 0.0]]
    else:
        block["transcript_raw"] += " x"
    assert acceptance.compare_same_gpu(alone, world.docs)


def test_the_command_needs_the_evidence_for_a_verdict_and_never_overwrites_a_report(world, capsys):
    base = ["--docs", str(world.folder), "--runs", str(world.runs)]
    assert live_parity.main(base) == 2
    report_path = world.tmp / "reports" / "r.json"
    full = base + ["--jobs", str(world.jobs), "--state", str(world.state)]
    assert live_parity.main(full + ["--report", str(report_path)]) == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["verdict"] == "ACCEPTED"
    with pytest.raises(SystemExit, match="already exists"):
        live_parity.main(full + ["--report", str(report_path)])
    capsys.readouterr()
    assert live_parity.main(base + ["--report-only"]) == 0 and "NOT AN ACCEPTANCE VERDICT" in capsys.readouterr().out


def test_the_command_fails_on_a_duplicate_document(world):
    (world.folder / "beta-copy.json").write_text((world.folder / "beta.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert live_parity.main(["--docs", str(world.folder), "--runs", str(world.runs), "--jobs", str(world.jobs), "--state", str(world.state)]) == 1


def test_percentiles_are_nearest_rank():
    assert live_parity.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 5 and live_parity.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert live_parity.percentile([], 0.5) is None and live_parity.percentile([7], 0.95) == 7
