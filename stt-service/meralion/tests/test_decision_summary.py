"""tools/decision_summary.py: exactly the two declared candidate invocations, one verdict (Stage 3 follow-up E, section 8.5). Synthetic decisions only."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import gpu_support

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("decision_summary", TOOLS / "decision_summary.py")
summary_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_tool)

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SETTINGS = "vllm-rp105-sched0-v1"
R1, R2, BASE = "vllm-l40s-e2-cand-pilot22-r1", "vllm-l40s-e2-cand-pilot22-r2", "tf-l40s-e2-pilot22-1"
COUNTS = {"raw_wer": {"candidate_errors": 11, "baseline_errors": 11, "reference_tokens": 1000}, "raw_entity_wer": {"candidate_errors": 3, "baseline_errors": 3, "reference_tokens": 100},
          "final_wer": {"candidate_errors": 10, "baseline_errors": 10, "reference_tokens": 1000}, "final_entity_wer": {"candidate_errors": 2, "baseline_errors": 2, "reference_tokens": 100}}


def make_decision(invocation_id, verdict="ACCEPTED", **over):
    decision = {"verdict": verdict, "exit_code": {"ACCEPTED": 0, "REJECTED": 1}.get(verdict, 2), "evidence_problems": [], "failed_items": [] if verdict == "ACCEPTED" else ["quality:x"],
                "candidate_settings": SETTINGS, "identity": {"candidate": {"invocation_ids": [invocation_id], "settings_ids": [SETTINGS], "worker_commits": [COMMIT], "documents_sha256": invocation_id},
                                                             "baseline": {"invocation_ids": [BASE], "settings_ids": ["stock-transformers-v1"], "worker_commits": [COMMIT], "documents_sha256": "same-baseline"}},
                "quality_details": {"integer_counts": copy.deepcopy(COUNTS), "new_large_omissions": {"raw": 0, "final": 0}, "new_loops": 0, "candidate_stop_reasons": {"end_token": 100}}}
    decision.update(over)
    return decision


class Setup:
    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.runs = self.tmp / "runs"
        for invocation_id, wall in ((R1, 150), (R2, 160), (BASE, 1600)):
            folder = self.runs / f"worker-v0-{invocation_id}"
            folder.mkdir(parents=True)
            (folder / "wall.json").write_text(json.dumps({"wall_seconds": wall, "exit_code": 0}), encoding="utf-8")
            gpu_support.write_scope(folder, used=40000)
        self.manifest_data = {"schema": "e2-manifest-v1", "slurm_job_id": "999", "worker_commit": COMMIT, "gpu_uuid": "GPU-1", "promoted_settings_id": SETTINGS,
                              "promoted_config": "config.json", "baseline_invocation_ids": {"pilot": BASE, "c4": "tf-c4"}, "candidate_invocation_ids": [R1, R2],
                              "candidate_c4_invocation_id": "vllm-c4", "control_invocation_id": "vllm-control"}
        self.manifest = self.write("manifest.json", self.manifest_data)
        self.decisions = [self.write("d1.json", make_decision(R1)), self.write("d2.json", make_decision(R2))]

    def write(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def summarise(self, decisions=None, manifest=None):
        decisions = self.decisions if decisions is None else decisions
        manifest = json.loads(self.manifest.read_text(encoding="utf-8")) if manifest is None else manifest
        return summary_tool.summarise(manifest, [(str(p), json.loads(Path(p).read_text(encoding="utf-8"))) for p in decisions], self.runs)


@pytest.fixture
def setup(tmp_path):
    return Setup(tmp_path)


def not_a_verdict(result, fragment):
    assert result["verdict"] == "NOT AN ACCEPTANCE VERDICT" and result["exit_code"] == 2
    assert any(fragment in problem for problem in result["problems"]), result["problems"]


def test_both_declared_runs_accepted_is_accepted_and_both_runs_are_reported(setup):
    result = setup.summarise()
    assert result["verdict"] == "ACCEPTED" and result["exit_code"] == 0 and result["problems"] == []
    assert [r["invocation_id"] for r in result["runs"]] == [R1, R2] and result["runs"][0]["wall_seconds"] == 150 and result["runs"][1]["gpu_peak_mib"] == 40000
    assert result["ranges_over_repetitions"]["raw_wer"] == {"min": 11, "max": 11} and result["ranges_over_repetitions"]["wall_seconds"] == {"min": 150, "max": 160}
    assert result["runs"][0]["raw_wer"]["candidate_percent"] == 1.1                                        # the percentage is displayed, never compared


def test_one_rejected_repetition_rejects_the_set_and_the_worst_repetition_governs(setup):
    setup.decisions[1] = setup.write("d2.json", make_decision(R2, "REJECTED"))
    result = setup.summarise()
    assert result["verdict"] == "REJECTED" and result["exit_code"] == 1 and result["problems"] == []


def test_one_not_an_acceptance_verdict_makes_the_whole_set_one(setup):
    setup.decisions[0] = setup.write("d1.json", make_decision(R1, "NOT AN ACCEPTANCE VERDICT", evidence_problems=["a report is stale"]))
    not_a_verdict(setup.summarise(), "is itself not an acceptance verdict")


def test_a_missing_decision_is_not_a_verdict(setup):
    not_a_verdict(setup.summarise(decisions=setup.decisions[:1]), "exactly two decisions")
    not_a_verdict(setup.summarise(decisions=setup.decisions[:1]), f"missing decision for the declared candidate invocation(s) {[R2]}")


def test_a_third_unexpected_decision_is_not_a_verdict(setup):
    third = setup.write("d3.json", make_decision("vllm-l40s-e2-cand-pilot22-r3"))
    result = setup.summarise(decisions=[*setup.decisions, third])
    not_a_verdict(result, "exactly two decisions")
    not_a_verdict(result, "unexpected decision for undeclared candidate invocation(s)")


def test_the_same_decision_given_twice_is_a_duplicate(setup):
    result = setup.summarise(decisions=[setup.decisions[0], setup.decisions[0]])
    not_a_verdict(result, f"duplicate decision for the candidate invocation {R1}")
    not_a_verdict(result, f"missing decision for the declared candidate invocation(s) {[R2]}")


def test_two_different_files_for_one_invocation_are_a_duplicate(setup):
    other = setup.write("d2b.json", make_decision(R1))
    not_a_verdict(setup.summarise(decisions=[setup.decisions[0], other]), "duplicate decision")


def test_a_decision_for_an_undeclared_invocation_in_place_of_a_declared_one_is_not_a_verdict(setup):
    swapped = setup.write("d2.json", make_decision("vllm-l40s-e2-cand-pilot22-r9"))
    result = setup.summarise(decisions=[setup.decisions[0], swapped])
    not_a_verdict(result, "unexpected decision")
    not_a_verdict(result, "missing decision")


def test_a_decision_for_another_settings_id_or_commit_or_baseline_is_not_a_verdict(setup):
    other = make_decision(R2, candidate_settings="vllm-rp105-v1")
    other["identity"]["candidate"]["settings_ids"] = ["vllm-rp105-v1"]
    not_a_verdict(setup.summarise(decisions=[setup.decisions[0], setup.write("x.json", other)]), "the manifest promotes")
    other = make_decision(R2)
    other["identity"]["candidate"]["worker_commits"] = ["e" * 40]
    not_a_verdict(setup.summarise(decisions=[setup.decisions[0], setup.write("y.json", other)]), "differ from the declared")
    other = make_decision(R2)
    other["identity"]["baseline"]["documents_sha256"] = "another-baseline"
    not_a_verdict(setup.summarise(decisions=[setup.decisions[0], setup.write("z.json", other)]), "not made against the same baseline documents")
    other = make_decision(R2)
    other["identity"]["baseline"]["invocation_ids"] = ["tf-other"]
    not_a_verdict(setup.summarise(decisions=[setup.decisions[0], setup.write("w.json", other)]), "its baseline is")


def test_a_declared_run_folder_without_its_files_is_not_a_verdict(setup):
    (setup.runs / f"worker-v0-{R2}" / "gpu.csv").unlink()
    not_a_verdict(setup.summarise(), f"the declared run {R2} has no gpu.csv")
    (setup.runs / f"worker-v0-{BASE}" / "wall.json").unlink()
    not_a_verdict(setup.summarise(), f"the declared run {BASE} has no wall.json")


def test_a_manifest_that_does_not_declare_exactly_two_candidate_runs_is_not_a_verdict(setup):
    manifest = json.loads(setup.manifest.read_text(encoding="utf-8"))
    manifest["candidate_invocation_ids"] = [R1]
    not_a_verdict(setup.summarise(manifest=manifest), "exactly two distinct candidate invocations")
    manifest["candidate_invocation_ids"] = [R1, R1]
    not_a_verdict(setup.summarise(manifest=manifest), "exactly two distinct candidate invocations")
    manifest = json.loads(setup.manifest.read_text(encoding="utf-8"))
    manifest["schema"] = "other"
    not_a_verdict(setup.summarise(manifest=manifest), "is not a e2-manifest-v1")


def test_a_decision_that_names_no_or_several_candidate_invocations_is_not_a_verdict(setup):
    broken = make_decision(R2)
    broken["identity"]["candidate"]["invocation_ids"] = [R1, R2]
    not_a_verdict(setup.summarise(decisions=[setup.decisions[0], setup.write("b.json", broken)]), "does not name exactly one candidate invocation")


def test_the_command_writes_the_summary_once_and_returns_the_exit_code(setup, tmp_path):
    out = tmp_path / "summary.json"
    argv = ["--manifest", str(setup.manifest), "--decisions", *map(str, setup.decisions), "--runs-root", str(setup.runs), "--out", str(out)]
    assert summary_tool.main(argv) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["verdict"] == "ACCEPTED" and len(written["inputs"]) == 3 and all(len(i["sha256"]) == 64 for i in written["inputs"])
    with pytest.raises(SystemExit, match="already exists"):
        summary_tool.main(argv)
    setup.decisions[1] = setup.write("d2.json", make_decision(R2, "REJECTED"))
    assert summary_tool.main(["--manifest", str(setup.manifest), "--decisions", *map(str, setup.decisions), "--runs-root", str(setup.runs), "--out", str(tmp_path / "second.json")]) == 1


# ---- frozen-historical baseline (full development validation, manifest schema full175-manifest-v1) ----------------------------------------------------------------------------------

F1, F2, REPLAY = "vllm-l40s-full175-r1", "vllm-l40s-full175-r2", "replay-175"
FP, DIGEST, SRC, DOCS = "a" * 64, "b" * 64, "c" * 64, "d" * 64


def historical_decision(invocation_id, verdict="ACCEPTED", **over):
    decision = make_decision(invocation_id, verdict)
    decision["identity"]["candidate"].update(config_fingerprints=[FP])
    decision["identity"]["baseline"].update(invocation_ids=[REPLAY], worker_commits=["replay"], settings_ids=["replay"], documents_sha256=DOCS)
    decision["identity"].update(baseline_kind="frozen-historical", baseline_source_sha256=SRC, candidate_frozen_digests=[DIGEST], candidate_allocation={"slurm_job_id": "999", "gpu_uuid": "GPU-1"})
    decision["not_established"] = ["same-hardware comparison"]
    decision.update(over)
    return decision


class HistoricalSetup(Setup):
    def __init__(self, tmp):
        super().__init__(tmp)
        for invocation_id in (F1, F2):
            folder = self.runs / f"worker-v0-{invocation_id}"
            folder.mkdir(parents=True)
            (folder / "wall.json").write_text(json.dumps({"wall_seconds": 600, "exit_code": 0}), encoding="utf-8")
            gpu_support.write_scope(folder, used=41000)
        self.manifest_data = {"schema": "full175-manifest-v1", "slurm_job_id": "999", "worker_commit": COMMIT, "gpu_uuid": "GPU-1", "promoted_settings_id": SETTINGS, "promoted_config": "config.json",
                              "config_fingerprint": FP, "settings_digest": DIGEST, "candidate_invocation_ids": [F1, F2],
                              "baseline": {"kind": "frozen-historical", "source_sha256": SRC, "documents_sha256": DOCS, "invocation_ids": [REPLAY]}}
        self.manifest = self.write("hmanifest.json", self.manifest_data)
        self.decisions = [self.write("h1.json", historical_decision(F1)), self.write("h2.json", historical_decision(F2))]


@pytest.fixture
def hsetup(tmp_path):
    return HistoricalSetup(tmp_path)


def test_two_historical_decisions_bound_to_the_declared_baseline_configuration_and_allocation_are_accepted_and_carry_what_is_not_established(hsetup):
    result = hsetup.summarise()
    assert result["verdict"] == "ACCEPTED" and result["problems"] == [] and result["baseline_kind"] == "frozen-historical" and result["baseline"] == REPLAY
    assert result["not_established"] == ["same-hardware comparison"] and [r["invocation_id"] for r in result["runs"]] == [F1, F2] and result["runs"][0]["wall_seconds"] == 600


def test_a_rejected_repetition_still_governs_in_the_historical_set(hsetup):
    hsetup.decisions[1] = hsetup.write("h2.json", historical_decision(F2, "REJECTED"))
    assert hsetup.summarise()["verdict"] == "REJECTED"


@pytest.mark.parametrize("path,value,fragment", [
    (("identity", "baseline_kind"), "matched-allocation", "not a decision against the declared frozen-historical baseline source"),
    (("identity", "baseline_source_sha256"), "0" * 64, "not a decision against the declared frozen-historical baseline source"),
    (("identity", "baseline", "documents_sha256"), "0" * 64, "its baseline documents are not the declared ones"),
    (("identity", "baseline", "invocation_ids"), ["replay-other"], "its baseline is"),
    (("identity", "candidate", "config_fingerprints"), ["9" * 64], "configuration fingerprints"),
    (("identity", "candidate_frozen_digests"), ["9" * 64], "settings digests"),
    (("identity", "candidate_allocation"), {"slurm_job_id": "1000", "gpu_uuid": "GPU-1"}, "allocation"),
    (("identity", "candidate_allocation"), {"slurm_job_id": "999", "gpu_uuid": "GPU-2"}, "allocation"),
    (("identity", "candidate", "worker_commits"), ["e" * 40], "differ from the declared")])
def test_a_decision_not_bound_to_the_declared_baseline_configuration_commit_or_allocation_is_not_a_verdict(hsetup, path, value, fragment):
    decision = historical_decision(F2)
    target = decision
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    not_a_verdict(hsetup.summarise(decisions=[hsetup.decisions[0], hsetup.write("x.json", decision)]), fragment)


def test_a_historical_manifest_must_declare_the_baseline_and_the_candidate_configuration(hsetup):
    for key in ("config_fingerprint", "settings_digest"):
        manifest = dict(hsetup.manifest_data)
        manifest.pop(key)
        not_a_verdict(hsetup.summarise(manifest=manifest), "must declare a frozen-historical baseline")
    manifest = json.loads(json.dumps(hsetup.manifest_data))
    manifest["baseline"]["kind"] = "matched-allocation"
    not_a_verdict(hsetup.summarise(manifest=manifest), "must declare a frozen-historical baseline")
    manifest = json.loads(json.dumps(hsetup.manifest_data))
    manifest["baseline"]["invocation_ids"] = [REPLAY, "second"]
    not_a_verdict(hsetup.summarise(manifest=manifest), "must declare a frozen-historical baseline")


def test_a_matched_allocation_decision_cannot_satisfy_a_historical_manifest_and_the_reverse(hsetup, tmp_path):
    setup = Setup(tmp_path / "plain")
    (tmp_path / "plain").mkdir(exist_ok=True)
    not_a_verdict(hsetup.summarise(decisions=[hsetup.decisions[0], hsetup.write("m.json", make_decision(F2))]), "not a decision against the declared frozen-historical baseline source")
    manifest = json.loads(setup.manifest.read_text(encoding="utf-8"))
    not_a_verdict(setup.summarise(decisions=[setup.write("hx.json", historical_decision(R1)), setup.write("hy.json", historical_decision(R2))], manifest=manifest), "differ from the declared")


def test_the_historical_run_folders_need_their_files_but_the_replay_baseline_needs_no_run_folder(hsetup):
    assert hsetup.summarise()["verdict"] == "ACCEPTED"
    (hsetup.runs / f"worker-v0-{F1}" / "wall.json").unlink()
    not_a_verdict(hsetup.summarise(), f"the declared run {F1} has no wall.json")


# ---- REV 4.1 A3: the GPU-memory object, numeric ranges over numeric values only -------------------------------------------------------------------------------------------------------

def test_verified_unverified_and_unavailable_gpu_measurements_are_reported_apart_and_ranges_stay_numeric(setup):
    folder1, folder2 = setup.runs / f"worker-v0-{R1}", setup.runs / f"worker-v0-{R2}"
    result = setup.summarise()
    assert [r["gpu_memory"]["scope"] for r in result["runs"]] == ["device_verified", "device_verified"] and result["ranges_over_repetitions"]["gpu_peak_mib"] == {"min": 40000, "max": 40000}
    gpu_support.write_scope(folder2, used=42000, allocated="MIG-aaaa-1111", parent="GPU-bbbb-2222", sampler_uuid="GPU-bbbb-2222", total=97887)            # the PARENT card's rows under a MIG allocation
    result = setup.summarise()
    unverified = result["runs"][1]
    assert unverified["gpu_memory"]["scope"] == "scope_unverified" and unverified["gpu_memory"]["reason"] == "parent_identity" and unverified["gpu_peak_mib"] is None
    assert unverified["gpu_memory"]["device_reported_peak_mib"] == 42000 and "not a slice measurement" in unverified["gpu_memory"]["label"]
    assert result["runs"][0]["gpu_peak_mib"] == 40000
    assert result["ranges_over_repetitions"]["gpu_peak_mib"] == {"min": 40000, "max": 40000}                    # the unverified 42000 is NOT in the numeric range
    scopes = {s["invocation_id"]: s for s in result["ranges_over_repetitions"]["gpu_scopes"]}
    assert scopes[R2]["scope"] == "scope_unverified" and scopes[R2]["peak_mib"] is None and scopes[R2]["device_reported_peak_mib"] == 42000 and scopes[R1]["peak_mib"] == 40000
    (folder1 / "gpu-scope.json").unlink()
    (folder2 / "gpu-scope.json").unlink()
    result = setup.summarise()
    assert [r["gpu_memory"]["scope"] for r in result["runs"]] == ["unavailable", "unavailable"] and [r["gpu_peak_mib"] for r in result["runs"]] == [None, None]
    assert result["ranges_over_repetitions"]["gpu_peak_mib"] == {"min": None, "max": None} and result["verdict"] == "ACCEPTED"                    # unavailable telemetry never changes the verdict


def test_the_old_numeric_function_is_gone_so_a_stale_caller_fails_loudly():
    import compare_backends
    assert not hasattr(compare_backends, "peak_gpu_mib")
    for name in ("decision_summary.py", "e1_selection.py", "full_report.py", "compare_backends.py"):
        assert "peak_gpu_mib(" not in (TOOLS / name).read_text(encoding="utf-8"), name
