"""tools/decision.py: one verdict, strict about its evidence (Stage 3 follow-up D, correction C4). Synthetic runs only; no git-ignored data."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import synthetic
from harness import run, two_jobs
from meralion_worker import config as cfg
from meralion_worker.decoders import frozen

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import acceptance  # noqa: E402


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decision = load_tool("decision")
COMMIT = "0123456789abcdef0123456789abcdef01234567"
CANDIDATE_CONFIG = cfg.CANDIDATE_RP105_CONFIG
PACKAGE_MANIFEST = cfg.CONFIG_ROOT / "third_party" / "meralion-3-asr-0.1.0.json"
DEFAULT_EXECUTION = {"mode": "default", "variable": None, "engine_core_client": "SyncMPClient", "expected_engine_core_client": "SyncMPClient"}
SCHED0_EXECUTION = {"mode": "profile-in-process", "variable": "0", "engine_core_client": "InprocClient", "expected_engine_core_client": "InprocClient"}
RUNTIME_VERSIONS = {"vllm": "0.16.0", "torch": "2.9.1+cu128", "transformers": "4.57.6", "meralion-3-asr": "0.1.0"}


TOOL_SHA256 = "a" * 64


def fake_blob_resolver(repo, commit, path):
    """The seam of the synthetic worlds: every commit 'contains' a test file with TOOL_SHA256. The real resolver (git cat-file on a real temporary repository) is tested in test_plugin_evidence.py."""
    return TOOL_SHA256, None


def plugin_report(false=(), **over):
    """A controlled-logits-test report with EVERY required case, bound to the verified package, the commit and the runtime of the candidate documents, made where no device UUID was visible.
    `false` names cases that are False (a genuinely failed case: valid evidence, REJECTED); the summary fields are kept consistent with the cases unless overridden."""
    manifest = json.loads(PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    evidence = {"worker_git_commit": COMMIT, "tool_sha256": TOOL_SHA256, "package": {"files_sha256": {name: manifest["files"][name] for name in decision.PLUGIN_FILES}},
                "versions": {key: RUNTIME_VERSIONS[key] for key in decision.PLUGIN_VERSIONS}, "cuda_visible_devices": None, "uuid_device_visible": False}
    evidence.update(over.pop("evidence", {}))
    cases = {name: name not in false for name in decision.REQUIRED_PLUGIN_CASES}
    report = {"test": "plugin_logits_test", "passed": not false, "failed": sorted(false), "not_testable_here": [], "cases": cases, "evidence": evidence}
    report.update(over)
    return report


class World:
    """A complete, consistent evidence set for two recordings (alpha, beta): candidate and baseline pilot runs, their standalone references, scores, reports, run-meta and plugin evidence."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.names = ["alpha", "beta"]
        self.candidate_fp = cfg.load_runtime_config(CANDIDATE_CONFIG, env={}).fingerprint
        self.docs, self.paths = {}, {}
        for side in ("candidate", "baseline"):
            self.build(side, two_jobs(), self.names)
            self.build(side + "_c4", synthetic.World({"alpha": dict(break_at=2)}), ["alpha"])
        self.jobs = tmp / "jobs.json"
        self.jobs.write_text(json.dumps({"jobs": [{"job_name": n, "input": f"/x/{n}.wav"} for n in self.names]}), encoding="utf-8")
        self.pilot = tmp / "pilot.json"
        self.pilot.write_text(json.dumps({"per_call": {"alpha": {"language": "ENGLISH"}, "beta": {"language": "EN_ZH_MIXED"}}}), encoding="utf-8")
        self.runs = tmp / "runs"
        self.scores = {"candidate_raw": "cand-raw", "candidate_final": "cand-final", "baseline_raw": "base-raw", "baseline_final": "base-final"}
        for key, run_id in self.scores.items():
            self.write_scores(run_id, "candidate" if key.startswith("candidate") else "baseline", "transcript_raw" if key.endswith("raw") else "transcript_final")
        for side in ("candidate", "baseline"):
            meta = {"slurm_job_id": "999", "gpu_uuid": "GPU-1", "vllm_enable_v1_multiprocessing": "unset"}
            (tmp / side / "run-meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.reports = {}
        for side in ("candidate", "baseline", "candidate_c4", "baseline_c4"):
            self.write_report(f"{side}_parity", {"verdict": "ACCEPTED", "evidence": acceptance.evidence_of(self.docs[side])})
        self.write_report("compare", {"evidence": {"tf": acceptance.evidence_of(self.docs["baseline"]), "vllm": acceptance.evidence_of(self.docs["candidate"])},
                                      "gates": {"all_calls_in_both": True, "tf_documents_are_unique": True, "vllm_documents_are_unique": True}})
        self.write_report("plugin_test", plugin_report())
        self.write_report("probe_stock", {"passed": True, "settings_id": "stock-vllm-package-v1", "worker_git_commit": COMMIT, "execution_kind": "diagnostic-in-process", "diagnostic_only": True,
                                          "execution": {"mode": "diagnostic-in-process", "engine_core_client": "InprocClient"}})
        self.write_report("probe_candidate", {"passed": True, "settings_id": "vllm-rp105-v1", "worker_git_commit": COMMIT, "execution_kind": "diagnostic-in-process", "diagnostic_only": True,
                                              "execution": {"mode": "diagnostic-in-process", "engine_core_client": "InprocClient"}})
        self.write_report("capture", {"format": 1, "settings_id": "vllm-rp105-v1", "worker_git_commit": COMMIT})
        self.write_replay()
        self.write_report("compare_raw", {"base": "base-raw", "candidates": {"cand-raw": {"large_omission_events": {"new": []}}}})
        self.write_report("compare_final", {"base": "base-final", "candidates": {"cand-final": {"large_omission_events": {"new": []}}}})

    def build(self, label, world, names):
        side = label.split("_")[0]
        run(world, self.tmp, base=label)
        folder = self.tmp / label / "out" / "transcripts"
        docs = {}
        for name in names:
            path = folder / f"{name}.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            block = document["meralion"]
            block["worker"]["git_commit"] = COMMIT
            block["decoder"].update(backend="vllm" if side == "candidate" else "transformers", settings_id="vllm-rp105-v1" if side == "candidate" else "stock-transformers-v1")
            block["config_fingerprint"] = self.candidate_fp if side == "candidate" else "f" * 64
            if side == "candidate":
                block["decoder"]["runtime"] = {"packages": dict(RUNTIME_VERSIONS), "patches": []}
                block["decoder"]["execution"] = dict(DEFAULT_EXECUTION)
                for w in block["trace"]["windows"]:
                    w.update(stop_reason="end_token", tokens=10)
                for d in block["retries"]["decodes"]:
                    d["result"].update(stop_reason="end_token", tokens=10)
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            docs[name] = document
        self.docs[label] = docs
        self.paths[label] = {"docs": folder, "state": self.tmp / label / "state"}

    def write_scores(self, run_id, side, field, errors=None, drop=None, duplicate=False):
        base = self.runs / run_id
        (base / "scores").mkdir(parents=True, exist_ok=True)
        (base / "raw" / "predictions").mkdir(parents=True, exist_ok=True)
        rows = []
        for name in self.names:
            block = self.docs[side][name]["meralion"]
            wer_errors = (errors or {}).get(name, 5)
            rows.append({"call_id": name, "wer": {"errors": wer_errors, "reference_words": 100000}, "entity_wer": {"errors": 2, "reference_words": 50}})
            (base / "raw" / "predictions" / f"{name}.json").write_text(json.dumps({"call_id": name, "transcript": block[field], "settings": {"audio_sha256": block["audio"]["sha256"], "config_fingerprint": block["config_fingerprint"]}}), encoding="utf-8")
        if drop:
            rows = [r for r in rows if r["call_id"] != drop]
        if duplicate:
            rows.append(rows[0])
        (base / "scores" / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def write_replay(self, settings_id="vllm-rp105-v1", **over):
        """The CPU replay report for the capture currently on disk, bound to it, to the verified package, the commit and the profile."""
        manifest = json.loads(PACKAGE_MANIFEST.read_text(encoding="utf-8"))
        evidence = {"capture_sha256": hashlib.sha256(self.reports["capture"].read_bytes()).hexdigest(), "settings_id": settings_id, "worker_commit": COMMIT,
                    "wheel_sha256": manifest["artifact"]["sha256"], "package_manifest_sha256": hashlib.sha256(PACKAGE_MANIFEST.read_bytes()).hexdigest(), "execution_kind": "default"}
        evidence.update(over.pop("evidence", {}))
        report = {"passed": True, "checks": {"live_request_correspondence": True, "replay_requests_equal": True, "replay_sampling_equal": True, "replay_text_equal": True}, "evidence": evidence}
        report.update(over)
        self.write_report("capture_replay", report)

    def rebind(self):
        """Rewrite every report that is bound to documents from the CURRENT documents (a test that edits documents on purpose keeps its reports honest)."""
        for side in ("candidate", "baseline", "candidate_c4", "baseline_c4"):
            self.docs[side] = acceptance.scan_documents(self.paths[side]["docs"])[0]
            self.write_report(f"{side}_parity", {"verdict": "ACCEPTED", "evidence": acceptance.evidence_of(self.docs[side])})
        self.write_report("compare", {"evidence": {"tf": acceptance.evidence_of(self.docs["baseline"]), "vllm": acceptance.evidence_of(self.docs["candidate"])},
                                      "gates": {"all_calls_in_both": True, "tf_documents_are_unique": True, "vllm_documents_are_unique": True}})

    def write_report(self, name, data):
        path = self.tmp / "reports" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        self.reports[name] = path

    def args(self, **over):
        values = dict(jobs=self.jobs, pilot=self.pilot, runs=self.runs, candidate_config=CANDIDATE_CONFIG, out=self.tmp / "decision.json", compare=self.reports["compare"], plugin_test=self.reports["plugin_test"],
                      probe_stock=self.reports["probe_stock"], probe_candidate=self.reports["probe_candidate"], config_compare_raw=self.reports["compare_raw"], config_compare_final=self.reports["compare_final"],
                      candidate_parity=self.reports["candidate_parity"], candidate_c4_parity=self.reports["candidate_c4_parity"], baseline_parity=self.reports["baseline_parity"],
                      baseline_c4_parity=self.reports["baseline_c4_parity"], candidate_docs=self.paths["candidate"]["docs"], candidate_state=self.paths["candidate"]["state"],
                      candidate_c4_docs=self.paths["candidate_c4"]["docs"], baseline_docs=self.paths["baseline"]["docs"], baseline_state=self.paths["baseline"]["state"],
                      baseline_c4_docs=self.paths["baseline_c4"]["docs"], candidate_settings="vllm-rp105-v1", score=dict(self.scores), capture=self.reports["capture"],
                      capture_replay=self.reports["capture_replay"], package_manifest=PACKAGE_MANIFEST, blob_resolver=staticmethod(fake_blob_resolver))
        values.update(over)
        return type("Args", (), values)()

    def decide(self, **over):
        return decision.decide(self.args(**over))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_a_complete_consistent_evidence_set_is_accepted(world):
    result = world.decide()
    assert result["verdict"] == "ACCEPTED" and result["exit_code"] == 0 and result["evidence_problems"] == [] and result["failed_items"] == []
    assert {entry["label"] for entry in result["inputs"]} >= {"candidate documents", "baseline run-meta", "candidate raw scores", "backend comparison report", "controlled logits test"}
    assert all(entry["sha256"] for entry in result["inputs"] if entry["label"].endswith("scores") or "report" in entry["label"])


def test_every_input_is_recorded_with_a_path_a_hash_and_its_identity(world):
    result = world.decide()
    documents = [e for e in result["inputs"] if e["label"] == "candidate documents"][0]
    assert documents["identity"]["documents_sha256"] == acceptance.evidence_of(world.docs["candidate"])["documents_sha256"] and documents["identity"]["recordings"] == 2
    assert [e for e in result["inputs"] if e["label"] == "candidate raw scores"][0]["identity"] == {"run_id": "cand-raw"}


# ---- the fixed quality rule, from integer counts --------------------------------------------------------------------------------------------------------------------------

def rescore(world, run_id, side, field, errors):
    world.write_scores(run_id, side, field, errors=errors)


def test_a_regression_below_the_displayed_precision_is_still_rejected(world):
    """1000 against 1001 errors in 200000 reference tokens both display as 0.50%: the comparison must use the integers."""
    for run_id, side, field, errors in (("base-raw", "baseline", "transcript_raw", {"alpha": 500, "beta": 500}), ("cand-raw", "candidate", "transcript_raw", {"alpha": 500, "beta": 501})):
        world.write_scores(run_id, side, field, errors=errors)
    result = world.decide()
    assert result["verdict"] == "REJECTED" and "quality:2_raw_wer_and_entity_wer_not_higher" in result["failed_items"]
    counts = result["quality_details"]["integer_counts"]["raw_wer"]
    assert counts["candidate_errors"] == 1001 and counts["baseline_errors"] == 1000 and counts["candidate_not_higher"] is False


def test_an_equal_error_count_and_a_lower_one_are_not_a_regression(world):
    world.write_scores("cand-final", "candidate", "transcript_final", errors={"alpha": 4, "beta": 5})
    assert world.decide()["verdict"] == "ACCEPTED"


def test_a_formatted_regression_alone_rejects(world):
    world.write_scores("cand-final", "candidate", "transcript_final", errors={"alpha": 6, "beta": 5})
    result = world.decide()
    assert result["verdict"] == "REJECTED" and result["failed_items"] == ["quality:3_formatted_wer_and_entity_wer_not_higher"]


def test_a_new_large_omission_rejects_in_either_the_raw_or_the_formatted_comparison(world):
    for name, run_id in (("compare_raw", "cand-raw"), ("compare_final", "cand-final")):
        world.write_report(name, {"base": "base-raw" if name == "compare_raw" else "base-final", "candidates": {run_id: {"large_omission_events": {"new": [{"call": "alpha", "tokens": 12}]}}}})
        assert world.decide()["failed_items"] == ["quality:4_no_new_large_omission_raw_and_formatted"]
        world.write_report(name, {"base": "base-raw" if name == "compare_raw" else "base-final", "candidates": {run_id: {"large_omission_events": {"new": []}}}})


def test_a_max_new_tokens_window_cannot_be_excluded_and_rejects(world):
    path = world.paths["candidate"]["docs"] / "beta.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["meralion"]["trace"]["windows"][2]["stop_reason"] = "max_new_tokens"
    path.write_text(json.dumps(document), encoding="utf-8")
    world.rebind()
    result = world.decide()
    assert result["verdict"] == "REJECTED" and "quality:6_no_max_new_tokens_no_unavailable_or_unexplained_stop" in result["failed_items"]


def test_an_unavailable_or_unexplained_stop_reason_rejects(world):
    for reason in (None, "other"):
        path = world.paths["candidate"]["docs"] / "alpha.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["trace"]["windows"][0]["stop_reason"] = reason
        path.write_text(json.dumps(document), encoding="utf-8")
        world.rebind()
        assert "quality:6_no_max_new_tokens_no_unavailable_or_unexplained_stop" in world.decide()["failed_items"]


def test_the_plugin_evidence_is_a_structural_prerequisite(world):
    world.write_report("plugin_test", plugin_report(false=("the_output_list_is_a_live_reference",)))          # complete, consistent evidence of a genuinely failed case
    failed = world.decide()
    assert failed["verdict"] == "REJECTED" and failed["evidence_problems"] == [] and "structural:plugin_controlled_logits_test_passed" in failed["failed_items"]
    world.write_report("plugin_test", plugin_report())
    world.write_report("probe_candidate", {"passed": False, "settings_id": "vllm-rp105-v1", "worker_git_commit": COMMIT})
    assert "structural:candidate_plugin_probe_passed" in world.decide()["failed_items"]


def test_a_structural_report_that_is_not_accepted_rejects(world):
    world.write_report("baseline_parity", {"verdict": "REJECTED", "evidence": acceptance.evidence_of(world.docs["baseline"])})
    assert world.decide()["verdict"] == "REJECTED"


def test_per_language_regressions_are_reported_but_do_not_gate(world):
    world.write_scores("cand-raw", "candidate", "transcript_raw", errors={"alpha": 4, "beta": 5})
    world.write_scores("base-raw", "baseline", "transcript_raw", errors={"alpha": 5, "beta": 4})
    result = world.decide()
    assert result["verdict"] == "ACCEPTED" and result["failed_items"] == []                    # the aggregate is equal (9 against 9 errors): the per-language regression is reported, not gating
    cells = result["per_language"]
    assert cells["EN_ZH_MIXED"]["raw_wer"]["candidate_higher"] is True and cells["ENGLISH"]["raw_wer"]["candidate_higher"] is False


# ---- evidence that does not belong together: NOT AN ACCEPTANCE VERDICT --------------------------------------------------------------------------------------------------------

def non_acceptance(result, fragment):
    assert result["verdict"] == "NOT AN ACCEPTANCE VERDICT" and result["exit_code"] == 2
    assert any(fragment in problem for problem in result["evidence_problems"]), result["evidence_problems"]


def test_a_passing_report_that_belongs_to_another_run_is_rejected_as_evidence(world):
    """The baseline's own passing parity report presented as the candidate's."""
    world.write_report("candidate_parity", json.loads(world.reports["baseline_parity"].read_text(encoding="utf-8")))
    non_acceptance(world.decide(), "candidate parity report: its evidence block is not that of the candidate documents")


def test_a_standalone_report_of_the_wrong_side_is_rejected(world):
    world.write_report("candidate_c4_parity", json.loads(world.reports["baseline_c4_parity"].read_text(encoding="utf-8")))
    non_acceptance(world.decide(), "candidate standalone parity report")


def test_a_stale_report_whose_documents_have_changed_since_is_rejected(world):
    path = world.paths["candidate"]["docs"] / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["meralion"]["transcript_raw"] += " edited after the report was made"
    path.write_text(json.dumps(document), encoding="utf-8")
    non_acceptance(world.decide(), "stale, or a report of another run")


def test_a_comparison_report_for_another_pairing_is_rejected(world):
    other = json.loads(world.reports["compare"].read_text(encoding="utf-8"))
    other["evidence"]["vllm"], other["evidence"]["tf"] = other["evidence"]["tf"], other["evidence"]["vllm"]
    world.write_report("compare", other)
    non_acceptance(world.decide(), "backend comparison report")


def test_a_missing_recording_in_the_documents_or_in_either_score_set_is_not_a_verdict(world):
    world.paths["candidate"]["docs"].joinpath("beta.json").unlink()
    non_acceptance(world.decide(), "missing outputs")


@pytest.mark.parametrize("key, side, field", [("candidate_raw", "candidate", "transcript_raw"), ("candidate_final", "candidate", "transcript_final"), ("baseline_raw", "baseline", "transcript_raw"), ("baseline_final", "baseline", "transcript_final")])
def test_a_missing_or_duplicated_score_row_in_any_of_the_four_score_sets_is_not_a_verdict(world, key, side, field):
    world.write_scores(world.scores[key], side, field, drop="beta")
    non_acceptance(world.decide(), "scored recordings differ")
    world.write_scores(world.scores[key], side, field, duplicate=True)
    non_acceptance(world.decide(), "duplicate score row")


def test_a_missing_formatted_score_set_is_not_a_verdict(world):
    non_acceptance(world.decide(score={**world.scores, "candidate_final": "does-not-exist"}), "does not exist")


def test_a_score_run_whose_predictions_are_not_this_runs_transcripts_is_not_a_verdict(world):
    path = world.runs / "cand-raw" / "raw" / "predictions" / "alpha.json"
    prediction = json.loads(path.read_text(encoding="utf-8"))
    prediction["transcript"] += " from another run"
    path.write_text(json.dumps(prediction), encoding="utf-8")
    non_acceptance(world.decide(), "is not this run's own transcript_raw")


def test_a_score_run_made_from_another_configuration_is_not_a_verdict(world):
    path = world.runs / "base-final" / "raw" / "predictions" / "beta.json"
    prediction = json.loads(path.read_text(encoding="utf-8"))
    prediction["settings"]["config_fingerprint"] = "0" * 64
    path.write_text(json.dumps(prediction), encoding="utf-8")
    non_acceptance(world.decide(), "another audio hash or configuration fingerprint")


def test_reference_counts_that_differ_between_candidate_and_baseline_are_not_a_verdict(world):
    path = world.runs / "cand-raw" / "scores" / "calls.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["wer"]["reference_words"] = 999
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    non_acceptance(world.decide(), "reference counts differ")


def test_a_candidate_with_the_wrong_settings_or_configuration_is_not_a_verdict(world):
    non_acceptance(world.decide(candidate_settings="stock-vllm-package-v1"), "candidate settings id")
    non_acceptance(world.decide(candidate_config=cfg.DEFAULT_CONFIG), "candidate config fingerprint")


def test_a_baseline_that_is_not_stock_transformers_under_strategy_v2_is_not_a_verdict(world):
    for label, mutate, fragment in (("settings", lambda b: b["decoder"].update(settings_id="stock-vllm-package-v1"), "baseline settings id"),
                                    ("strategy", lambda b: b["config"]["strategy"].update(id="fg-m2ax-seq-ap-lmonly-v1"), "baseline strategy")):
        fresh = World(world.tmp / label)
        path = fresh.paths["baseline"]["docs"] / "alpha.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document["meralion"])
        path.write_text(json.dumps(document), encoding="utf-8")
        non_acceptance(fresh.decide(), fragment)


def test_commits_must_be_one_clean_recorded_commit(world):
    for bad in ("unknown", COMMIT + "-dirty"):
        fresh = World(world.tmp / bad.replace("-", "_"))
        for side in ("candidate", "baseline"):
            for path in fresh.paths[side]["docs"].glob("*.json"):
                document = json.loads(path.read_text(encoding="utf-8"))
                document["meralion"]["worker"]["git_commit"] = bad
                path.write_text(json.dumps(document), encoding="utf-8")
        non_acceptance(fresh.decide(), "one clean recorded commit")


def test_different_commits_in_the_two_runs_are_not_a_verdict(world):
    path = world.paths["baseline"]["docs"] / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["meralion"]["worker"]["git_commit"] = "f" * 40
    path.write_text(json.dumps(document), encoding="utf-8")
    non_acceptance(world.decide(), "one clean recorded commit")


def test_the_two_runs_must_share_the_allocation_and_the_physical_gpu_and_not_use_the_diagnostic_mode(world):
    for key, value, fragment in (("gpu_uuid", "GPU-2", "gpu_uuid"), ("slurm_job_id", "1000", "slurm_job_id")):
        meta_path = world.tmp / "baseline" / "run-meta.json"
        original = meta_path.read_text(encoding="utf-8")
        meta = json.loads(original)
        meta[key] = value
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        non_acceptance(world.decide(), fragment)
        meta_path.write_text(original, encoding="utf-8")
    meta_path = world.tmp / "candidate" / "run-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["vllm_enable_v1_multiprocessing"] = "0"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    non_acceptance(world.decide(), "which its execution profile does not allow")


def test_a_probe_for_other_settings_or_another_commit_is_not_a_verdict(world):
    world.write_report("probe_candidate", {"passed": True, "settings_id": "stock-vllm-package-v1", "worker_git_commit": COMMIT})
    non_acceptance(world.decide(), "candidate plugin probe: made for settings")
    world.write_report("probe_candidate", {"passed": True, "settings_id": "vllm-rp105-v1", "worker_git_commit": "e" * 40})
    non_acceptance(world.decide(), "made with worker commit")


def test_a_standalone_reference_must_be_one_recording_with_the_same_identity_as_its_pilot(world):
    path = world.paths["candidate_c4"]["docs"] / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["meralion"]["config_fingerprint"] = "a" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    non_acceptance(world.decide(), "candidate_c4: not the same settings, configuration and commit")


def test_a_missing_input_file_is_not_a_verdict(world):
    non_acceptance(world.decide(plugin_test=world.tmp / "nope.json"), "does not exist")


def test_a_failing_structural_state_check_rejects_rather_than_blames_the_evidence(world):
    plan = world.paths["candidate"]["state"] / "inv-1" / "plan.json"
    data = json.loads(plan.read_text(encoding="utf-8"))
    data["selected"] = data["selected"][:1]
    plan.write_bytes(json.dumps(data).encode("utf-8"))
    result = world.decide()
    assert "structural:candidate_windows_plan_and_state_recomputed_ok" in result["failed_items"] or result["verdict"] == "NOT AN ACCEPTANCE VERDICT"


def test_the_command_writes_the_decision_once_and_returns_the_exit_code(world, tmp_path, monkeypatch):
    monkeypatch.setattr(decision, "committed_blob_sha256", fake_blob_resolver)             # the command line has no resolver option: the real one is replaced for this test only
    out = tmp_path / "decision-cli.json"
    argv = []
    for name, value in world.args(out=out).__class__.__dict__.items():
        if name.startswith("_") or name in ("score", "candidate_settings", "blob_resolver"):
            continue
        argv += [f"--{name.replace('_', '-')}", str(value)]
    argv += ["--score"] + [f"{k}={v}" for k, v in world.scores.items()]
    assert decision.main(argv) == 0 and json.loads(out.read_text(encoding="utf-8"))["verdict"] == "ACCEPTED"
    with pytest.raises(SystemExit, match="already exists"):
        decision.main(argv)


# ---- follow-up E: execution profiles, probe kinds and the capture/replay gate ---------------------------------------------------------------------------------------------------

SCHED0_CONFIG = cfg.CONFIG_ROOT / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json"
SCHED0_ID = "vllm-rp105-sched0-v1"


def to_sched0(world, execution=None, probe_kind="profile-in-process", launcher="unset"):
    """Turn the world's candidate into a scheduling-profile candidate (documents, fingerprint, scores, probe, capture) and return the decide() overrides."""
    fingerprint = cfg.load_runtime_config(SCHED0_CONFIG, env={}).fingerprint
    for side in ("candidate", "candidate_c4"):
        for path in world.paths[side]["docs"].glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            block = document["meralion"]
            block["decoder"].update(settings_id=SCHED0_ID, execution=dict(execution or SCHED0_EXECUTION))
            block["config_fingerprint"] = fingerprint
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    world.rebind()
    world.write_scores("cand-raw", "candidate", "transcript_raw")
    world.write_scores("cand-final", "candidate", "transcript_final")
    meta_path = world.tmp / "candidate" / "run-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["vllm_enable_v1_multiprocessing"] = launcher
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    world.write_report("probe_candidate", {"passed": True, "settings_id": SCHED0_ID, "worker_git_commit": COMMIT, "execution_kind": probe_kind, "diagnostic_only": probe_kind == "diagnostic-in-process",
                                           "execution": {"mode": probe_kind, "engine_core_client": "InprocClient"}})
    world.write_report("capture", {"format": 1, "settings_id": SCHED0_ID, "worker_git_commit": COMMIT})
    world.write_replay(settings_id=SCHED0_ID)
    return dict(candidate_config=SCHED0_CONFIG, candidate_settings=SCHED0_ID, probe_candidate=world.reports["probe_candidate"], capture=world.reports["capture"], capture_replay=world.reports["capture_replay"])


def set_execution(world, execution, sides=("candidate", "candidate_c4")):
    for side in sides:
        for path in world.paths[side]["docs"].glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            if execution is None:
                document["meralion"]["decoder"].pop("execution")
            else:
                document["meralion"]["decoder"]["execution"] = execution
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    world.rebind()


def test_a_scheduling_profile_candidate_is_accepted_when_it_is_observed_in_process_and_its_probe_is_a_profile_probe(world):
    result = world.decide(**to_sched0(world))
    assert result["verdict"] == "ACCEPTED", result["evidence_problems"] + result["failed_items"]
    assert result["identity"]["candidate"]["settings_ids"] == [SCHED0_ID] and result["identity"]["baseline"]["settings_ids"] == ["stock-transformers-v1"]
    assert result["quality_details"]["new_loops"] == 0 and "candidate_plugin_probe_is_diagnostic_only" not in result["informational"]


@pytest.mark.parametrize("execution", [
    {"mode": "default", "variable": None, "engine_core_client": "SyncMPClient"},                                    # the profile's documents say the default mode ran
    {"mode": "profile-in-process", "variable": "0", "engine_core_client": "SyncMPClient"},                           # the variable was set but the client is not in process
    {"mode": "profile-in-process", "variable": None, "engine_core_client": "InprocClient"},                          # the client is in process but the variable was not recorded
    {"mode": "diagnostic-in-process", "variable": "0", "engine_core_client": "InprocClient"},                        # a diagnostic construction is not a profile
    None])
def test_a_scheduling_profile_must_be_observed_as_such_in_every_document(world, execution):
    overrides = to_sched0(world)
    set_execution(world, execution)
    world.write_scores("cand-raw", "candidate", "transcript_raw")
    world.write_scores("cand-final", "candidate", "transcript_final")
    non_acceptance(world.decide(**overrides), "candidate" if execution is None else "a scheduling profile must be observed")


@pytest.mark.parametrize("execution", [
    {"mode": "profile-in-process", "variable": "0", "engine_core_client": "InprocClient"},                           # a default profile running in process
    {"mode": "diagnostic-in-process", "variable": "0", "engine_core_client": "InprocClient"},
    {"mode": "default", "variable": "0", "engine_core_client": "SyncMPClient"},
    None])
def test_a_default_profile_must_be_observed_default_and_multiprocess(world, execution):
    set_execution(world, execution)
    non_acceptance(world.decide(), "candidate")


def test_the_launcher_record_follows_the_profile_not_a_global_rule(world):
    assert world.decide(**to_sched0(world, launcher="0"))["verdict"] == "ACCEPTED"                              # a scheduling profile tolerates its own value
    fresh = World(world.tmp / "conflict")
    non_acceptance(fresh.decide(**to_sched0(fresh, launcher="1")), "which its execution profile does not allow")


def test_a_default_profile_still_refuses_a_zero_launcher_record_and_a_baseline_with_one_too(world):
    for side in ("candidate", "baseline"):
        fresh = World(world.tmp / side)
        meta_path = fresh.tmp / side / "run-meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["vllm_enable_v1_multiprocessing"] = "0"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        non_acceptance(fresh.decide(), "which its execution profile does not allow")


@pytest.mark.parametrize("report", [
    {"execution_kind": "profile-in-process", "diagnostic_only": False, "execution": {"engine_core_client": "InprocClient"}},       # a default profile probed as if it were production
    {"execution_kind": "diagnostic-in-process", "diagnostic_only": False, "execution": {"engine_core_client": "InprocClient"}},
    {"diagnostic_only": True},                                                                                                     # an old-style report without the stamp
])
def test_a_default_profile_probe_must_be_stamped_diagnostic(world, report):
    world.write_report("probe_stock", {"passed": True, "settings_id": "stock-vllm-package-v1", "worker_git_commit": COMMIT, **report})
    non_acceptance(world.decide(), "stock plugin probe: a default-profile probe must be a diagnostic in-process probe")


@pytest.mark.parametrize("kind, diagnostic, client", [("diagnostic-in-process", True, "InprocClient"), ("profile-in-process", True, "InprocClient"), ("profile-in-process", False, "SyncMPClient"), (None, None, None)])
def test_a_scheduling_profile_probe_must_be_a_profile_probe(world, kind, diagnostic, client):
    overrides = to_sched0(world)
    world.write_report("probe_candidate", {"passed": True, "settings_id": SCHED0_ID, "worker_git_commit": COMMIT, "execution_kind": kind, "diagnostic_only": diagnostic,
                                           "execution": {"engine_core_client": client}})
    non_acceptance(world.decide(**overrides), "candidate plugin probe: a scheduling-profile probe must be profile-in-process")


def test_a_diagnostic_default_profile_probe_is_accepted_but_recorded_as_diagnostic_only(world):
    result = world.decide()
    assert result["verdict"] == "ACCEPTED" and set(result["informational"]) == {"stock_plugin_probe_is_diagnostic_only", "candidate_plugin_probe_is_diagnostic_only"}


@pytest.mark.parametrize("key, value", [("capture_sha256", "0" * 64), ("settings_id", "stock-vllm-package-v1"), ("worker_commit", "e" * 40), ("wheel_sha256", "1" * 64), ("package_manifest_sha256", "2" * 64)])
def test_a_capture_replay_report_of_another_capture_package_commit_or_profile_is_not_a_verdict(world, key, value):
    world.write_replay(evidence={key: value})
    non_acceptance(world.decide(), f"capture replay report: its {key}")


def test_a_capture_made_for_another_profile_or_commit_is_not_a_verdict(world):
    for key, value in (("settings_id", "stock-vllm-package-v1"), ("worker_git_commit", "e" * 40)):
        capture = {"format": 1, "settings_id": "vllm-rp105-v1", "worker_git_commit": COMMIT, key: value}
        world.write_report("capture", capture)
        world.write_replay()
        non_acceptance(world.decide(), f"live capture: its {key}")


def test_a_failing_or_empty_replay_rejects_the_candidate_and_a_missing_capture_is_not_a_verdict(world):
    world.write_replay(passed=False, checks={"live_request_correspondence": True, "replay_sampling_equal": False})
    failing = world.decide()
    assert failing["verdict"] == "REJECTED" and "structural:adapter_equivalent_by_capture_replay" in failing["failed_items"]
    world.write_replay(checks={})
    assert "structural:adapter_equivalent_by_capture_replay" in world.decide()["failed_items"]
    world.write_replay()
    non_acceptance(world.decide(capture=world.tmp / "nope.json"), "does not exist")


# ---- REV 3.2 Amendment 6: the controlled logits test is evidence only for the commit, package files and runtime it names ----------------------------------------------------------

@pytest.mark.parametrize("override, fragment", [
    (dict(evidence={"worker_git_commit": "e" * 40}), "controlled logits test: made with worker commit"),
    (dict(evidence={"package": {"files_sha256": {"_vllm_plugin/__init__.py": "0" * 64, "_vllm_plugin/transformers_utils/no_repeat_logits_processor.py": "1" * 64}}}), "in the report, the verified package's is"),
    (dict(evidence={"versions": {"vllm": "0.17.0", "torch": "2.9.1+cu128", "meralion-3-asr": "0.1.0"}}), "vllm is '0.17.0' in the report but"),
    (dict(evidence={"versions": {"vllm": "0.16.0", "torch": "2.8.0", "meralion-3-asr": "0.1.0"}}), "torch is '2.8.0' in the report"),
    (dict(evidence={"uuid_device_visible": True, "cuda_visible_devices": "MIG-abc"}), "a device UUID was visible"),
    (dict(not_testable_here=["a_slot_swap_moves_the_history_with_it"]), "cases that could not be run"),
])
def test_a_plugin_report_for_another_commit_package_runtime_or_a_device_mapped_host_is_an_evidence_problem(world, override, fragment):
    world.write_report("plugin_test", plugin_report(**override))
    non_acceptance(world.decide(), fragment)


FIRST_CASE, SECOND_CASE = decision.REQUIRED_PLUGIN_CASES[0], decision.REQUIRED_PLUGIN_CASES[1]


def with_cases(report, **changes):
    report["cases"] = {**report["cases"], **changes}
    return report


def without_case(report, name):
    report["cases"] = {key: value for key, value in report["cases"].items() if key != name}
    return report


def test_the_exact_probe_codex_reproduced_is_no_longer_accepted(world):
    """A report with tool_sha256 = 64 zeroes, no cases at all and passed=true used to give ACCEPTED (only the flag was read)."""
    report = plugin_report(evidence={"tool_sha256": "0" * 64}, cases={}, passed=True)
    world.write_report("plugin_test", report)
    result = world.decide()
    assert result["verdict"] == "NOT AN ACCEPTANCE VERDICT"
    text = " ".join(result["evidence_problems"])
    assert "tool_sha256" in text and "required cases are missing" in text


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: without_case(r, FIRST_CASE), "required cases are missing"),
    (lambda r: with_cases(r, surprise=True), "cases that are not required"),
    (lambda r: with_cases(r, **{FIRST_CASE: None}), "were not tested"),
    (lambda r: with_cases(r, **{FIRST_CASE: "yes"}), "neither true nor false"),
    (lambda r: with_cases(r, **{FIRST_CASE: 1}), "neither true nor false"),
    (lambda r: with_cases(r, **{FIRST_CASE: False}), "failed list"),                                        # a false case whose summary says nothing failed
    (lambda r: r.update(passed=False) or r, "contradicts the cases"),                                       # all cases true, passed false
    (lambda r: with_cases(r, **{FIRST_CASE: False}) and r.update(failed=[FIRST_CASE]) or r, "contradicts the cases"),   # a false case but passed true
    (lambda r: r.update(not_testable_here=[SECOND_CASE]) or r, "cases that could not be run"),
    (lambda r: r.update(test="something_else") or r, "not a plugin_logits_test report"),
    (lambda r: r.update(cases=["a", "b"]) or r, "no cases object"),
    (lambda r: r["evidence"].update(tool_sha256="0" * 64) or r, "tool_sha256 is"),
    (lambda r: r["evidence"].update(tool_sha256=None) or r, "tool_sha256 is"),
])
def test_a_missing_malformed_or_contradictory_plugin_report_is_an_evidence_problem_never_a_pass(world, mutate, fragment):
    world.write_report("plugin_test", mutate(plugin_report()))
    result = world.decide()
    non_acceptance(result, fragment)


def test_complete_consistent_evidence_of_failed_cases_is_valid_evidence_and_the_candidate_is_rejected(world):
    for failing in ((FIRST_CASE,), (FIRST_CASE, SECOND_CASE)):
        world.write_report("plugin_test", plugin_report(false=failing))
        result = world.decide()
        assert result["verdict"] == "REJECTED" and result["evidence_problems"] == []
        assert "structural:plugin_controlled_logits_test_passed" in result["failed_items"]


def test_the_gate_comes_from_the_case_values_not_from_the_reports_flag(world):
    report = plugin_report()
    report["passed"] = False                                                      # the flag alone cannot reject a report whose cases all pass, nor accept one that does not
    assert decision.plugin_gate(report) is True
    report = plugin_report(false=(FIRST_CASE,))
    report["passed"] = True
    assert decision.plugin_gate(report) is False


def test_a_plugin_report_without_an_evidence_block_is_an_evidence_problem(world):
    report = plugin_report()
    del report["evidence"]
    world.write_report("plugin_test", report)
    non_acceptance(world.decide(), "the report has no evidence block")


def test_the_plugin_binding_needs_the_package_manifest_and_reads_the_candidate_documents_runtime(world, tmp_path):
    result = world.decide()
    assert result["verdict"] == "ACCEPTED"
    path = world.paths["candidate"]["docs"] / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["meralion"]["decoder"]["runtime"]["packages"]["vllm"] = "0.15.0"
    path.write_text(json.dumps(document), encoding="utf-8")
    world.rebind()
    non_acceptance(world.decide(), "vllm is '0.16.0' in the report but")


# ---- REV 3.0 A2: one uninterrupted attempt -----------------------------------------------------------------------------------------------------------------------------------------

def edit_meta(world, side, **changes):
    path = world.tmp / side / "run-meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(changes)
    path.write_text(json.dumps(meta), encoding="utf-8")


@pytest.mark.parametrize("side", ["candidate", "baseline"])
def test_a_restarted_or_second_attempt_run_is_not_acceptance_evidence(world, side):
    assert world.decide()["verdict"] == "ACCEPTED"
    edit_meta(world, side, restart_count=0, attempt=1)                                                   # explicit first attempt is fine
    assert world.decide()["verdict"] == "ACCEPTED"
    edit_meta(world, side, restart_count=1)
    non_acceptance(world.decide(), f"{side}: the run was restarted by the scheduler")
    edit_meta(world, side, restart_count=0, attempt=2)
    non_acceptance(world.decide(), f"{side}: run-meta records attempt 2")
    edit_meta(world, side, attempt=1)
    (world.tmp / side / "attempts").mkdir()
    non_acceptance(world.decide(), f"{side}: an attempts/ folder exists next to the run")


# ---- REV 3.0 B: the FP16 profile's engine must be the observed one -------------------------------------------------------------------------------------------------------------------

FP16_ID = "vllm-rp105-sched0-fp16-v1"
FP16_CONFIG = cfg.CONFIG_ROOT / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0-fp16.json"


def to_fp16(world, engine=None):
    """The scheduling-profile world, re-labelled as the FP16 profile with the engine the profile expects (or `engine`)."""
    overrides = to_sched0(world)
    fingerprint = cfg.load_runtime_config(FP16_CONFIG, env={}).fingerprint
    observed = engine if engine is not None else {**frozen.effective_engine_expectation(FP16_ID), "num_gpu_blocks": 100, "block_size": 16}
    for side in ("candidate", "candidate_c4"):
        for path in world.paths[side]["docs"].glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            block = document["meralion"]
            block["decoder"].update(settings_id=FP16_ID, engine_effective=observed)
            block["config_fingerprint"] = fingerprint
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    world.rebind()
    world.write_scores("cand-raw", "candidate", "transcript_raw")
    world.write_scores("cand-final", "candidate", "transcript_final")
    world.write_report("probe_candidate", {**json.loads(world.reports["probe_candidate"].read_text(encoding="utf-8")), "settings_id": FP16_ID})
    world.write_report("capture", {"format": 1, "settings_id": FP16_ID, "worker_git_commit": COMMIT})
    world.write_replay(settings_id=FP16_ID)
    return dict(overrides, candidate_config=FP16_CONFIG, candidate_settings=FP16_ID, probe_candidate=world.reports["probe_candidate"], capture=world.reports["capture"], capture_replay=world.reports["capture_replay"])


def test_an_fp16_run_with_the_expected_engine_is_accepted(world):
    assert world.decide(**to_fp16(world))["verdict"] == "ACCEPTED"


@pytest.mark.parametrize("engine_change, fragment", [(dict(dtype="torch.bfloat16"), "dtype"), (dict(max_num_seqs=8), "max_num_seqs"), (dict(attention_backend="TRITON_ATTN"), "attention_backend")])
def test_an_fp16_run_whose_engine_was_bf16_or_had_lower_concurrency_is_an_evidence_problem(world, engine_change, fragment):
    engine = {**frozen.effective_engine_expectation(FP16_ID), **engine_change}
    non_acceptance(world.decide(**to_fp16(world, engine=engine)), fragment)


def test_an_fp16_run_without_an_engine_record_is_an_evidence_problem(world):
    overrides = to_fp16(world)
    for path in world.paths["candidate"]["docs"].glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["decoder"].pop("engine_effective")
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    world.rebind()
    non_acceptance(world.decide(**overrides), "has no engine_effective record")
