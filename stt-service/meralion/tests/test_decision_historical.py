"""tools/decision.py --baseline-kind frozen-historical (full development validation). Synthetic evidence only; the matched-allocation tests stay in test_decision.py and must keep passing."""
import hashlib
import json
import subprocess

import pytest

import test_decision as td
from meralion_worker import config as cfg
from meralion_worker.decoders import frozen

acceptance = td.acceptance
decision = td.decision
SCHED0_ID = td.SCHED0_ID
RUN_COMMIT = "89abcdef0123456789abcdef0123456789abcdef"                       # the commit the full runs used (different from the commit the structural evidence was made at)


class Historical(td.World):
    """A frozen-historical evidence set: replay-generated baseline documents, a declared source, one candidate run, no standalone references and no baseline parity reports."""

    def __init__(self, tmp):
        super().__init__(tmp)
        self.overrides = td.to_sched0(self)
        digest = frozen.frozen_digest(SCHED0_ID)
        for path in self.paths["candidate"]["docs"].glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            document["meralion"]["decoder"]["frozen_digest"] = digest
            document["meralion"]["worker"]["git_commit"] = RUN_COMMIT
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        for path in self.paths["baseline"]["docs"].glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            block = document["meralion"]
            block["decoder"].update(backend="replay", settings_id="replay", frozen_digest="replay")
            block["worker"]["git_commit"] = "replay"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        self.rebind()
        self.write_scores("cand-raw", "candidate", "transcript_raw")
        self.write_scores("cand-final", "candidate", "transcript_final")
        self.write_scores("base-raw", "baseline", "transcript_raw")
        self.write_scores("base-final", "baseline", "transcript_final")
        for name in self.names:                                              # the frozen predictions carry no worker configuration fingerprint
            for run_id in ("base-raw", "base-final"):
                path = self.runs / run_id / "raw" / "predictions" / f"{name}.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                data["settings"].pop("config_fingerprint")
                path.write_text(json.dumps(data), encoding="utf-8")
        meta_path = tmp / "candidate" / "run-meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(worker_git_commit=RUN_COMMIT, pip_freeze_sha256="f" * 64, host="node1")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        self.rebind()
        self.write_report("compare", {"evidence": {"tf": acceptance.evidence_of(self.docs["baseline"]), "vllm": acceptance.evidence_of(self.docs["candidate"])},
                                      "gates": {"all_calls_in_both": True, "tf_documents_are_unique": True, "vllm_documents_are_unique": True}})
        self.write_report("replay_report", {"passed": True, "config_fingerprint": next(iter(acceptance.evidence_of(self.docs["baseline"])["config_fingerprints"])), "expected": {"calls": 2},
                                            "reference": {"raw_equal": 2, "results_transcripts_equal_target": 2, "mismatches": []}})
        self.write_source()
        self.languages = tmp / "languages.json"
        self.languages.write_text(json.dumps({"alpha": {"language": "ENGLISH"}, "beta": {"language": "EN_ZH_MIXED"}}), encoding="utf-8")

    def write_source(self, **over):
        source = {"schema": decision.SOURCE_SCHEMA,
                  "original_generation": {"scored_runs": {"raw": "base-raw", "final": "base-final"}, "hardware": "NVIDIA RTX PRO 6000 Blackwell Max-Q, MIG 1g.24gb slice", "hardware_evidence": "handoff document line 111",
                                          "runner": "stt-experiments/meralion-runner/run_windows.py", "replay_documents_are_gpu_evidence": False},
                  "replay": {"documents_sha256": acceptance.evidence_of(self.docs["baseline"])["documents_sha256"], "report_sha256": hashlib.sha256(self.reports["replay_report"].read_bytes()).hexdigest()}}
        source.update(over)
        self.write_report("source", source)

    def hist_args(self, **over):
        values = dict(self.overrides, baseline_kind="frozen-historical", baseline_source=self.reports["source"], baseline_replay_report=self.reports["replay_report"], languages=self.languages, repo=self.repo.path)
        values.update(over)
        return self.args(**values)

    repo = None


@pytest.fixture
def hist(tmp_path):
    world = Historical(tmp_path)
    world.repo = make_repo(tmp_path / "repo")
    return world


def make_repo(path, change_runtime=False):
    path.mkdir()
    base = ["stt-service/meralion/meralion_worker/a.py", "stt-service/meralion/configs/c.json"]
    def git(*args):
        subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], check=True, capture_output=True)
    git("init", "-q")
    for name in base:
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text("x", encoding="utf-8")
    git("add", "."); git("commit", "-q", "-m", "evidence commit")
    first = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (path / "stt-service/meralion/tools").mkdir(parents=True)
    (path / "stt-service/meralion/tools/t.py").write_text("tools only", encoding="utf-8")
    git("add", "."); git("commit", "-q", "-m", "tools")
    second = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return type("Repo", (), {"path": path, "first": first, "second": second, "git": staticmethod(git)})()


def bind_artefacts(hist, commit):
    """The probes, capture and replay report were made at `commit`; the runs used RUN_COMMIT."""
    for name, extra in (("probe_stock", {}), ("probe_candidate", {})):
        data = json.loads(hist.reports[name].read_text(encoding="utf-8"))
        data["worker_git_commit"] = commit
        hist.write_report(name, data)
    hist.write_report("plugin_test", td.plugin_report(evidence={"worker_git_commit": commit}))                 # the controlled logits test is bound to a commit too
    capture = json.loads(hist.reports["capture"].read_text(encoding="utf-8"))
    capture["worker_git_commit"] = commit
    hist.write_report("capture", capture)
    manifest = json.loads(td.PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    replay = json.loads(hist.reports["capture_replay"].read_text(encoding="utf-8"))
    replay["evidence"].update(capture_sha256=hashlib.sha256(hist.reports["capture"].read_bytes()).hexdigest(), worker_commit=commit)
    hist.write_report("capture_replay", replay)
    hist.overrides.update(probe_candidate=hist.reports["probe_candidate"], capture=hist.reports["capture"], capture_replay=hist.reports["capture_replay"], plugin_test=hist.reports["plugin_test"])


def run_decide(hist, **over):
    repo = hist.repo
    bind_artefacts(hist, repo.first)
    # the runs used the SECOND commit of the synthetic repo
    for path in hist.paths["candidate"]["docs"].glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["worker"]["git_commit"] = repo.second
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    meta_path = hist.tmp / "candidate" / "run-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["worker_git_commit"] = repo.second
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    hist.rebind()
    hist.write_scores("cand-raw", "candidate", "transcript_raw")
    hist.write_scores("cand-final", "candidate", "transcript_final")
    hist.write_report("compare", {"evidence": {"tf": acceptance.evidence_of(hist.docs["baseline"]), "vllm": acceptance.evidence_of(hist.docs["candidate"])},
                                  "gates": {"all_calls_in_both": True, "tf_documents_are_unique": True, "vllm_documents_are_unique": True}})
    return decision.decide(hist.hist_args(**over))


def problems(result, fragment):
    assert result["verdict"] == "NOT AN ACCEPTANCE VERDICT" and result["exit_code"] == 2, (result["verdict"], result["failed_items"])
    assert any(fragment in p for p in result["evidence_problems"]), result["evidence_problems"]


def test_a_complete_frozen_historical_evidence_set_is_accepted_and_says_what_it_cannot_establish(hist):
    result = run_decide(hist)
    assert result["verdict"] == "ACCEPTED", result["evidence_problems"] + result["failed_items"]
    assert result["identity"]["baseline_kind"] == "frozen-historical" and result["identity"]["baseline_source_sha256"] == hashlib.sha256(hist.reports["source"].read_bytes()).hexdigest()
    assert result["identity"]["candidate_allocation"] == {"slurm_job_id": "999", "gpu_uuid": "GPU-1", "pip_freeze_sha256": "f" * 64, "host": "node1"}
    assert result["identity"]["candidate_frozen_digests"] == [frozen.frozen_digest(SCHED0_ID)]
    text = " ".join(result["not_established"])
    for fragment in ("same-hardware comparison", "replay-generated", "baseline stop reasons", "no worker configuration fingerprint", "standalone single-recording reference"):
        assert fragment in text, fragment
    assert result["baseline"]["documents"].endswith("not GPU execution evidence")
    assert {k for k in result["items"] if "runtime_tree" in k} == {"structural:runtime_tree_unchanged_since_stock_plugin_probe", "structural:runtime_tree_unchanged_since_candidate_plugin_probe",
                                                                  "structural:runtime_tree_unchanged_since_live_capture", "structural:runtime_tree_unchanged_since_controlled_logits_test"}
    assert not any(k.startswith("structural:baseline_") and "parity" in k for k in result["items"])


def test_missing_baseline_stop_reasons_stay_unavailable_and_are_counted_not_passed_as_zero(hist):
    for path in hist.paths["baseline"]["docs"].glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        for number, window in enumerate(document["meralion"]["trace"]["windows"]):
            window["stop_reason"] = None if number == 0 else "end_token"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    hist.rebind()
    hist.write_source()                                                                  # the declared source is bound to the baseline documents' hash
    result = run_decide(hist)
    assert result["verdict"] == "ACCEPTED" and any("unavailable for 2 of" in line for line in result["not_established"])


def test_the_default_matched_mode_is_unchanged_and_still_requires_a_clean_baseline_commit_and_the_same_gpu(tmp_path):
    world = td.World(tmp_path)
    assert world.decide()["verdict"] == "ACCEPTED"
    assert "not_established" in world.decide() and world.decide()["not_established"] == [] and world.decide()["identity"]["baseline_kind"] == "matched-allocation"
    world.write_report("probe_stock", {**json.loads(world.reports["probe_stock"].read_text(encoding="utf-8")), "worker_git_commit": "e" * 40})
    assert "made with worker commit" in " ".join(world.decide(probe_stock=world.reports["probe_stock"])["evidence_problems"])      # the tree rule is NOT available in the default mode


def test_a_baseline_that_is_not_replay_generated_is_not_a_frozen_historical_baseline(hist):
    for path in hist.paths["baseline"]["docs"].glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["decoder"].update(backend="transformers", settings_id="stock-transformers-v1")
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    problems(run_decide(hist), "baseline settings id")


def test_the_candidate_must_still_have_one_clean_commit_the_frozen_digest_and_the_execution_profile(hist):
    result = run_decide(hist)
    assert result["verdict"] == "ACCEPTED"
    for path in hist.paths["candidate"]["docs"].glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["decoder"]["frozen_digest"] = "0" * 64
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    hist.rebind()
    problems(decision.decide(hist.hist_args()), "settings digest")
    for path in hist.paths["candidate"]["docs"].glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["decoder"]["frozen_digest"] = frozen.frozen_digest(SCHED0_ID)
        document["meralion"]["worker"]["git_commit"] = "unknown"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    hist.rebind()
    problems(decision.decide(hist.hist_args()), "clean recorded commit")
    td.set_execution(hist, {"mode": "default", "variable": None, "engine_core_client": "SyncMPClient"}, sides=("candidate",))
    problems(decision.decide(hist.hist_args()), "scheduling profile")


def test_the_allocation_record_of_the_candidate_is_required_and_must_name_the_documents_commit(hist):
    meta_path = hist.tmp / "candidate" / "run-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    run_decide(hist)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("gpu_uuid")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    problems(decision.decide(hist.hist_args()), "records no gpu_uuid")
    meta.update(gpu_uuid="GPU-1", worker_git_commit="0" * 40)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    problems(decision.decide(hist.hist_args()), "run-meta records commit")


def test_exact_membership_window_and_state_checks_and_score_binding_are_kept(hist):
    run_decide(hist)
    hist.write_scores("cand-raw", "candidate", "transcript_raw", drop="beta")
    problems(decision.decide(hist.hist_args()), "scored recordings differ")
    hist.write_scores("cand-raw", "candidate", "transcript_raw")
    path = hist.runs / "base-final" / "raw" / "predictions" / "alpha.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["transcript"] = "another text"
    path.write_text(json.dumps(data), encoding="utf-8")
    problems(decision.decide(hist.hist_args()), "is not this run's own")
    data["transcript"] = hist.docs["baseline"]["alpha"]["meralion"]["transcript_final"]
    data["settings"]["audio_sha256"] = "9" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    problems(decision.decide(hist.hist_args()), "another audio hash")


def test_a_historical_baseline_still_needs_a_declared_source_bound_to_its_documents_scores_and_replay_report(hist):
    run_decide(hist)
    problems(decision.decide(hist.hist_args(baseline_source=None)), "needs --baseline-source")
    hist.write_source(schema="other")
    problems(decision.decide(hist.hist_args()), "not a historical-baseline-source-v1")
    hist.write_source()
    hist.write_source(replay={"documents_sha256": "0" * 64, "report_sha256": hashlib.sha256(hist.reports["replay_report"].read_bytes()).hexdigest()})
    problems(decision.decide(hist.hist_args()), "replay documents hash")
    hist.write_source(replay={"documents_sha256": acceptance.evidence_of(hist.docs["baseline"])["documents_sha256"], "report_sha256": "0" * 64})
    problems(decision.decide(hist.hist_args()), "replay report hash")
    hist.write_source()
    generation = json.loads(hist.reports["source"].read_text(encoding="utf-8"))["original_generation"]
    for key, value, fragment in (("replay_documents_are_gpu_evidence", True, "replay metadata is not evidence of a GPU execution"), ("hardware", "", "original_generation.hardware"),
                                 ("scored_runs", {"raw": "x", "final": "y"}, "declares scored runs")):
        hist.write_source(original_generation={**generation, key: value})
        problems(decision.decide(hist.hist_args()), fragment)
    hist.write_source()
    hist.write_report("replay_report", {"passed": True, "config_fingerprint": "0" * 64, "expected": {"calls": 2}, "reference": {"raw_equal": 1, "results_transcripts_equal_target": 2, "mismatches": []}})
    hist.write_source()
    result = decision.decide(hist.hist_args())
    problems(result, "replay reproducing")
    problems(result, "configuration fingerprint is not that of the baseline documents")


def test_probes_and_capture_made_at_another_commit_need_an_identical_runtime_tree_in_git(hist):
    repo = hist.repo
    assert run_decide(hist)["verdict"] == "ACCEPTED"                                       # tools-only commit between: the trees are identical
    (repo.path / "stt-service/meralion/meralion_worker/a.py").write_text("changed", encoding="utf-8")
    repo.git("add", "."); repo.git("commit", "-q", "-m", "runtime change")
    third = subprocess.run(["git", "-C", str(repo.path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for path in hist.paths["candidate"]["docs"].glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["worker"]["git_commit"] = third
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    meta_path = hist.tmp / "candidate" / "run-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["worker_git_commit"] = third
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    hist.rebind()
    hist.write_scores("cand-raw", "candidate", "transcript_raw")
    hist.write_scores("cand-final", "candidate", "transcript_final")
    hist.write_report("compare", {"evidence": {"tf": acceptance.evidence_of(hist.docs["baseline"]), "vllm": acceptance.evidence_of(hist.docs["candidate"])},
                                  "gates": {"all_calls_in_both": True, "tf_documents_are_unique": True, "vllm_documents_are_unique": True}})
    problems(decision.decide(hist.hist_args()), "runtime tree is not shown to be identical")
    # an unresolvable commit is a problem too, never a pass
    bind_artefacts(hist, "d" * 40)
    problems(decision.decide(hist.hist_args()), "cannot be resolved")


def test_the_command_line_requires_the_historical_arguments_only_in_that_mode(hist, tmp_path):
    with pytest.raises(SystemExit):
        decision.main(["--baseline-kind", "frozen-historical", "--jobs", "x"])
    parser = decision.build_parser()
    namespace = parser.parse_args(["--jobs", "j", "--pilot", "p", "--runs", "r", "--candidate-config", "c", "--out", "o", "--compare", "x", "--plugin-test", "x", "--probe-stock", "x", "--probe-candidate", "x",
                                   "--config-compare-raw", "x", "--config-compare-final", "x", "--candidate-parity", "x", "--candidate-docs", "x", "--candidate-state", "x", "--baseline-docs", "x",
                                   "--baseline-state", "x", "--capture", "x", "--capture-replay", "x", "--package-manifest", "x", "--score", "a=b", "c=d", "e=f", "g=h"])
    assert namespace.baseline_kind == "matched-allocation" and namespace.candidate_c4_docs is None


def test_the_replay_documents_placeholder_audio_hash_is_not_used_the_frozen_predictions_are_compared_with_the_candidates_real_audio(hist):
    for path in hist.paths["baseline"]["docs"].glob("*.json"):                         # replay documents: a placeholder, not the real audio hash
        document = json.loads(path.read_text(encoding="utf-8"))
        document["meralion"]["audio"]["sha256"] = "7" * 64
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    hist.rebind()
    hist.write_source()
    assert run_decide(hist)["verdict"] == "ACCEPTED"
    assert any("baseline audio identity" in line for line in decision.decide(hist.hist_args())["not_established"])
    path = hist.runs / "base-raw" / "raw" / "predictions" / "beta.json"                # a frozen run that recorded a different audio than the candidate decoded
    data = json.loads(path.read_text(encoding="utf-8"))
    data["settings"]["audio_sha256"] = "8" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    problems(decision.decide(hist.hist_args()), "another audio hash")
