"""tools/e1_selection.py: the fixed promotion rule of E1 (Stage 3 follow-up E, section 8.4). Synthetic runs only; no git-ignored data."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

import synthetic
from harness import run

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = load_tool("e1_selection")
COMMIT = "0123456789abcdef0123456789abcdef01234567"
NAMES = ["CALL_134", "CALL_103"]
ARMS = {"A": "vllm-rp105-v1", "B": "vllm-rp105-sched0-v1", "C": "stock-vllm-package-v1", "D": "stock-vllm-package-sched0-v1"}
RUNAWAY = (24960000, 25440000)


class E1World:
    """Four arms, two fresh-engine runs each, over the same two recordings, plus a Transformers reference; every run complete, clean and equal to the reference unless a test says otherwise."""

    def __init__(self, tmp, errors=None):
        self.tmp = Path(tmp)
        self.runs_root, self.score_root, self.compare_dir = self.tmp / "runs", self.tmp / "scores", self.tmp / "compare"
        self.compare_dir.mkdir(parents=True)
        self.errors = errors or {}
        self.run_ids = {arm: [f"vllm-l40s-e1-{arm}-{n}" for n in (1, 2)] for arm in ARMS}
        for arm_runs in self.run_ids.values():
            for run_id in arm_runs:
                self.build_run(run_id)
        self.build_reference()
        self.manifest = self.tmp / "e1-manifest.json"
        self.manifest.write_text(json.dumps({"jobs": NAMES, "arms": {arm: {"settings_id": ARMS[arm], "config": f"config-{arm}.json", "run_ids": ids} for arm, ids in self.run_ids.items()}}), encoding="utf-8")
        self.rescore()

    def world(self):
        return synthetic.World({NAMES[0]: dict(break_at=2), NAMES[1]: dict(break_at=4)})

    def folder(self, run_id):
        return self.runs_root / f"worker-v0-{run_id}"

    def build_run(self, run_id):
        run(self.world(), self.runs_root, invocation_id=run_id, base=f"worker-v0-{run_id}")
        folder = self.folder(run_id)
        for path in (folder / "out" / "transcripts").glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            block = document["meralion"]
            block["worker"]["git_commit"] = COMMIT
            for window in block["trace"]["windows"]:
                window.update(stop_reason="end_token", tokens=10)
            for decode in block["retries"]["decodes"]:
                decode["result"].update(stop_reason="end_token", tokens=10)
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        (folder / "wall.json").write_text(json.dumps({"wall_seconds": 80, "exit_code": 0}), encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps({"exit_code": 0, "done": len(NAMES), "failed": 0, "members": len(NAMES)}), encoding="utf-8")
        (folder / "run-meta.json").write_text(json.dumps({"worker_git_commit": COMMIT, "slurm_job_id": "999", "gpu_uuid": "GPU-1"}), encoding="utf-8")

    def build_reference(self):
        run(self.world(), self.tmp, invocation_id="reference", base="reference")
        self.reference_docs = self.tmp / "reference" / "out" / "transcripts"

    def edit(self, run_id, mutate):
        for path in (self.folder(run_id) / "out" / "transcripts").glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            mutate(document["meralion"], path.stem)
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    def docs(self, run_id):
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (self.folder(run_id) / "out" / "transcripts").glob("*.json")}

    def write_scores(self, run_id, kind):
        score_run = f"worker-v0-{run_id}-{'raw' if kind == 'raw' else 'amt1'}"
        base = self.score_root / score_run
        shutil.rmtree(base, ignore_errors=True)
        (base / "scores").mkdir(parents=True)
        (base / "raw" / "predictions").mkdir(parents=True)
        rows = []
        for name, document in self.docs(run_id).items():
            block = document["meralion"]
            errors = self.errors.get((run_id, name), self.errors.get(run_id, 5))
            rows.append({"call_id": name, "wer": {"errors": errors, "reference_words": 1000}, "entity_wer": {"errors": 1, "reference_words": 50}})
            field = "transcript_raw" if kind == "raw" else "transcript_final"
            (base / "raw" / "predictions" / f"{name}.json").write_text(json.dumps({"call_id": name, "transcript": block[field],
                                                                                     "settings": {"audio_sha256": block["audio"]["sha256"], "config_fingerprint": block["config_fingerprint"]}}), encoding="utf-8")
        (base / "scores" / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def write_compare(self, run_id, kind, new=()):
        score_run = f"worker-v0-{run_id}-{'raw' if kind == 'raw' else 'amt1'}"
        (self.compare_dir / f"{run_id}-{kind}.json").write_text(json.dumps({"base": "reference", "candidates": {score_run: {"large_omission_events": {"new": list(new)}}}}), encoding="utf-8")

    def rescore(self):
        for ids in self.run_ids.values():
            for run_id in ids:
                if self.folder(run_id).is_dir():
                    for kind in ("raw", "final"):
                        self.write_scores(run_id, kind)
                        if not (self.compare_dir / f"{run_id}-{kind}.json").is_file():
                            self.write_compare(run_id, kind)

    def args(self, **over):
        values = dict(manifest=self.manifest, runs_root=self.runs_root, score_root=self.score_root, compare_dir=self.compare_dir, reference_docs=self.reference_docs)
        values.update(over)
        return type("Args", (), values)()

    def decide(self, **over):
        return selection.decide(self.args(**over))


@pytest.fixture
def world(tmp_path):
    return E1World(tmp_path)


def ineligible(result, arm, fragment):
    assert result["arms"][arm]["eligible"] is False and result["promoted"] != arm
    assert any(fragment in reason for reasons in result["arms"][arm]["reasons"].values() for reason in reasons), result["arms"][arm]["reasons"]


# ---- eligibility: every missing or failed piece of work makes the arm ineligible -------------------------------------------------------------------------------------------------

def test_a_complete_clean_set_of_runs_is_all_eligible_and_ties_to_the_fixed_arm_order(world):
    result = world.decide()
    assert all(a["eligible"] for a in result["arms"].values()) and result["promoted"] == "C" and [r["arm"] for r in result["ranking"]] == ["C", "A", "D", "B"]


def test_a_missing_window_a_missing_recording_and_a_retry_missing_from_the_runs_own_plan_make_the_arm_ineligible(world):
    world.edit(world.run_ids["A"][0], lambda block, name: block["trace"]["windows"].pop() if name == NAMES[0] else None)
    ineligible(world.decide(), "A", "first-pass windows")
    fresh = E1World(world.tmp / "b")
    (fresh.folder(fresh.run_ids["B"][1]) / "out" / "transcripts" / f"{NAMES[1]}.json").unlink()
    ineligible(fresh.decide(), "B", "missing outputs")
    fresh = E1World(world.tmp / "c")
    fresh.edit(fresh.run_ids["D"][0], lambda block, name: (block["retries"]["decodes"].clear(), block["retries"].update(selected=0)) if block["retries"]["decodes"] else None)
    ineligible(fresh.decide(), "D", "decoded retries")


def test_a_nonzero_exit_an_incomplete_summary_and_a_missing_run_make_the_arm_ineligible(tmp_path):
    for label, change, fragment in (("exit", lambda w, r: (w.folder(r) / "wall.json").write_text(json.dumps({"wall_seconds": 5, "exit_code": 1})), "did not exit 0"),
                                    ("summary", lambda w, r: (w.folder(r) / "summary.json").write_text(json.dumps({"exit_code": 0, "done": 1, "failed": 0})), "every recording done"),
                                    ("missing", lambda w, r: shutil.rmtree(w.folder(r)), "is missing")):
        fresh = E1World(tmp_path / label)
        change(fresh, fresh.run_ids["A"][1])
        ineligible(fresh.decide(), "A", fragment)


@pytest.mark.parametrize("stop", ["max_new_tokens", "other", None])
def test_a_token_limit_an_unexplained_or_an_unavailable_stop_reason_disqualifies(tmp_path, stop):
    fresh = E1World(tmp_path)
    fresh.edit(fresh.run_ids["B"][0], lambda block, name: block["trace"]["windows"][0].update(stop_reason=stop) if name == NAMES[0] else None)
    ineligible(fresh.decide(), "B", "did not end with the end token")
    assert fresh.decide()["runs"][fresh.run_ids["B"][0]]["stop_reasons"]


def test_runs_that_do_not_share_one_clean_commit_one_allocation_and_one_gpu_are_all_ineligible(tmp_path):
    for label, key, value in (("commit", "worker_git_commit", "f" * 40), ("gpu", "gpu_uuid", "GPU-2"), ("job", "slurm_job_id", "1000")):
        fresh = E1World(tmp_path / label)
        meta_path = fresh.folder(fresh.run_ids["C"][0]) / "run-meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta[key] = value
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        result = fresh.decide()
        assert result["promoted"] is None and not any(a["eligible"] for a in result["arms"].values())
    for bad in ("unknown", COMMIT + "-dirty"):
        fresh = E1World(tmp_path / bad.replace("-", "_"))
        for ids in fresh.run_ids.values():
            for run_id in ids:
                meta_path = fresh.folder(run_id) / "run-meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["worker_git_commit"] = bad
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert fresh.decide()["promoted"] is None


def test_scores_that_are_not_this_runs_own_output_or_are_missing_disqualify(tmp_path):
    fresh = E1World(tmp_path)
    fresh.edit(fresh.run_ids["A"][0], lambda block, name: block.update(transcript_raw=block["transcript_raw"] + " edited after scoring"))
    ineligible(fresh.decide(), "A", "not this run's own output")
    fresh = E1World(tmp_path / "second")
    shutil.rmtree(fresh.score_root / f"worker-v0-{fresh.run_ids['D'][1]}-amt1")
    ineligible(fresh.decide(), "D", "final scores are missing")
    fresh = E1World(tmp_path / "third")
    (fresh.compare_dir / f"{fresh.run_ids['B'][0]}-raw.json").unlink()
    ineligible(fresh.decide(), "B", "config_compare output for the raw")


# ---- clarification 1: new large omissions and new loops disqualify before any ranking ----------------------------------------------------------------------------------------------

def test_the_arm_with_the_lowest_errors_is_not_eligible_with_a_new_large_omission_in_one_repetition(tmp_path):
    errors = {"vllm-l40s-e1-B-1": 1, "vllm-l40s-e1-B-2": 1}                                    # B has by far the lowest aggregate errors
    fresh = E1World(tmp_path, errors=errors)
    fresh.write_compare(fresh.run_ids["B"][1], "raw", new=[{"call": NAMES[0], "length": 7}])
    result = fresh.decide()
    ineligible(result, "B", "new large omission")
    assert result["promoted"] == "C" and "rank_key" not in result["arms"]["B"] and "B" not in [r["arm"] for r in result["ranking"]]


def test_a_formatted_only_large_omission_disqualifies_too(tmp_path):
    fresh = E1World(tmp_path, errors={"vllm-l40s-e1-A-1": 1, "vllm-l40s-e1-A-2": 1})
    fresh.write_compare(fresh.run_ids["A"][0], "final", new=[{"call": NAMES[1], "length": 5}])
    ineligible(fresh.decide(), "A", "in the final transcripts")


def test_a_new_detected_loop_in_one_repetition_disqualifies_before_ranking(tmp_path):
    fresh = E1World(tmp_path, errors={"vllm-l40s-e1-D-1": 1, "vllm-l40s-e1-D-2": 1})
    looping = "one two three four one two three four one two three four one two three four"
    fresh.edit(fresh.run_ids["D"][1], lambda block, name: block["trace"]["windows"][1].update(text=looping) if name == NAMES[0] else None)
    result = fresh.decide()
    ineligible(result, "D", "new detected loop")
    assert result["promoted"] == "C" and result["runs"][fresh.run_ids["D"][1]]["new_loops"]


def test_an_arm_clean_in_its_first_repetition_and_failing_in_its_second_is_ineligible(tmp_path):
    fresh = E1World(tmp_path, errors={"vllm-l40s-e1-A-1": 1, "vllm-l40s-e1-A-2": 1})
    fresh.write_compare(fresh.run_ids["A"][1], "raw", new=[{"call": NAMES[0], "length": 9}])
    result = fresh.decide()
    assert result["runs"][fresh.run_ids["A"][0]]["eligible"] is True and result["runs"][fresh.run_ids["A"][1]]["eligible"] is False and result["arms"]["A"]["eligible"] is False


def test_a_failed_run_never_helps_an_arm_a_stock_arm_with_one_good_run_and_one_failed_run_cannot_win(tmp_path):
    fresh = E1World(tmp_path, errors={"vllm-l40s-e1-D-1": 0, "vllm-l40s-e1-D-2": 0})            # the best possible scores, but the second run failed
    (fresh.folder(fresh.run_ids["D"][1]) / "wall.json").write_text(json.dumps({"wall_seconds": 3, "exit_code": 1}), encoding="utf-8")
    result = fresh.decide()
    assert result["arms"]["D"]["eligible"] is False and result["promoted"] != "D" and "D" not in [r["arm"] for r in result["ranking"]]


# ---- the complete ranking ---------------------------------------------------------------------------------------------------------------------------------------------------------

def test_level_one_the_worst_repetition_decides_not_the_best_or_the_mean(tmp_path):
    errors = {"vllm-l40s-e1-A-1": 0, "vllm-l40s-e1-A-2": 20, "vllm-l40s-e1-B-1": 7, "vllm-l40s-e1-B-2": 7}      # A: best 0 but worst 20; B: 7 and 7
    result = E1World(tmp_path, errors=errors).decide()
    assert result["promoted"] == "C"                                                           # C and D score 10 each in both runs: below A (worst 40) and B (worst 14)
    ranks = {r["arm"]: r["key"] for r in result["ranking"]}
    assert ranks["B"][0] == 14 and ranks["A"][0] == 40 and [r["arm"] for r in result["ranking"]].index("B") < [r["arm"] for r in result["ranking"]].index("A")


def test_level_two_the_sum_decides_when_the_worst_repetitions_tie(tmp_path):
    errors = {"vllm-l40s-e1-A-1": 5, "vllm-l40s-e1-A-2": 5, "vllm-l40s-e1-B-1": 3, "vllm-l40s-e1-B-2": 5, "vllm-l40s-e1-C-1": 9, "vllm-l40s-e1-C-2": 9, "vllm-l40s-e1-D-1": 9, "vllm-l40s-e1-D-2": 9}
    fresh = E1World(tmp_path, errors={k: v for k, v in errors.items()})
    # per-call errors are per run (each of two recordings): A worst 10, B worst 10 (3+3.. = 6 and 10)
    result = fresh.decide()
    keys = {r["arm"]: r["key"] for r in result["ranking"]}
    assert keys["A"][0] == keys["B"][0] == 10 and keys["B"][1] < keys["A"][1] and result["promoted"] == "B"


def test_level_three_fewer_differing_recordings_between_repetitions_decides_when_the_errors_tie(tmp_path):
    fresh = E1World(tmp_path)
    fresh.edit(fresh.run_ids["C"][1], lambda block, name: block.update(transcript_raw=block["transcript_raw"] + " x") if name == NAMES[0] else None)
    fresh.rescore()
    result = fresh.decide()
    keys = {r["arm"]: r["key"] for r in result["ranking"]}
    assert keys["C"][:2] == keys["A"][:2] and keys["C"][2] == 1 and keys["A"][2] == 0 and result["promoted"] == "A"


def test_level_four_the_fixed_arm_order_decides_a_full_tie(tmp_path):
    result = E1World(tmp_path).decide()
    assert [r["arm"] for r in result["ranking"]] == ["C", "A", "D", "B"] and result["promoted"] == "C"


def test_no_eligible_arm_means_no_promotion_and_exit_code_one(tmp_path):
    fresh = E1World(tmp_path)
    for ids in fresh.run_ids.values():
        (fresh.folder(ids[0]) / "wall.json").write_text(json.dumps({"wall_seconds": 3, "exit_code": 1}), encoding="utf-8")
    out = tmp_path / "report.json"
    argv = ["--manifest", str(fresh.manifest), "--runs-root", str(fresh.runs_root), "--score-root", str(fresh.score_root), "--compare-dir", str(fresh.compare_dir), "--reference-docs", str(fresh.reference_docs),
            "--out", str(out)]
    assert selection.main(argv) == 1 and json.loads(out.read_text(encoding="utf-8"))["promoted"] is None
    with pytest.raises(SystemExit, match="already exists"):
        selection.main(argv)


def test_the_report_writes_a_promoted_arm_and_the_command_returns_zero(tmp_path):
    fresh = E1World(tmp_path)
    out = tmp_path / "report.json"
    assert selection.main(["--manifest", str(fresh.manifest), "--runs-root", str(fresh.runs_root), "--score-root", str(fresh.score_root), "--compare-dir", str(fresh.compare_dir),
                           "--reference-docs", str(fresh.reference_docs), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["promoted"] == "C"


# ---- the scoping of conclusions ------------------------------------------------------------------------------------------------------------------------------------------------------

def hit_runaway(world, run_id):
    def mutate(block, name):
        if name == NAMES[0]:
            block["trace"]["windows"][0].update(start_sample=RUNAWAY[0], end_sample=RUNAWAY[1], stop_reason="max_new_tokens")
    world.edit(run_id, mutate)


def test_a_fixed_claim_needs_the_control_to_reproduce_the_runaway_here_and_no_candidate_to_hit_it(tmp_path):
    fresh = E1World(tmp_path)
    result = fresh.decide()
    assert result["stock_control_reproduces_runaway"] is False and result["claims_allowed"]["candidate_fixed_the_historical_runaway"] is False and result["historical_engine_history_reproduced"] is False
    hit_runaway(fresh, fresh.run_ids["C"][0])
    result = fresh.decide()
    assert result["stock_control_reproduces_runaway"] is True and result["claims_allowed"]["candidate_fixed_the_historical_runaway"] is True and "fresh engine" in result["claims_allowed"]["note"]
    hit_runaway(fresh, fresh.run_ids["A"][1])
    assert fresh.decide()["claims_allowed"]["candidate_fixed_the_historical_runaway"] is False


def test_a_runaway_in_the_control_makes_the_stock_arm_ineligible_but_is_still_reported(tmp_path):
    fresh = E1World(tmp_path)
    hit_runaway(fresh, fresh.run_ids["C"][0])
    result = fresh.decide()
    assert result["arms"]["C"]["eligible"] is False and result["stock_control_reproduces_runaway"] is True and result["promoted"] == "A"
