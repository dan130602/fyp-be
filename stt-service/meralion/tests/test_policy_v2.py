"""Strategy v2: the same pipeline without the invocation-wide retry budget (Stage 3 follow-up B).

CALL_064 got three retries inside the 22-call invocation and one alone, only because floor(0.15 x windows) shrinks with the batch. Synthetic recordings reproduce the effect at its smallest: a
recording of six windows has a budget of floor(0.15 x 6) = 0 when it runs alone, so v1 skips its eligible retry ("run budget") while v2 keeps it, exactly as in a larger invocation.
"""
import json

import pytest

import synthetic
from harness import raw_files, run
from meralion_worker import config as cfg
from meralion_worker import retry_selection, strategy_fg
from meralion_worker.state import ResumeError

SIX_WINDOWS = 105 * 16000                                 # 30 s windows every 15 s: six windows cover 105 s


def read(tmp_path, base, name):
    return json.loads((tmp_path / base / "out" / "transcripts" / f"{name}.json").read_text(encoding="utf-8"))


def plan_of(tmp_path, base):
    return json.loads((tmp_path / base / "state" / "inv-1" / "plan.json").read_text(encoding="utf-8"))


def world(names):
    specs = {"beta": dict(break_at=4, replace_first="half a percent"), "alpha": dict(num_samples=SIX_WINDOWS, break_at=2), "zeta": dict(num_samples=SIX_WINDOWS, break_at=2)}
    return synthetic.World({name: specs[name] for name in names})


def test_a_six_window_recording_has_a_zero_budget_alone_under_v1_but_keeps_its_retry_under_v2(tmp_path):
    assert len(strategy_fg.windows_for(SIX_WINDOWS)) == 6
    run(world(["alpha"]), tmp_path, effective=synthetic.budgeted_effective(), base="v1_alone")
    plan = plan_of(tmp_path, "v1_alone")
    assert plan["run_budget"] == 0 and plan["run_budget_bound"] is True and plan["selected"] == [] and [s["reason"] for s in plan["skipped"]] == ["run budget"]
    run(world(["alpha"]), tmp_path, base="v2_alone")
    plan = plan_of(tmp_path, "v2_alone")
    assert plan["run_budget"] is None and plan["run_budget_bound"] is False and len(plan["selected"]) == 1 and plan["skipped"] == []


def test_under_v2_a_recording_gets_the_same_retries_and_transcripts_alone_as_inside_a_batch(tmp_path):
    run(world(["beta", "zeta"]), tmp_path, base="batch")                                      # zeta sorts after beta, so its retry is numbered second in the invocation
    run(world(["zeta"]), tmp_path, base="alone")
    batch, alone = read(tmp_path, "batch", "zeta")["meralion"], read(tmp_path, "alone", "zeta")["meralion"]
    assert [d["window"] for d in batch["retries"]["decodes"]] == [d["window"] for d in alone["retries"]["decodes"]] != []
    assert [d["retry_id"] for d in batch["retries"]["decodes"]] != [d["retry_id"] for d in alone["retries"]["decodes"]]      # the ids are numbered by position in the invocation: only they differ
    assert [d["result"]["text"] for d in batch["retries"]["decodes"]] == [d["result"]["text"] for d in alone["retries"]["decodes"]]
    assert batch["transcript_raw"] == alone["transcript_raw"] and batch["transcript_final"] == alone["transcript_final"]
    assert [(o["window"], o["triggers"]) for o in batch["retries"]["outcomes"]] == [(o["window"], o["triggers"]) for o in alone["retries"]["outcomes"]]


def test_under_v1_the_same_recording_alone_loses_the_retry_it_keeps_inside_the_batch(tmp_path):
    """The reproduced defect: batch-dependent retries under the invocation-wide budget."""
    run(world(["alpha", "beta"]), tmp_path, effective=synthetic.budgeted_effective(), base="v1_batch")
    run(world(["alpha"]), tmp_path, effective=synthetic.budgeted_effective(), base="v1_alone")
    assert read(tmp_path, "v1_batch", "alpha")["meralion"]["retries"]["selected"] == 1
    assert read(tmp_path, "v1_alone", "alpha")["meralion"]["retries"]["selected"] == 0


def test_v2_equals_v1_wherever_the_budget_did_not_bind(tmp_path):
    run(synthetic.World({"alpha": dict(break_at=2), "beta": dict(break_at=4, replace_first="half a percent")}), tmp_path, effective=synthetic.budgeted_effective(), base="v1")
    run(synthetic.World({"alpha": dict(break_at=2), "beta": dict(break_at=4, replace_first="half a percent")}), tmp_path, base="v2")
    assert plan_of(tmp_path, "v1")["run_budget_bound"] is False
    for name in ("alpha", "beta"):
        a, b = read(tmp_path, "v1", name), read(tmp_path, "v2", name)
        assert a["meralion"]["transcript_raw"] == b["meralion"]["transcript_raw"] and a["results"] == b["results"]
        assert a["meralion"]["retries"] == b["meralion"]["retries"] and a["meralion"]["trace"] == b["meralion"]["trace"]
    assert [g["window"] for g in plan_of(tmp_path, "v1")["selected"]] == [g["window"] for g in plan_of(tmp_path, "v2")["selected"]]


def test_the_unresolved_per_call_cap_still_applies_under_v2():
    """Only the invocation-wide budget is gone: a call still gets at most max(1, floor(0.10 x windows)) unresolved retries."""
    bounds = strategy_fg.windows_for(22 * 15 * 16000)
    texts = synthetic.window_texts(len(bounds))
    for seam in (2, 9, 15):                                                                   # three broken seams in one recording of more than 20 windows
        texts[seam + 1] = " ".join(synthetic.WORDS[300 + 10 * (seam + 1): 310 + 10 * (seam + 1)] + texts[seam + 1].split()[10:])
    windows = [{"start_sample": a, "end_sample": b, "text": t} for (a, b), t in zip(bounds, texts)]
    pred = strategy_fg.first_merge("call", bounds[-1][1], windows)
    selection = retry_selection.select_retries({"call": pred}, kinds={"unresolved", "low_margin_conflict"}, unresolved_call_cap=0.10, run_cap=None)
    cap = max(1, int(0.10 * len(bounds)))
    unresolved = [g for g in selection.selected if any("unresolved" in t["kinds"] for t in g["triggers"])]
    assert len(bounds) > 20 and cap == 2 and len(unresolved) == 2
    assert [s["reason"] for s in selection.skipped] == ["unresolved per-call budget"]          # the third broken seam is skipped by the PER-CALL cap, not by a run budget
    assert selection.run_budget is None and selection.run_budget_bound is False


def test_run_cap_none_selects_exactly_what_an_unreachable_budget_selects_and_a_small_budget_skips_the_rest(tmp_path):
    bounds = strategy_fg.windows_for(SIX_WINDOWS)
    texts = synthetic.window_texts(len(bounds), break_at=2)
    windows = [{"start_sample": a, "end_sample": b, "text": t} for (a, b), t in zip(bounds, texts)]
    pred = {"x": strategy_fg.first_merge("x", SIX_WINDOWS, windows)}
    kinds = {"unresolved", "low_margin_conflict"}
    none = retry_selection.select_retries(pred, kinds=kinds, unresolved_call_cap=0.10, run_cap=None)
    huge = retry_selection.select_retries(pred, kinds=kinds, unresolved_call_cap=0.10, run_cap=1e9)
    small = retry_selection.select_retries(pred, kinds=kinds, unresolved_call_cap=0.10, run_cap=0.0)
    assert [g["window"] for g in none.selected] == [g["window"] for g in huge.selected] != [] and none.skipped == huge.skipped
    assert none.run_budget is None and none.run_budget_bound is False and small.run_budget == 0 and small.run_budget_bound is True and small.selected == []


def test_every_retry_group_belongs_to_a_seam_record(tmp_path):
    """Groups <= trigger candidates <= seam records (NOT <= windows - 1: the merger keeps extra seam records for excluded windows)."""
    from meralion_worker import vendored
    vendored.ensure_on_path()
    import retry_plan
    for break_at in (None, 1, 3):
        bounds = strategy_fg.windows_for(120 * 16000)
        windows = [{"start_sample": a, "end_sample": b, "text": t} for (a, b), t in zip(bounds, synthetic.window_texts(len(bounds), break_at=break_at))]
        pred = strategy_fg.first_merge("x", 120 * 16000, windows)
        found = retry_plan.plan_call(pred, "worker", {"unresolved", "low_margin_conflict"})
        groups = retry_plan.group_triggers(found)
        assert len(groups) <= len(found) <= len(pred["seams"])


@pytest.mark.parametrize("first, second", [(None, "budgeted"), ("budgeted", None), ("legacy", None)])
def test_a_state_started_under_one_strategy_cannot_be_resumed_under_another(tmp_path, first, second):
    pick = {None: synthetic.effective, "budgeted": synthetic.budgeted_effective, "legacy": synthetic.legacy_effective}
    w = synthetic.World({"alpha": dict(break_at=2)})
    run(w, tmp_path, effective=pick[first]())
    before = raw_files(tmp_path / "run")
    with pytest.raises(ResumeError, match="incompatible resume"):
        run(w, tmp_path, effective=pick[second]())
    assert raw_files(tmp_path / "run") == before


def test_the_default_config_loads_the_v2_strategy_and_the_budgeted_one_stays_selectable():
    assert synthetic.effective().data["strategy"]["id"] == "fg-m2ax-seq-ap-lmonly-v2" and synthetic.budgeted_effective().data["strategy"]["id"] == "fg-m2ax-seq-ap-lmonly-v1"
    assert synthetic.effective().fingerprint != synthetic.budgeted_effective().fingerprint
    assert cfg.load_runtime_config(cfg.BUDGETED_CONFIG, env={}).data["config_id"].endswith("fee2-amt1")


def test_a_document_written_without_a_budget_validates_and_records_null(tmp_path):
    from meralion_worker import output
    run(world(["alpha"]), tmp_path, base="doc")
    document = read(tmp_path, "doc", "alpha")
    output.validate_document(document)
    assert document["meralion"]["invocation"]["run_budget"] is None and document["meralion"]["invocation"]["run_budget_bound"] is False
    assert document["meralion"]["config"]["strategy"]["run_cap"] is None


def test_the_dockerfile_carries_no_default_configuration_of_its_own():
    """Stage 4: the image's configuration comes from the RUNTIME_CONFIG build argument (no default), never from a default baked into the Dockerfile: an image build cannot promote or change the code's default."""
    import re
    from pathlib import Path
    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "MERALION_CONFIG=/opt/meralion/configs/runtime/${RUNTIME_CONFIG}" in dockerfile
    assert cfg.DEFAULT_CONFIG.name not in dockerfile and re.search(r"^ARG RUNTIME_CONFIG$", dockerfile, re.M)       # no config file is named, and the argument has no default value
    assert cfg.DEFAULT_CONFIG.is_file()                                                                          # the code's own default remains a file of the configs tree the image copies
