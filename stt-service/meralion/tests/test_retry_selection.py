"""Retry selection with hand-built seam records (no merger involved): grouping, order, the two budgets and their boundaries."""
import math

import pytest

from meralion_worker.retry_selection import select_retries

RATE = 16000
KINDS = {"unresolved", "low_margin_conflict"}


def pred(call, n_windows, seams):
    """n_windows FG windows (30 s every 15 s). seams: {seam_index: dict of extra seam fields}; a seam index not listed is a plain anchored seam."""
    windows = [{"start_sample": i * 15 * RATE, "end_sample": i * 15 * RATE + 30 * RATE, "text": f"w{i}"} for i in range(n_windows)]
    seam_records = [{"seam": k, "windows": [k, k + 1], "status": "anchored", "replaced": [], **seams.get(k, {})} for k in range(n_windows - 1)]
    return {"call_id": call, "audio_seconds": (n_windows + 1) * 15, "windows": windows, "seams": seam_records}


UNRESOLVED = {"status": "unresolved"}
LOW_MARGIN = {"replaced": [{"edge_margin": 0.1, "left_text": "a", "right_text": "b"}]}
CONFIDENT = {"replaced": [{"edge_margin": 0.9, "left_text": "a", "right_text": "b"}]}


def select(preds, run_cap=0.15, call_cap=0.10):
    return select_retries({p["call_id"]: p for p in preds}, kinds=KINDS, unresolved_call_cap=call_cap, run_cap=run_cap)


def test_no_triggers_no_retries():
    result = select([pred("a", 10, {})])
    assert result.selected == [] and result.skipped == [] and result.groups_total == 0 and result.windows_total == 10 and not result.run_budget_bound


def test_confident_conflicts_are_not_triggers():
    assert select([pred("a", 10, {2: CONFIDENT})]).selected == []


def test_an_unresolved_seam_and_a_low_margin_conflict_each_make_a_group():
    result = select([pred("a", 20, {2: UNRESOLVED, 9: LOW_MARGIN})])
    assert len(result.selected) == 2
    kinds = {frozenset(k for t in g["triggers"] for k in t["kinds"]) for g in result.selected}
    assert kinds == {frozenset({"unresolved"}), frozenset({"low_margin_conflict"})}


def test_the_retry_window_is_one_30_s_window_centred_on_the_seam_overlap():
    """Seam 2 joins windows 2 (30-60 s) and 3 (45-75 s): the exact overlap is [45 s, 60 s], its centre 52.5 s, so the retry window is 37.5 s to 67.5 s."""
    result = select([pred("a", 20, {2: UNRESOLVED})])
    group = result.selected[0]
    assert list(group["window"]) == [int(37.5 * RATE), int(67.5 * RATE)] and group["window"][1] - group["window"][0] == 30 * RATE
    assert group["triggers"][0]["overlap"] == [45 * RATE, 60 * RATE] and group["triggers"][0]["covered_fraction"] == 1.0


def test_groups_are_ordered_by_priority_then_call_then_window_start():
    result = select([pred("b", 30, {3: UNRESOLVED, 20: LOW_MARGIN}), pred("a", 30, {10: UNRESOLVED, 4: LOW_MARGIN})])
    order = [(g["priority"], g["call_id"], g["window"][0]) for g in result.selected]
    assert order == sorted(order) and [p for p, _, _ in order] == [1, 1, 5, 5]
    assert [c for _, c, _ in order] == ["a", "b", "a", "b"]


@pytest.mark.parametrize("windows, cap", [(9, 1), (10, 1), (19, 1), (20, 2), (29, 2), (30, 3)])
def test_the_unresolved_per_call_cap_is_max_one_floor_ten_percent(windows, cap):
    assert max(1, math.floor(0.10 * windows)) == cap
    many = {k: UNRESOLVED for k in range(0, windows - 1, 2)}                  # alternate seams so the retry windows do not group together
    result = select([pred("a", windows, many)], run_cap=1.0)
    selected = [g for g in result.selected if any("unresolved" in t["kinds"] for t in g["triggers"])]
    assert len(selected) == min(cap, result.groups_total)
    assert all(s["reason"] == "unresolved per-call budget" for s in result.skipped)
    assert len(result.skipped) == result.groups_total - len(selected)


def test_low_margin_groups_have_no_per_call_cap():
    seams = {k: LOW_MARGIN for k in range(0, 38, 3)}                           # 13 well separated conflicts in one 40-window call
    result = select([pred("a", 40, seams)], run_cap=1.0)
    assert len(result.selected) == len(seams) == 13 and result.skipped == []


def test_the_run_budget_is_floor_fifteen_percent_of_the_windows_of_the_invocation():
    result = select([pred("a", 10, {}), pred("b", 10, {})])
    assert result.windows_total == 20 and result.run_budget == 3


def test_the_run_budget_boundary_skips_the_fourth_group_with_the_run_budget_reason():
    """20 windows in total give a budget of 3; four low-margin groups exist, so the fourth is skipped."""
    seams = {k: LOW_MARGIN for k in (1, 4, 7, 10)}
    result = select([pred("a", 20, seams)])
    assert result.run_budget == 3 and len(result.selected) == 3 and result.run_budget_bound is True
    assert [s["reason"] for s in result.skipped] == ["run budget"]
    assert result.skipped[0]["window"][0] > result.selected[-1]["window"][0]                  # later windows are the ones dropped


def test_the_budget_is_shared_across_the_recordings_of_the_invocation():
    seams = {k: LOW_MARGIN for k in (1, 4, 7)}
    alone = select([pred("a", 20, seams)])
    shared = select([pred("a", 20, seams), pred("b", 200, {})])                  # many more windows in the invocation: a bigger budget for "a"
    assert alone.run_budget == 3 and shared.run_budget == 33
    assert len(alone.selected) == len(shared.selected) == 3
    tight = select([pred("a", 20, seams), pred("b", 20, {2: LOW_MARGIN, 8: LOW_MARGIN})])
    assert tight.run_budget == 6 and len(tight.selected) == 5 and not tight.run_budget_bound
    squeezed = select([pred("a", 7, {1: LOW_MARGIN, 3: LOW_MARGIN}), pred("b", 7, {1: LOW_MARGIN})])
    assert squeezed.run_budget == 2 and len(squeezed.selected) == 2 and squeezed.run_budget_bound and squeezed.skipped[0]["reason"] == "run budget"


def test_a_budget_of_zero_skips_everything_with_the_run_budget_reason():
    result = select([pred("a", 6, {1: UNRESOLVED})])
    assert result.run_budget == 0 and result.selected == [] and result.skipped[0]["reason"] == "run budget" and result.run_budget_bound


def test_the_priority_order_decides_who_survives_a_binding_budget():
    result = select([pred("a", 14, {1: LOW_MARGIN, 8: UNRESOLVED})])             # budget floor(0.15 x 14) = 2: both fit
    assert result.run_budget == 2 and len(result.selected) == 2
    result = select([pred("a", 7, {1: LOW_MARGIN, 4: UNRESOLVED})])              # budget 1: the unresolved group (priority 1) beats the earlier low-margin one (priority 5)
    assert [t["kinds"] for t in result.selected[0]["triggers"]] == [["unresolved"]]
    assert result.skipped[0]["reason"] == "run budget"


def test_a_group_with_an_unresolved_and_a_low_margin_trigger_counts_as_unresolved():
    seams = {3: {"status": "unresolved", **LOW_MARGIN}}
    result = select([pred("a", 20, seams)])
    assert len(result.selected) == 1 and result.selected[0]["priority"] == 1


def test_an_excluded_window_record_still_triggers():
    seams = {2: {"status": "unresolved", "excluded": True}}
    assert len(select([pred("a", 20, seams)]).selected) == 1
