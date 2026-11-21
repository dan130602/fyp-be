"""The strategy on the synthetic fixtures (tests/synthetic.py): windows, first merge, retry insertion, and the properties the pipeline relies on."""
import pytest

import synthetic
from meralion_worker import strategy_fg as S
from meralion_worker.retry_selection import select_retries
from meralion_worker.formatters import apply_formatter

N = synthetic.SAMPLES_120S


def pred_for(texts, n=N, job="job"):
    bounds = S.windows_for(n)
    return S.first_merge(job, n, [{"start_sample": a, "end_sample": b, "text": t} for (a, b), t in zip(bounds, texts)])


def select(pred, run_cap=0.15):
    return select_retries({pred["call_id"]: pred}, kinds={"unresolved", "low_margin_conflict"}, unresolved_call_cap=0.10, run_cap=run_cap)


def test_a_120_second_recording_has_seven_fg_windows_every_15_seconds():
    bounds = S.windows_for(N)
    assert len(bounds) == 7 and bounds[0] == (0, 480000) and bounds[1] == (240000, 720000) and bounds[-1] == (1440000, 1920000)
    assert all(b - a <= 480000 for a, b in bounds)


def test_a_recording_under_30_seconds_has_one_window_no_seams_and_no_retries():
    n = 20 * 16000
    assert S.windows_for(n) == [(0, n)]
    pred = S.first_merge("short", n, [{"start_sample": 0, "end_sample": n, "text": "just one window of text"}])
    assert pred["seams"] == [] and pred["transcript"] == "just one window of text"
    assert select(pred).selected == []
    out = S.reconcile(pred, [])
    assert out["transcript_raw"] == "just one window of text" and out["counts"] == {"inserted": 0, "adopted": 0, "failed": 0}


def test_consistent_overlapping_windows_merge_without_triggers():
    pred = pred_for(synthetic.window_texts(7))
    assert [s["status"] for s in pred["seams"]] == ["anchored"] * 6
    assert select(pred).selected == []
    assert S.reconcile(pred, [])["transcript_raw"] == pred["transcript"] == " ".join(synthetic.WORDS[:80])      # 7 windows cover words 0..79 exactly once


def test_the_broken_seam_is_unresolved_and_is_the_only_trigger():
    """The fixture's contract: exactly one unresolved seam (between windows 2 and 3), one retry planned for it, and a budget of exactly one."""
    pred = pred_for(synthetic.window_texts(7, break_at=2))
    statuses = [s["status"] for s in pred["seams"]]
    assert statuses == ["anchored", "anchored", "unresolved", "anchored", "anchored", "anchored"]
    selection = select(pred)
    assert selection.windows_total == 7 and selection.run_budget == 1 and not selection.run_budget_bound
    assert len(selection.selected) == 1 and list(selection.selected[0]["window"]) == [600000, 1080000]
    assert [t["kinds"] for t in selection.selected[0]["triggers"]] == [["unresolved"]]


def retries_for(pred, texts, retry_text):
    group = select(pred).selected[0]
    return [{"retry_id": "job_r000", "window": list(group["window"]), "triggers": group["triggers"], "text": retry_text, "tokens": None, "route": None}]


def test_a_retry_that_anchors_to_both_neighbours_is_inserted():
    texts = synthetic.window_texts(7, break_at=2)
    pred = pred_for(texts)
    out = S.reconcile(pred, retries_for(pred, texts, synthetic.retry_text_for(texts, 2)))
    assert out["counts"] == {"inserted": 1, "adopted": 0, "failed": 0}
    assert [(d["kind"], d["result"]) for o in out["outcomes"] for d in o["decisions"]] == [("insert", "inserted")]
    assert any(c["retry"] for c in out["chain"]) and len(out["chain"]) == 8
    assert out["seams"] != pred["seams"] and out["provenance"] is not pred["provenance"]


def test_a_retry_that_cannot_anchor_is_recorded_as_failed_and_the_transcript_is_unchanged():
    texts = synthetic.window_texts(7, break_at=2)
    pred = pred_for(texts)
    out = S.reconcile(pred, retries_for(pred, texts, "completely different words that match nothing at all anywhere here"))
    assert out["counts"] == {"inserted": 0, "adopted": 0, "failed": 1}
    assert out["transcript_raw"] == pred["transcript"]


def test_reconcile_returns_the_raw_unformatted_text_and_formatting_is_a_separate_step():
    texts = synthetic.window_texts(7, break_at=2, replace_first="half a percent")
    pred = pred_for(texts)
    assert "half a percent" in S.reconcile(pred, [])["transcript_raw"]
    assert "half a percent" not in apply_formatter("percentages-v2", S.reconcile(pred, [])["transcript_raw"])
    assert "0.5 percent" in apply_formatter("percentages-v2", S.reconcile(pred, [])["transcript_raw"])
    assert apply_formatter("none", S.reconcile(pred, [])["transcript_raw"]) == S.reconcile(pred, [])["transcript_raw"]


def test_the_first_merge_is_deterministic():
    texts = synthetic.window_texts(7, break_at=2)
    assert pred_for(texts)["transcript"] == pred_for(texts)["transcript"]
    assert pred_for(texts)["seams"] == pred_for(texts)["seams"]


def test_the_policy_and_vendored_modules_are_the_frozen_ones():
    assert S.POLICY == "m2ax_seq_ap_lm"
    assert S.overlap.MERGE_VERSION == "2.3"                                                    # the frozen blob (2f982b2), not the later v2.4
    assert "m2ax_seq_ap_lm" in S.overlap.POLICIES
    import inspect
    assert list(inspect.signature(S.retry_merge.reconcile_call).parameters)[:7] == ["pred", "retries", "policy", "m4_choices", "conflict_tiebreak", "low_margin_tiebreak", "critical_only_skip"]


def test_the_critical_content_exclusion_is_on_and_value_tiebreaking_is_off():
    """reconcile() must call reconcile_call with conflict_tiebreak=False, low_margin_tiebreak=True, critical_only_skip=True (the frozen strategy)."""
    seen = {}
    original = S.retry_merge.reconcile_call

    def spy(pred, retries, policy, **kwargs):
        seen.update(kwargs, policy=policy)
        return original(pred, retries, policy, **kwargs)

    S.retry_merge.reconcile_call = spy
    try:
        texts = synthetic.window_texts(7, break_at=2)
        pred = pred_for(texts)
        S.reconcile(pred, retries_for(pred, texts, synthetic.retry_text_for(texts, 2)))
    finally:
        S.retry_merge.reconcile_call = original
    assert seen == {"policy": "m2ax_seq_ap_lm", "m4_choices": None, "conflict_tiebreak": False, "low_margin_tiebreak": True, "critical_only_skip": True}
