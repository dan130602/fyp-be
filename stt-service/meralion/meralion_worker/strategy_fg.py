"""The processing strategy fg-m2ax-seq-ap-lmonly-v1 as pure functions (no I/O, no decoder): windows, first merge, reconciliation of retries.

    FG windows (30 s every 15 s)  ->  first merge, policy m2ax_seq_ap_lm (its own exclusions, provenance)  ->  retry selection (retry_selection.py)
    ->  reconcile: unresolved retries inserted only if they anchor to both neighbours; low-margin regions decided by a retry only between the two EXISTING versions, never when the region
        involves a number, a negation or an English consent word; value/negation tie-breaking off.

Follows handoff section 11 with the vendored modules (git blobs from commit 2f982b2) imported as `overlap` and `retry_merge`.
"""
from meralion_worker import vendored

vendored.ensure_on_path()
import overlap                                                   # noqa: E402
import retry_merge                                               # noqa: E402

POLICY = "m2ax_seq_ap_lm"
RATE = overlap.RATE


def windows_for(num_samples):
    return [(int(a), int(b)) for a, b in overlap.plan_windows(num_samples)]


def first_merge(job_name, num_samples, windows):
    """windows: [{start_sample, end_sample, text}] in order. Returns the pred dict the retry planner and the reconciler read."""
    texts = [w["text"] for w in windows]
    bounds = [(w["start_sample"], w["end_sample"]) for w in windows]
    merged, seams, provenance = overlap.merge_windows(texts, expected_overlap_share=overlap.seam_shares(bounds), policy=POLICY, bounds=bounds, provenance=True)
    return {"call_id": job_name, "audio_seconds": num_samples / RATE, "windows": [dict(w) for w in windows], "seams": seams, "provenance": provenance, "transcript": merged}


def reconcile(pred, retries):
    """retries: [{retry_id, window, triggers, text, tokens, route}] sorted by window start. Returns the raw (unformatted) final transcript and the records."""
    if not retries:
        return {"transcript_raw": pred["transcript"], "chain": None, "seams": pred["seams"], "provenance": pred["provenance"], "outcomes": [], "counts": {"inserted": 0, "adopted": 0, "failed": 0}}
    merged, chain, seams, provenance, outcomes, counts = retry_merge.reconcile_call(
        pred, retries, POLICY, m4_choices=None, conflict_tiebreak=False, low_margin_tiebreak=True, critical_only_skip=True)
    failed = sum(1 for outcome in outcomes for decision in outcome["decisions"] if decision["result"] == "retry_failed")
    return {"transcript_raw": merged, "chain": [{"label": c["label"], "bounds": list(c["bounds"]), "retry": c["retry"], "excluded": c["excluded"]} for c in chain],
            "seams": seams, "provenance": provenance, "outcomes": outcomes, "counts": {"inserted": counts["inserted"], "adopted": counts["adopted"], "failed": failed}}
