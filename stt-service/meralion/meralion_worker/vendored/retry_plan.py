"""M5 retry plan: which shifted 30 s windows to decode again, chosen ONLY from model outputs and merge records (no reference text, no scores, no call conditions).

    python -X utf8 retry_plan.py --merged <folder of merged predictions> [--merged <another>] --out retry_plan.json

Triggers (seam records of the merged predictions; an `excluded` record of a window the merger dropped still counts), in priority order:
  1 unresolved seam                       2 cross_language_divergence flag
  3 value or negation conflict            4 probable-duplicate flag
Placement: one 30 s window centred on the seam's exact overlap interval [start of the later window, end of the earlier one], clipped to the recording (for the FG grid: [15k + 7.5 s, 15k + 37.5 s]).
Deduplication (every retry stays ONE window of at most 30 s, intervals are never combined): triggers whose candidate windows overlap by more than 50 % form a group; the group's
representative is the candidate window that covers the most of the group's trigger overlap intervals (ties: highest-priority trigger, then earliest). Every trigger record is kept with the
fraction of its overlap the representative covers; a trigger the representative does not fully cover gets its own window in the next round.
Budgets: per call at most max(1, floor(0.10 x the call's windows)); per run at most 15 % of all windows, applied to the SHARED pool of every --merged folder (the union of their triggers is
decoded once); when a budget binds, groups are taken by priority, then by call and seam order, and the skipped ones are listed. A retry is a third prediction from the same model: correlated
evidence, never proof. Triggers cannot see an omission that every window shares.
"""
import argparse
import json
import math
import sys
from pathlib import Path

RATE = 16000
WINDOW = 30 * RATE
LOW_MARGIN = 0.3        # must equal overlap.LOW_MARGIN: the edge-position heuristic's decision at a disputed region is not informative below this margin
PRIORITY = {"unresolved": 1, "cross_language_divergence": 2, "value_or_negation_conflict": 3, "probable_duplicate": 4, "low_margin_conflict": 5}


def triggers_of(seam):
    out = []
    if seam.get("status") == "unresolved":
        out.append("unresolved")
    if "cross_language_divergence" in seam.get("risk_flags", []):
        out.append("cross_language_divergence")
    if seam.get("numeric_conflicts", 0) or seam.get("negation_conflicts", 0) or any(r.get("value_conflict") for r in seam.get("replaced", [])):
        out.append("value_or_negation_conflict")
    if seam.get("probable_duplicate_passage") or "probable_duplicate_passage" in seam.get("risk_flags", []) or seam.get("duplicates"):
        out.append("probable_duplicate")
    if any(abs(r.get("edge_margin", 9)) < LOW_MARGIN for r in seam.get("replaced", [])):
        out.append("low_margin_conflict")
    return out


def overlap_of(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def plan_call(pred, label, only_kinds=None):
    """Candidate triggers of one merged prediction: dicts with the seam, the trigger kinds, the exact overlap interval and the candidate 30 s window."""
    wins = pred["windows"]
    total = round(pred["audio_seconds"] * RATE)
    found = []
    for seam in pred["seams"]:
        kinds = [k for k in triggers_of(seam) if only_kinds is None or k in only_kinds]
        if not kinds or "windows" not in seam:
            continue
        k, m = seam["windows"]
        if k >= len(wins) or m >= len(wins):
            continue
        lo, hi = wins[m]["start_sample"], wins[k]["end_sample"]
        if hi <= lo:                                              # excluded-window records: the neighbours abut, the overlap of the pair is empty
            lo, hi = wins[k]["start_sample"], wins[m]["end_sample"]
        centre = (lo + hi) // 2
        start = max(0, centre - WINDOW // 2)
        end = min(total, start + WINDOW) if start == 0 else min(total, centre + WINDOW // 2)
        found.append({"call_id": pred["call_id"], "seam": seam["seam"], "windows": [k, m], "kinds": kinds, "priority": min(PRIORITY[x] for x in kinds), "overlap": [lo, hi],
                      "candidate": [start, end], "excluded_record": bool(seam.get("excluded")), "from": label})
    return found


def group_triggers(found):
    """Rounds of grouping (see the module docstring). Returns groups: {representative: [start, end], triggers: [...with coverage]}."""
    pending = sorted(found, key=lambda t: (t["priority"], t["call_id"], t["seam"]))
    groups = []
    while pending:
        head = pending[0]
        window = head["candidate"]
        members = [t for t in pending if t["call_id"] == head["call_id"] and overlap_of(t["candidate"], window) > 0.5 * (window[1] - window[0])]
        best, best_cov = None, -1.0
        for candidate in [t["candidate"] for t in members]:
            cov = sum(overlap_of(t["overlap"], candidate) / max(1, t["overlap"][1] - t["overlap"][0]) for t in members)
            if cov > best_cov + 1e-9:
                best, best_cov = candidate, cov
        if overlap_of(head["overlap"], best) < head["overlap"][1] - head["overlap"][0]:
            best = head["candidate"]                                  # the head trigger must be fully covered by its own group's window
        covered = []
        for t in members:
            fraction = overlap_of(t["overlap"], best) / max(1, t["overlap"][1] - t["overlap"][0])
            if fraction >= 0.999 or t is head:
                covered.append(dict(t, covered_fraction=round(fraction, 3)))
        ids = {id(t) for t in members if any(c["seam"] == t["seam"] and c["call_id"] == t["call_id"] and c["from"] == t["from"] for c in covered)}
        groups.append({"call_id": head["call_id"], "window": best, "priority": min(c["priority"] for c in covered), "triggers": covered})
        pending = [t for t in pending if id(t) not in ids]
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-kinds", nargs="+", choices=sorted(PRIORITY), help="plan only these trigger kinds (the clean-175 configuration uses: unresolved)")
    parser.add_argument("--run-cap", type=float, default=0.15)
    parser.add_argument("--call-cap", type=float, default=0.10)
    args = parser.parse_args(argv)
    found, windows_per_call = [], {}
    for folder in args.merged:
        for path in sorted(folder.glob("CALL_*.json")):
            pred = json.loads(path.read_text(encoding="utf-8"))
            windows_per_call[pred["call_id"]] = max(windows_per_call.get(pred["call_id"], 0), len(pred["windows"]))
            found += plan_call(pred, folder.name, set(args.only_kinds) if args.only_kinds else None)
    groups = group_triggers(found)
    groups.sort(key=lambda g: (g["priority"], g["call_id"], g["window"][0]))
    run_cap = math.floor(args.run_cap * sum(windows_per_call.values()))
    per_call, selected, skipped = {}, [], []
    for g in groups:
        cap = max(1, math.floor(args.call_cap * windows_per_call[g["call_id"]]))
        if len(selected) >= run_cap:
            skipped.append(dict(g, reason="run budget"))
        elif per_call.get(g["call_id"], 0) >= cap:
            skipped.append(dict(g, reason="call budget"))
        else:
            per_call[g["call_id"]] = per_call.get(g["call_id"], 0) + 1
            selected.append(dict(g, retry_id=f"{g['call_id']}_r{len(selected):03d}"))
    out = {"version": "retry-plan-v1", "run_budget": run_cap, "windows_total": sum(windows_per_call.values()), "merged_folders": [str(f) for f in args.merged], "candidate_triggers": len(found),
           "groups": len(groups), "selected": selected, "skipped": skipped,
           "by_kind": {k: sum(1 for t in found if k in t["kinds"]) for k in PRIORITY},
           "call_retry_rates": {c: {"retries": per_call.get(c, 0), "windows": windows_per_call[c], "rate": round(per_call.get(c, 0) / windows_per_call[c], 3)} for c in sorted(windows_per_call)}}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"triggers {len(found)} {out['by_kind']} -> groups {len(groups)} -> selected {len(selected)} (run budget {run_cap}), skipped {len(skipped)}: {dict((r, sum(1 for s in skipped if s['reason']==r)) for r in ('run budget', 'call budget'))}")
    print("retry rate overall:", round(len(selected) / out["windows_total"], 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
