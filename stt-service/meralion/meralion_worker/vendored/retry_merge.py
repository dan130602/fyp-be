"""M5 reconciliation (offline): add the retry windows to a merged prediction under the frozen rules, keeping every raw prediction and every decision.

    python -X utf8 retry_merge.py --merged <base merged folder> --retries <out-fg1/decodes> --plan retry_plan.json --dest <folder> [--policy m2ax]

No reference text is read. For every retry window (a third prediction from the same model: correlated evidence, never verification):
  * UNRESOLVED seam (and the divergence / duplicate triggers): the retry is INSERTED into the chain between the seam's neighbours only if it anchors to both of them under the merger's own
    seam rule (status `anchored` for both new seams). Otherwise the seam keeps its earlier outcome and is recorded `retry_failed`.
  * VALUE or NEGATION conflict: the retry is a tie-breaker between the two EXISTING versions. A version is adopted only if its units, with one unit of context on each side taken from the
    window that wrote it, occur in the retry text, and the other version's do not. Support for both or neither is `retry_inconclusive` / `retry_agrees`: the earlier choice stands. A third
    value is never adopted, and the retry text is never merged into the conflict.
No recursion. Excluded windows of the base merge stay excluded. Windows are never combined or rewritten; the output keeps every window (and the retries, marked) so the raw predictions stay recoverable.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlap as o                                      # noqa: E402


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def unit_runs(units, run):
    """Start indexes where the unit list `run` occurs contiguously inside `units` (value+currency identity under the active policy)."""
    n = len(run)
    return [i for i in range(len(units) - n + 1) if all(o.same_unit(units[i + j], run[j]) for j in range(n))]


def supports(candidate_text, own_text, retry_units):
    """Does the retry text contain the candidate with the context its own window gave it? Context = one unit before and after the first occurrence in the window that wrote it."""
    cand = o.units_of(o.tokens(candidate_text))
    own = o.units_of(o.tokens(own_text))
    if not cand:
        return False, "empty"
    for i in unit_runs(own, cand):
        before, after = own[i - 1:i], own[i + len(cand):i + len(cand) + 1]
        pattern = before + cand + after
        if unit_runs(retry_units, pattern):
            return True, "with_context"
    return False, "not_found_with_context"


def reconcile_call(pred, retries, policy, m4_choices=None, conflict_tiebreak=True, low_margin_tiebreak=False, critical_only_skip=False):
    windows = pred["windows"]
    texts = [w["text"] for w in windows]
    bounds = [(w["start_sample"], w["end_sample"]) for w in windows]
    excluded = {w["window"] for w in pred.get("provenance", {}).get("windows", []) if w.get("excluded")}
    report, inserted, choices = [], [], dict(m4_choices or {})          # M4 choices first; an M5 tie-break on the same conflict overrides them
    saved_policy, o._ACTIVE = o._ACTIVE, o.POLICIES[policy]
    try:
        for r in retries:
            kinds = {k for t in r["triggers"] for k in t["kinds"]}
            outcome = {"retry_id": r["retry_id"], "window": r["window"], "triggers": [(t["seam"], t["kinds"], t["covered_fraction"]) for t in r["triggers"]], "decisions": []}
            if kinds & {"unresolved", "cross_language_divergence", "probable_duplicate"}:
                lo = min(w for t in r["triggers"] for w in t["windows"])
                hi = max(w for t in r["triggers"] for w in t["windows"])
                left = max((k for k in range(lo + 1) if k not in excluded), default=None)
                right = min((k for k in range(hi, len(windows)) if k not in excluded), default=None)
                if left is None or right is None or left == right:
                    outcome["decisions"].append({"kind": "insert", "result": "retry_failed", "why": "no neighbour on one side"})
                else:
                    tb = [bounds[left], tuple(r["window"]), bounds[right]]
                    shares = o.seam_shares(tb)
                    _, trial = o.merge_windows([texts[left], r["text"], texts[right]], expected_overlap_share=shares, policy=policy, skip=set(), bounds=tb)
                    ok = len(trial) == 2 and all(s["status"] == "anchored" for s in trial)
                    outcome["decisions"].append({"kind": "insert", "between": [left, right], "result": "inserted" if ok else "retry_failed",
                                                 "trial_statuses": [s["status"] for s in trial]})
                    if ok:
                        inserted.append(r)
            if "value_or_negation_conflict" in kinds and conflict_tiebreak:
                retry_units = o.units_of(o.tokens(r["text"]))
                for t in r["triggers"]:
                    if "value_or_negation_conflict" not in t["kinds"]:
                        continue
                    k, m = t["windows"]
                    seam = next((s for s in pred["seams"] if s.get("windows") == [k, m] and not s.get("excluded")), None)
                    for region in (seam or {}).get("replaced", []):
                        if not region.get("value_conflict"):
                            continue
                        left_ok, left_why = supports(region["left_text"], texts[k], retry_units)
                        right_ok, right_why = supports(region["right_text"], texts[m], retry_units)
                        key = (f"w{k}", f"w{m}", " ".join(t_.norm for t_ in o.tokens(region["left_text"])), " ".join(t_.norm for t_ in o.tokens(region["right_text"])))
                        if left_ok != right_ok:
                            choices[key] = "left" if left_ok else "right"
                            result = "adopted_" + choices[key]
                        else:
                            result = "retry_agrees" if left_ok and right_ok else "retry_inconclusive"
                        outcome["decisions"].append({"kind": "conflict", "seam": [k, m], "left": region["left_text"].strip(), "right": region["right_text"].strip(), "earlier_choice": region["kept"],
                                                     "retry_supports_left": left_ok, "retry_supports_right": right_ok, "result": result})
            if "low_margin_conflict" in kinds and low_margin_tiebreak:
                # a disputed region whose edge-position decision was a coin toss (|margin| < LOW_MARGIN): the retry may pick between the two EXISTING versions, identified by (seam, ordinal), never by their text
                retry_units = o.units_of(o.tokens(r["text"]))
                for t in r["triggers"]:
                    if "low_margin_conflict" not in t["kinds"]:
                        continue
                    k, m = t["windows"]
                    seam = next((s for s in pred["seams"] if s.get("windows") == [k, m] and not s.get("excluded")), None)
                    for ordinal, region in enumerate((seam or {}).get("replaced", [])):
                        if abs(region.get("edge_margin", 9)) >= o.LOW_MARGIN:
                            continue
                        if critical_only_skip and (region.get("number_involved") or region.get("negation_involved") or set((region["left_text"] + " " + region["right_text"]).lower().replace(",", " ").replace(".", " ").split()) & o.CONSENT_WORDS):
                            outcome["decisions"].append({"kind": "low_margin_conflict", "seam": [k, m], "ordinal": ordinal, "edge_margin": region["edge_margin"], "left": region["left_text"].strip(), "right": region["right_text"].strip(),
                                                         "earlier_choice": region["kept"], "result": "not_decided_critical_content", "retry_supports_left": None, "retry_supports_right": None})
                            continue                          # a retry never decides a value, negation or consent conflict (the standing rule of the M5u baseline)
                        left_ok, left_why = supports(region["left_text"], texts[k], retry_units)
                        right_ok, right_why = supports(region["right_text"], texts[m], retry_units)
                        if left_ok != right_ok:
                            choices[(f"w{k}", f"w{m}", "region", ordinal)] = "left" if left_ok else "right"
                            result = "adopted_" + ("left" if left_ok else "right")
                        else:
                            result = "retry_agrees" if left_ok and right_ok else "retry_inconclusive"
                        outcome["decisions"].append({"kind": "low_margin_conflict", "seam": [k, m], "ordinal": ordinal, "edge_margin": region["edge_margin"], "left": region["left_text"].strip(), "right": region["right_text"].strip(),
                                                     "earlier_choice": region["kept"], "retry_supports_left": left_ok, "retry_supports_right": right_ok, "result": result})
            report.append(outcome)
    finally:
        o._ACTIVE = saved_policy
    chain = [{"label": f"w{k}", "text": texts[k], "bounds": bounds[k], "retry": False, "excluded": k in excluded} for k in range(len(windows))]
    chain += [{"label": r["retry_id"], "text": r["text"], "bounds": tuple(r["window"]), "retry": True, "excluded": False} for r in inserted]
    chain.sort(key=lambda c: (c["bounds"][0], c["bounds"][1]))
    skip = {i for i, c in enumerate(chain) if c["excluded"]}
    cb = [c["bounds"] for c in chain]
    merged, seams, prov = o.merge_windows([c["text"] for c in chain], expected_overlap_share=o.seam_shares(cb), policy=policy, bounds=cb, provenance=True, skip=skip,
                                          labels=[c["label"] for c in chain], choices=choices)
    adopted = sum(1 for r in report for d in r["decisions"] if d["result"].startswith("adopted"))
    return merged, chain, seams, prov, report, {"inserted": len(inserted), "adopted": adopted}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--retries", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--policy", default="m2ax")
    parser.add_argument("--no-conflict-tiebreak", action="store_true", help="ablation: keep the unresolved-seam insertion but never let a retry decide a value/negation conflict (the base merger's decisions stand)")
    parser.add_argument("--low-margin-tiebreak", action="store_true", help="v2.3 experiment: retries of kind low_margin_conflict may decide disputed regions with |edge margin| < LOW_MARGIN (needs a policy with low_margin_choices)")
    parser.add_argument("--exclude-critical-from-low-margin", action="store_true", help="a retry never decides a low-margin region that involves a number, negation or consent word (kept as the merger chose)")
    parser.add_argument("--m4-choices", type=Path, help="_m4_choices.json from m4_apply.py: applied first, then M5 tie-breaks override them")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.dest.mkdir(parents=True, exist_ok=True)
    by_call, missing = {}, []
    for g in plan["selected"]:
        path = args.retries / f"{g['retry_id']}.json"
        if not path.exists():
            missing.append(g["retry_id"])
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        by_call.setdefault(g["call_id"], []).append({"retry_id": g["retry_id"], "window": g["window"], "triggers": g["triggers"], "text": d["text"], "tokens": d["tokens"], "route": d["termination"]["route"]})
    m4 = {}
    if args.m4_choices:
        for c in json.loads(args.m4_choices.read_text(encoding="utf-8")):
            m4.setdefault(c["call_id"], {})[tuple(c["key"])] = c["side"]
    totals = {"calls": 0, "retries": 0, "inserted": 0, "adopted": 0, "failed": 0, "conflict_decisions": {}}
    for path in sorted(args.merged.glob("CALL_*.json")):
        pred = json.loads(path.read_text(encoding="utf-8"))
        retries = sorted(by_call.get(pred["call_id"], []), key=lambda r: r["window"][0])
        merged, chain, seams, prov, report, counts = reconcile_call(pred, retries, args.policy, m4.get(pred["call_id"]), not args.no_conflict_tiebreak, args.low_margin_tiebreak, args.exclude_critical_from_low_margin) if (retries or m4.get(pred["call_id"])) else (pred["transcript"], None, pred["seams"], pred.get("provenance"), [], {"inserted": 0, "adopted": 0})
        out = dict(pred)
        out["transcript"] = merged
        out["seams"] = seams
        out["provenance"] = prov
        out["retry"] = {"policy": args.policy, "conflict_tiebreak": not args.no_conflict_tiebreak, "outcomes": report, "counts": counts, "chain": None if chain is None else [{"label": c["label"], "bounds": list(c["bounds"]), "retry": c["retry"], "excluded": c["excluded"]} for c in chain]}
        if chain is not None:
            out["windows"] = [{"index": i, "start_sample": c["bounds"][0], "end_sample": c["bounds"][1], "boundary_reason": "retry_window" if c["retry"] else "overlap_window", "text": c["text"],
                               "label": c["label"], "retry": c["retry"], "excluded_by_merger": c["excluded"]} for i, c in enumerate(chain)]
        write_json(args.dest / path.name, out)
        totals["calls"] += 1
        totals["retries"] += len(retries)
        totals["inserted"] += counts["inserted"]
        totals["adopted"] += counts["adopted"]
        totals["failed"] += sum(1 for r in report for d in r["decisions"] if d["result"] == "retry_failed")
        for r in report:
            for d in r["decisions"]:
                if d["kind"] == "conflict":
                    totals["conflict_decisions"][d["result"]] = totals["conflict_decisions"].get(d["result"], 0) + 1
    print(json.dumps(totals), "| retries not decoded:", missing)
    write_json(args.dest / "_retry_totals.json", totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
