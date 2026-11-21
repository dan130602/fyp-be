"""Live parity (Stage 3): compare the worker's OWN output documents from a live run with independent evidence. Verification only; never part of the image; reads no reference text.

    python tools/live_parity.py --docs <transcripts folder> --runs <evaluation-engine/runs> --jobs <jobs.json> --state <state folder> [--summary wall.json] [--report r.json]
                                [--cross-hardware] [--same-gpu-reference <transcripts folder of the pilot run on the same GPU>]
    python tools/live_parity.py ... --report-only          (no --jobs / --state: NOT an acceptance verdict)

Acceptance needs the expected recordings (--jobs) AND the run's persisted state (--state, the folder that holds <invocation id>/plan.json). Checked:
  independent of any saved run   unique documents; membership against the job manifest; complete first-pass windows against the audio length; decoded retries equal the persisted plan by
                                 retry id AND bounds; first-pass and retry files agree with the documents; the plan re-derived from the persisted first pass equals the persisted plan
  against the saved development run  every first-pass window text; every retry text; the planned set against the frozen plan; both transcripts against the frozen targets.
Comparability is separate from correctness: a transcript comparison is informational when the invocation-wide budget bound or the plan differs from the frozen one, but the independent checks
above are never waived. --cross-hardware (a different GPU than the saved evidence) reports first-pass text changes and retry text changes as two separate counts instead of failing on them.
--same-gpu-reference compares a standalone run with the same recordings inside another invocation on the same GPU: bounds, triggers, texts and transcripts, never retry ids.
Exit code 0 only for an acceptance verdict of ACCEPTED (or, with --report-only, when every check that could be made held).
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance                                                     # noqa: E402

FORMATTED_FOLDERS = {"percentages-v2": "final_m2ax_seq_ap_lmonly_m5_fee2", "percentages-v2+amount-v1": "final_m2ax_seq_ap_lmonly_m5_fee2_amt1"}
RAW_FOLDER = "final_m2ax_seq_ap_lmonly_m5"
load_json = acceptance.load_json


def percentile(values, share):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]


def saved_comparison(docs, runs):
    """First-pass and retry texts, the planned set and both transcripts against the saved development run."""
    runs = Path(runs)
    saved_retry = {}
    for group in load_json(runs / "fg175-lm-plan-all.json")["selected"]:
        saved_retry[(group["call_id"], tuple(group["window"]))] = runs / "fg175-lm-retries-all" / f"{group['retry_id']}.json"
    frozen_plan = {(g["call_id"], tuple(g["window"])) for g in load_json(runs / "fg175-lm-plan-lmonly.json")["selected"]}
    window_total = window_diff = retry_total = retry_diff = retry_not_saved = 0
    differing_windows, differing_retries, not_in_saved = [], [], []
    planned, decode_seconds, formatters, bound = set(), [], set(), set()
    for call in sorted(docs):
        block = docs[call]["meralion"]
        bound.add(bool(block["invocation"]["run_budget_bound"]))
        formatters.add(block["formatter"]["name"])
        saved_windows = {(w["start_sample"], w["end_sample"]): w["text"] for w in load_json(runs / "fg175-all" / "overlap" / f"{call}.json")["windows"]}
        for window in block["trace"]["windows"]:
            window_total += 1
            decode_seconds.append(window["decode_seconds"])
            key = (window["start_sample"], window["end_sample"])
            if key not in saved_windows:
                window_diff += 1
                not_in_saved.append({"call": call, "window": list(key), "kind": "first-pass window not in the saved run"})
            elif saved_windows[key] != window["text"]:
                window_diff += 1
                differing_windows.append({"call": call, "window": list(key), "worker": window["text"], "saved": saved_windows[key]})
        for decode in block["retries"]["decodes"]:
            retry_total += 1
            decode_seconds.append(decode["result"]["decode_seconds"])
            key = (call, tuple(decode["window"]))
            planned.add(key)
            if key not in saved_retry:
                retry_not_saved += 1                                   # a retry the saved run never decoded: it cannot be compared
                continue
            saved = load_json(saved_retry[key])["text"]
            if saved != decode["result"]["text"]:
                retry_diff += 1
                differing_retries.append({"call": call, "window": list(decode["window"]), "worker": decode["result"]["text"], "saved": saved})
    frozen_restricted = {key for key in frozen_plan if key[0] in docs}
    plan_equal = planned == frozen_restricted
    budget_bound = any(bound)
    comparable = plan_equal and not budget_bound
    rows = []
    for call in sorted(docs):
        document = docs[call]
        name = document["meralion"]["formatter"]["name"]
        raw_saved = load_json(runs / "fg175-all" / RAW_FOLDER / f"{call}.json")["transcript"]
        final_saved = load_json(runs / "fg175-all" / FORMATTED_FOLDERS[name] / f"{call}.json")["transcript"]
        rows.append({"call": call, "raw_equal": document["meralion"]["transcript_raw"] == raw_saved,
                     "final_equal": document["results"]["transcripts"][0]["transcript"] == final_saved == document["meralion"]["transcript_final"]})
    unequal = [row["call"] for row in rows if not (row["raw_equal"] and row["final_equal"])]
    return {"windows": {"total": window_total, "first_pass_text_changes": window_diff, "differing": window_diff, "not_in_saved_run": len(not_in_saved), "examples": (differing_windows + not_in_saved)[:10]},
            "retries": {"total": retry_total, "retry_text_changes": retry_diff, "differing": retry_diff, "planned_but_never_decoded_in_the_frozen_run": retry_not_saved, "examples": differing_retries[:10]},
            "plan": {"planned": len(planned), "frozen_restricted": len(frozen_restricted), "equal": plan_equal, "only_worker": sorted(map(list, planned - frozen_restricted))[:10],
                     "only_frozen": sorted(map(list, frozen_restricted - planned))[:10]},
            "transcripts": {"comparable": comparable, "equal": len(rows) - len(unequal), "differing": unequal, "rows": rows}, "formatters": sorted(formatters), "run_budget_bound": budget_bound,
            "decode_seconds": decode_seconds}


def compare(docs, runs, summary=None, *, expected_jobs=None, state_root=None, cross_hardware=False, document_problems=(), reference_docs=None):
    report = saved_comparison(docs, runs)
    independent = {}
    independent["documents_are_unique"] = list(document_problems)
    if expected_jobs is not None:
        independent["membership_matches_the_job_manifest"] = acceptance.check_membership(expected_jobs, docs)
    independent["first_pass_windows_are_complete"] = [problem for document in docs.values() for problem in acceptance.check_window_coverage(document)]
    state_problems = {"persisted_plan_matches_the_documents": [], "plan_recomputes_from_the_persisted_first_pass": []}
    if state_root is not None:
        ids = acceptance.invocation_ids(docs)
        if len(ids) != 1:
            state_problems["persisted_plan_matches_the_documents"].append(f"the documents come from {len(ids)} invocations: {ids[:3]}")
        else:
            try:
                state = acceptance.load_state(state_root, ids[0])
                state_problems["persisted_plan_matches_the_documents"] = acceptance.check_persisted_plan(docs, state)
                state_problems["plan_recomputes_from_the_persisted_first_pass"] = acceptance.check_recomputed_plan(docs, state)
            except (FileNotFoundError, KeyError) as error:
                state_problems["persisted_plan_matches_the_documents"].append(f"the persisted state cannot be read: {error}")
        independent.update(state_problems)
    saved = {"first_pass_texts_equal_the_saved_run": report["windows"]["differing"] == 0, "retry_texts_equal_the_saved_run": report["retries"]["differing"] == 0 and report["retries"]["planned_but_never_decoded_in_the_frozen_run"] == 0,
             "plan_equals_the_frozen_plan_or_the_budget_bound": report["plan"]["equal"] or report["run_budget_bound"],
             "transcripts_equal_the_saved_targets_where_comparable": (not report["transcripts"]["comparable"]) or not report["transcripts"]["differing"]}
    if cross_hardware:
        saved = {name: True for name in saved}                          # reported as separate counts above; they do not decide acceptance on different hardware
    checks = {name: not problems for name, problems in independent.items()}
    checks.update(saved)
    if reference_docs is not None:
        problems = acceptance.compare_same_gpu(docs, reference_docs)
        independent["standalone_equals_the_same_recordings_in_the_reference_run"] = problems
        checks["standalone_equals_the_same_recordings_in_the_reference_run"] = not problems
    timing = {"decode_seconds_per_window": {"count": len(report["decode_seconds"]), "median": percentile(report["decode_seconds"], 0.5), "p95": percentile(report["decode_seconds"], 0.95),
                                            "max": max(report["decode_seconds"]) if report["decode_seconds"] else None, "sum": round(sum(report["decode_seconds"]), 3)},
              "model_check_s": sorted({d["meralion"]["timing"]["model_check_s"] for d in docs.values()}), "model_load_s": sorted({d["meralion"]["timing"]["model_load_s"] for d in docs.values()}),
              "invocation_wall_s": (summary or {}).get("wall_seconds")}
    del report["decode_seconds"]
    has_evidence = expected_jobs is not None and state_root is not None
    passed = all(checks.values())
    report.update(evidence=acceptance.evidence_of(docs), calls=len(docs), cross_hardware=cross_hardware, independent_problems={k: v[:10] for k, v in independent.items() if v}, checks=checks, passed=passed, timing=timing,
                  acceptance_evidence=has_evidence, verdict=("ACCEPTED" if passed else "REJECTED") if has_evidence else "NOT AN ACCEPTANCE VERDICT (--jobs and --state are required for one)")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cross-hardware", action="store_true")
    parser.add_argument("--same-gpu-reference", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.report_only and (args.jobs is None or args.state is None):
        print("live_parity: an acceptance verdict needs --jobs and --state; use --report-only for a report that is not one", file=sys.stderr)
        return 2
    docs, document_problems = acceptance.scan_documents(args.docs)
    if not docs:
        print(f"live_parity: no output documents in {args.docs}", file=sys.stderr)
        return 2
    reference = acceptance.scan_documents(args.same_gpu_reference)[0] if args.same_gpu_reference else None
    report = compare(docs, args.runs, load_json(args.summary) if args.summary else None, expected_jobs=acceptance.read_jobs(args.jobs) if args.jobs else None,
                     state_root=args.state, cross_hardware=args.cross_hardware, document_problems=document_problems, reference_docs=reference)
    text = json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if args.report:
        if args.report.exists():
            raise SystemExit(f"report already exists, not overwriting it: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    brief = {k: report[k] for k in ("verdict", "calls", "formatters", "run_budget_bound", "cross_hardware", "checks", "independent_problems")}
    brief.update(windows={k: report["windows"][k] for k in ("total", "first_pass_text_changes")}, retries={k: report["retries"][k] for k in ("total", "retry_text_changes")},
                 plan_equals_frozen=report["plan"]["equal"], transcripts={k: report["transcripts"][k] for k in ("comparable", "equal")}, timing=report["timing"])
    print(json.dumps(brief, indent=1, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
