"""Compare the Transformers worker and the vLLM worker on the same calls from their OWN output documents (Stage 3; never part of the image; reads no reference text, so no WER here:
the scores come from the unchanged scorer through to_predictions.py and import_predictions.py).

    python tools/compare_backends.py --tf <transcripts folder> --vllm <transcripts folder> --jobs <jobs.json> --state-tf <state folder> --state-vllm <state folder> --out report.json
                                     [--tf-run <run dir>] [--vllm-run <run dir>] [--tf-gpu gpu.csv] [--vllm-gpu gpu.csv]
    python tools/compare_backends.py ... --report-only        (without --jobs / --state-*: NOT an acceptance verdict)

Acceptance needs the expected recordings (--jobs) and EACH backend's persisted state: membership, complete first-pass windows, and retries equal to that backend's own persisted plan are checked
per side (the two backends may legitimately plan different retries, so the plans are never compared with each other). The command exits 1 when any gate fails.

Reported: window text differences by bounds; loops (the rule of decode_core.py: the most repeated 4-gram occurs 3 or more times) that vLLM has and Transformers does not, matched by bounds; stop-reason
counts per backend with the native values of every `other` or unavailable one; token-count distributions per backend (never compared across backends: the two definitions differ, plan R1); transcript
equality per call; speed and memory (summed decode seconds, model load, invocation wall time and peak GPU memory from the sampled nvidia-smi csv, with the vLLM reservation caveat).
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance                                                     # noqa: E402
import gpu_scope                                                      # noqa: E402

LOOP_REPEATS = 3


def repeated_4gram(words, n=4):
    seen = {}
    for i in range(len(words) - n + 1):
        key = tuple(words[i:i + n])
        seen[key] = seen.get(key, 0) + 1
    return max(seen.values()) if seen else 0


def percentile(values, share):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]


def all_decodes(docs):
    """Every decode in a set of documents as (kind, call, bounds, result dict), kind 'window' or 'retry'."""
    rows = []
    for call, document in sorted(docs.items()):
        block = document["meralion"]
        for w in block["trace"]["windows"]:
            rows.append(("window", call, (w["start_sample"], w["end_sample"]), {k: w.get(k) for k in ("text", "tokens", "stop_reason", "decode_seconds", "native")}))
        for d in block["retries"]["decodes"]:
            rows.append(("retry", call, tuple(d["window"]), d["result"]))
    return rows


def distribution(values):
    values = [v for v in values if v is not None]
    return {"count": len(values), "min": min(values) if values else None, "median": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values) if values else None}


def stop_summary(rows):
    counts = Counter(str(result["stop_reason"]) for _, _, _, result in rows)
    unusual = [{"kind": kind, "call": call, "window": list(bounds), "stop_reason": result["stop_reason"], "native": result.get("native")}
               for kind, call, bounds, result in rows if result["stop_reason"] not in ("end_token",)]
    return {"counts": dict(counts), "not_end_token": unusual[:50], "not_end_token_total": len(unusual)}


def run_facts(docs, run_dir, gpu_csv):
    wall = None
    if run_dir and (Path(run_dir) / "wall.json").is_file():
        wall = json.loads((Path(run_dir) / "wall.json").read_text(encoding="utf-8"))["wall_seconds"]
    rows = all_decodes(docs)
    return {"decode_seconds_sum": round(sum(r[3]["decode_seconds"] for r in rows), 3), "decodes": len(rows), "model_load_s": sorted({d["meralion"]["timing"]["model_load_s"] for d in docs.values()}),
            "model_check_s": sorted({d["meralion"]["timing"]["model_check_s"] for d in docs.values()}), "invocation_wall_s": wall, "gpu_memory": gpu_scope.memory_report(gpu_csv)}


def side_checks(label, docs, problems, expected, state_root):
    """The independent per-backend checks: -> {gate name: problems}."""
    found = {f"{label}_documents_are_unique": list(problems), f"{label}_first_pass_windows_are_complete": [p for d in docs.values() for p in acceptance.check_window_coverage(d)]}
    if expected is not None:
        found[f"{label}_membership_matches_the_job_manifest"] = acceptance.check_membership(expected, docs)
    if state_root is not None:
        ids = acceptance.invocation_ids(docs)
        if len(ids) != 1:
            found[f"{label}_persisted_plan_matches_the_documents"] = [f"the {label} documents come from {len(ids)} invocations"]
        else:
            try:
                state = acceptance.load_state(state_root, ids[0])
                found[f"{label}_persisted_plan_matches_the_documents"] = acceptance.check_persisted_plan(docs, state)
                found[f"{label}_plan_recomputes_from_the_persisted_first_pass"] = acceptance.check_recomputed_plan(docs, state)
            except (FileNotFoundError, KeyError) as error:
                found[f"{label}_persisted_plan_matches_the_documents"] = [f"the persisted state cannot be read: {error}"]
    return found


def compare(tf_docs, vllm_docs, *, expected_jobs=None, tf_state=None, vllm_state=None, tf_problems=(), vllm_problems=(), tf_run=None, vllm_run=None, tf_gpu=None, vllm_gpu=None):
    calls_tf, calls_vllm = set(tf_docs), set(vllm_docs)
    shared = sorted(calls_tf & calls_vllm)
    tf_rows, vllm_rows = all_decodes(tf_docs), all_decodes(vllm_docs)
    tf_by, vllm_by = {(k, c, b): r for k, c, b, r in tf_rows}, {(k, c, b): r for k, c, b, r in vllm_rows}
    common = sorted(set(tf_by) & set(vllm_by))
    differing = [{"kind": k, "call": c, "window": list(b), "tf": tf_by[(k, c, b)]["text"], "vllm": vllm_by[(k, c, b)]["text"]} for k, c, b in common if tf_by[(k, c, b)]["text"] != vllm_by[(k, c, b)]["text"]]
    tf_loops = {(k, c, b) for (k, c, b), r in tf_by.items() if repeated_4gram(r["text"].split()) >= LOOP_REPEATS}
    vllm_loops = {(k, c, b) for (k, c, b), r in vllm_by.items() if repeated_4gram(r["text"].split()) >= LOOP_REPEATS}
    new_loops = sorted(vllm_loops - tf_loops)
    transcripts = {"raw_equal": [c for c in shared if tf_docs[c]["meralion"]["transcript_raw"] == vllm_docs[c]["meralion"]["transcript_raw"]],
                   "final_equal": [c for c in shared if tf_docs[c]["results"]["transcripts"][0]["transcript"] == vllm_docs[c]["results"]["transcripts"][0]["transcript"]]}
    report = {
        "calls": {"tf": len(calls_tf), "vllm": len(calls_vllm), "shared": len(shared), "only_tf": sorted(calls_tf - calls_vllm), "only_vllm": sorted(calls_vllm - calls_tf)},
        "decodes": {"compared": len(common), "differing_text": len(differing), "examples": differing[:20], "only_tf": len(set(tf_by) - set(vllm_by)), "only_vllm": len(set(vllm_by) - set(tf_by))},
        "loops": {"tf": len(tf_loops), "vllm": len(vllm_loops), "new_in_vllm": [{"kind": k, "call": c, "window": list(b)} for k, c, b in new_loops], "fixed_in_vllm": len(tf_loops - vllm_loops)},
        "stop_reasons": {"tf": stop_summary(tf_rows), "vllm": stop_summary(vllm_rows)},
        "token_counts": {"tf": distribution([r[3]["tokens"] for r in tf_rows]), "vllm": distribution([r[3]["tokens"] for r in vllm_rows]), "note": "not comparable across backends (plan R1)"},
        "transcripts": {"shared_calls": len(shared), "raw_equal": len(transcripts["raw_equal"]), "final_equal": len(transcripts["final_equal"]),
                        "raw_differing": [c for c in shared if c not in transcripts["raw_equal"]], "final_differing": [c for c in shared if c not in transcripts["final_equal"]]},
        "speed_memory": {"tf": run_facts(tf_docs, tf_run, tf_gpu), "vllm": run_facts(vllm_docs, vllm_run, vllm_gpu),
                         "note": "vLLM pre-allocates gpu_memory_utilization (0.85) of the device, so its peak memory is a reservation, not a need; speed is reported, never gated"},
    }
    independent = {**side_checks("tf", tf_docs, tf_problems, expected_jobs, tf_state), **side_checks("vllm", vllm_docs, vllm_problems, expected_jobs, vllm_state)}
    counts = report["stop_reasons"]["vllm"]["counts"]
    report["gates"] = {"all_calls_in_both": not report["calls"]["only_tf"] and not report["calls"]["only_vllm"],
                       "no_new_loops": not new_loops,
                       "no_max_new_tokens_window_in_vllm": counts.get("max_new_tokens", 0) == 0,
                       "stop_reason_available_in_vllm": counts.get("None", 0) == 0,
                       "no_aborted_or_unexplained_stop": counts.get("other", 0) == 0}
    report["gates"].update({name: not problems for name, problems in independent.items()})
    report["independent_problems"] = {name: problems[:10] for name, problems in independent.items() if problems}
    report["evidence"] = {"tf": acceptance.evidence_of(tf_docs), "vllm": acceptance.evidence_of(vllm_docs)}
    has_evidence = expected_jobs is not None and tf_state is not None and vllm_state is not None
    report["acceptance_evidence"] = has_evidence
    report["passed"] = all(report["gates"].values())
    report["verdict"] = ("ACCEPTED" if report["passed"] else "REJECTED") if has_evidence else "NOT AN ACCEPTANCE VERDICT (--jobs, --state-tf and --state-vllm are required for one)"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tf", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--state-tf", type=Path)
    parser.add_argument("--state-vllm", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--tf-run", type=Path)
    parser.add_argument("--vllm-run", type=Path)
    parser.add_argument("--tf-gpu", type=Path)
    parser.add_argument("--vllm-gpu", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"report already exists, not overwriting it: {args.out}")
    if not args.report_only and (args.jobs is None or args.state_tf is None or args.state_vllm is None):
        print("compare_backends: an acceptance verdict needs --jobs, --state-tf and --state-vllm; use --report-only for a report that is not one", file=sys.stderr)
        return 2
    tf_docs, tf_problems = acceptance.scan_documents(args.tf)
    vllm_docs, vllm_problems = acceptance.scan_documents(args.vllm)
    if not tf_docs or not vllm_docs:
        print("compare_backends: no output documents on one side", file=sys.stderr)
        return 2
    report = compare(tf_docs, vllm_docs, expected_jobs=acceptance.read_jobs(args.jobs) if args.jobs else None, tf_state=args.state_tf, vllm_state=args.state_vllm, tf_problems=tf_problems,
                     vllm_problems=vllm_problems, tf_run=args.tf_run, vllm_run=args.vllm_run, tf_gpu=args.tf_gpu, vllm_gpu=args.vllm_gpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "gates": report["gates"], "independent_problems": report["independent_problems"], "decodes": {k: report["decodes"][k] for k in ("compared", "differing_text")}, "loops": {k: report["loops"][k] for k in ("tf", "vllm")},
                      "stop_reasons": {k: report["stop_reasons"][k]["counts"] for k in ("tf", "vllm")}}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
