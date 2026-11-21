"""Repeatability of two fresh-engine runs of the same configuration (verification only; never part of the image; reads the scorer's per-call error COUNTS, never reference text).

    python tools/repeat_compare.py --docs-a A --docs-b B --label-a r1 --label-b r2 --runs <evaluation-engine/runs> --score raw=RUN_A,RUN_B --score final=RUN_A,RUN_B \
        [--config-compare raw=compare.json --config-compare final=compare.json] --out repeat.json

Reports, per recording and in total: raw and final transcript equality; first-pass window text differences by bounds; retry-plan differences (windows planned by one run only) and retry-output
differences on the windows both planned; the integer WER and entity-WER error counts of both runs and their per-call differences; and, when the config_compare outputs of run B against run A are given,
the critical-content flags and new large omissions between them. Identical aggregate scores do not establish identical transcripts, and two matching runs do not establish determinism: the report
says so and lists every difference it found.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance                                                         # noqa: E402


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_of(runs, run_id):
    path = Path(runs) / run_id / "scores" / "calls.jsonl"
    return {row["call_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def compare_documents(a, b):
    """-> report over the recordings both runs hold."""
    shared = sorted(set(a) & set(b))
    raw_diff = [c for c in shared if a[c]["meralion"]["transcript_raw"] != b[c]["meralion"]["transcript_raw"]]
    final_diff = [c for c in shared if a[c]["results"]["transcripts"][0]["transcript"] != b[c]["results"]["transcripts"][0]["transcript"]]
    window_diffs, plan_only_a, plan_only_b, retry_text_diffs = [], [], [], []
    windows_total = retries_common = 0
    for call in shared:
        wa = {(w["start_sample"], w["end_sample"]): w["text"] for w in a[call]["meralion"]["trace"]["windows"]}
        wb = {(w["start_sample"], w["end_sample"]): w["text"] for w in b[call]["meralion"]["trace"]["windows"]}
        for bounds in sorted(set(wa) | set(wb)):
            windows_total += 1
            if wa.get(bounds) != wb.get(bounds):
                window_diffs.append({"call": call, "window": list(bounds), "a": wa.get(bounds), "b": wb.get(bounds)})
        ra = {tuple(d["window"]): d["result"]["text"] for d in a[call]["meralion"]["retries"]["decodes"]}
        rb = {tuple(d["window"]): d["result"]["text"] for d in b[call]["meralion"]["retries"]["decodes"]}
        plan_only_a += [{"call": call, "window": list(w)} for w in sorted(set(ra) - set(rb))]
        plan_only_b += [{"call": call, "window": list(w)} for w in sorted(set(rb) - set(ra))]
        for w in sorted(set(ra) & set(rb)):
            retries_common += 1
            if ra[w] != rb[w]:
                retry_text_diffs.append({"call": call, "window": list(w), "a": ra[w], "b": rb[w]})
    return {"recordings": {"a": len(a), "b": len(b), "shared": len(shared), "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a))},
            "transcripts": {"raw_differing": len(raw_diff), "final_differing": len(final_diff), "raw_differing_calls": raw_diff, "final_differing_calls": final_diff,
                            "identical_raw": len(shared) - len(raw_diff), "identical_final": len(shared) - len(final_diff)},
            "first_pass_windows": {"compared": windows_total, "differing_text": len(window_diffs), "differing_calls": sorted({d["call"] for d in window_diffs}), "examples": window_diffs[:10]},
            "retry_plan": {"planned_by_a_only": len(plan_only_a), "planned_by_b_only": len(plan_only_b), "a_only": plan_only_a[:20], "b_only": plan_only_b[:20], "planned_by_both": retries_common},
            "retry_outputs": {"differing_text_on_common_windows": len(retry_text_diffs), "examples": retry_text_diffs[:10]}}


def compare_scores(rows_a, rows_b):
    calls = sorted(set(rows_a) & set(rows_b))
    out = {}
    for key in ("wer", "entity_wer"):
        errors_a = sum((rows_a[c].get(key) or {}).get("errors", 0) for c in calls)
        errors_b = sum((rows_b[c].get(key) or {}).get("errors", 0) for c in calls)
        reference = sum((rows_a[c].get(key) or {}).get("reference_words", 0) for c in calls)
        changed = sorted(((rows_b[c].get(key) or {}).get("errors", 0) - (rows_a[c].get(key) or {}).get("errors", 0), c) for c in calls)
        out[key] = {"errors_a": errors_a, "errors_b": errors_b, "reference_tokens": reference, "difference_b_minus_a": errors_b - errors_a,
                    "calls_with_a_different_count": sum(1 for d, _ in changed if d), "largest_decreases": [c for d, c in changed[:3] if d < 0], "largest_increases": [c for d, c in changed[::-1][:3] if d > 0]}
    return out


def build(args):
    a, problems_a = acceptance.scan_documents(args.docs_a)
    b, problems_b = acceptance.scan_documents(args.docs_b)
    report = {"label_a": args.label_a, "label_b": args.label_b, "document_problems": {"a": problems_a, "b": problems_b}, **compare_documents(a, b), "scores": {}, "config_compare": {},
              "note": "identical aggregate scores do not establish identical transcripts, and two matching runs do not establish guaranteed determinism"}
    for item in args.score:
        kind, runs = item.split("=", 1)
        run_a, run_b = runs.split(",")
        report["scores"][kind] = {"run_a": run_a, "run_b": run_b, **compare_scores(rows_of(args.runs, run_a), rows_of(args.runs, run_b))}
    for item in args.config_compare:
        kind, path = item.split("=", 1)
        candidate = next(iter(load(path)["candidates"].values()))
        report["config_compare"][kind] = {"critical_flags": len(candidate.get("critical_flags") or []), "critical_flags_detail": (candidate.get("critical_flags") or [])[:20],
                                          "new_large_omissions": len(candidate["large_omission_events"]["new"]), "recovered_large_omissions": len(candidate["large_omission_events"]["recovered"]),
                                          "paired_wer": candidate.get("paired_wer")}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("docs-a", "docs-b", "runs", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--score", action="append", default=[], metavar="KIND=RUN_A,RUN_B")
    parser.add_argument("--config-compare", action="append", default=[], metavar="KIND=FILE")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"report already exists, not overwriting it: {args.out}")
    report = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"transcripts": {k: report["transcripts"][k] for k in ("identical_raw", "identical_final", "raw_differing", "final_differing")}, "first_pass_windows": {k: report["first_pass_windows"][k] for k in ("compared", "differing_text")},
                      "retry_plan": {k: report["retry_plan"][k] for k in ("planned_by_a_only", "planned_by_b_only")}, "retry_outputs": report["retry_outputs"]["differing_text_on_common_windows"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
