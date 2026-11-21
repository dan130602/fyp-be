"""Per-repetition result tables for the full development validation (verification only; never part of the image; reads the scorer's per-call error COUNTS, never reference text).

    python tools/full_report.py --jobs jobs-dev175.json --pilot pilot_calls_overlap_v1.json --languages languages.json --runs <evaluation-engine/runs> --out full-report.json \
        --baseline-raw RUN --baseline-final RUN \
        --repetition NAME RAW_RUN FINAL_RUN DOCS_FOLDER RUN_FOLDER DECISION_JSON COMPARE_RAW_JSON COMPARE_FINAL_JSON   (repeat once per repetition)
    python tools/full_report.py --write-languages languages.json --dataset-root <development dataset> --jobs jobs-dev175.json

For each repetition separately (nothing is averaged or chosen): integer errors over reference counts for WER and entity WER, raw and final, for all recordings, for the 22 original pilot recordings (which
influenced the selection of this configuration, so they are NOT independent validation) and for the remaining recordings, and per language; the paired per-call change against the baseline; the decision
tool's new large omissions, new loops, stop reasons and verdict; wall time, throughput, and GPU memory as sampled by nvidia-smi (a vLLM RESERVATION, not a need) next to the engine's own memory lines
from the worker log when present. Percentages are displayed only, never compared.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance                                                         # noqa: E402
import compare_backends                                                   # noqa: E402

KEYS = (("wer", "wer"), ("entity_wer", "entity_wer"))
MEMORY_LINES = re.compile(r"(Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|torch.compile takes|Memory profiling takes)")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(runs, run_id):
    path = Path(runs) / run_id / "scores" / "calls.jsonl"
    return {row["call_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def pct(errors, reference):
    return None if not reference else round(100.0 * errors / reference, 4)


def pooled(rows, calls, key):
    errors = sum((rows[c].get(key) or {}).get("errors", 0) for c in calls)
    reference = sum((rows[c].get(key) or {}).get("reference_words", 0) for c in calls)
    return {"errors": errors, "reference_tokens": reference, "percent": pct(errors, reference)}


def table(rows, groups):
    return {name: {metric: pooled(rows, calls, key) for metric, key in KEYS} for name, calls in groups.items()}


def paired(candidate_rows, baseline_rows, calls, key="wer"):
    changes = {c: (candidate_rows[c].get(key) or {}).get("errors", 0) - (baseline_rows[c].get(key) or {}).get("errors", 0) for c in calls}
    ordered = sorted(changes.items(), key=lambda item: (item[1], item[0]))
    return {"calls_better": sum(1 for v in changes.values() if v < 0), "calls_same": sum(1 for v in changes.values() if v == 0), "calls_worse": sum(1 for v in changes.values() if v > 0),
            "error_change": sum(changes.values()), "largest_improvements": ordered[:5], "largest_regressions": [item for item in ordered[::-1][:5] if item[1] > 0]}


def engine_memory_lines(run_folder):
    log = Path(run_folder) / "worker.log"
    if not log.is_file():
        return []
    return [line.strip()[:240] for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if MEMORY_LINES.search(line)][:12]


def run_facts(docs, run_folder):
    folder = Path(run_folder)
    wall = load(folder / "wall.json") if (folder / "wall.json").is_file() else {}
    rows = compare_backends.all_decodes(docs)
    audio_seconds = round(sum(d["meralion"]["audio"]["seconds"] for d in docs.values()), 2)
    seconds = wall.get("wall_seconds")
    return {"wall_seconds": seconds, "exit_code": wall.get("exit_code"), "audio_seconds": audio_seconds, "windows": sum(1 for r in rows if r[0] == "window"), "retries": sum(1 for r in rows if r[0] == "retry"),
            "decodes": len(rows), "audio_seconds_per_wall_second": round(audio_seconds / seconds, 2) if seconds else None, "decodes_per_wall_second": round(len(rows) / seconds, 2) if seconds else None,
            "model_load_s": sorted({d["meralion"]["timing"]["model_load_s"] for d in docs.values()}), "gpu_memory": compare_backends.gpu_scope.memory_report(folder / "gpu.csv"),
            "engine_memory_lines": engine_memory_lines(folder)}


def languages_from_dataset(dataset_root, call_ids):
    """{call: {"language": language_profile}} from each call's ground_truth.json METADATA (no reference text is read)."""
    result = {}
    for call in call_ids:
        metadata = load(Path(dataset_root) / "calls" / call / "ground_truth.json").get("metadata", {})
        result[call] = {"language": metadata["language_profile"]}
    return result


def build(args):
    calls = acceptance.read_jobs(args.jobs)
    pilot = set(load(args.pilot)["per_call"]) if args.pilot else set()
    languages = {c: v["language"] for c, v in load(args.languages).items()}
    groups = {"all": calls, "pilot_22_selection_influenced": [c for c in calls if c in pilot], "remaining_not_in_pilot": [c for c in calls if c not in pilot]}
    for language in sorted({languages[c] for c in calls}):
        groups[f"language:{language}"] = [c for c in calls if languages[c] == language]
    baseline = {kind: read_rows(args.runs, run) for kind, run in (("raw", args.baseline_raw), ("final", args.baseline_final))}
    report = {"calls": len(calls), "groups": {k: len(v) for k, v in groups.items()},
              "baseline": {"raw_run": args.baseline_raw, "final_run": args.baseline_final, **{kind: table(rows, groups) for kind, rows in baseline.items()}},
              "pilot_note": "the 22 pilot recordings influenced the selection of this configuration: they are not independent validation", "repetitions": {}}
    for name, raw_run, final_run, docs_folder, run_folder, decision_path, compare_raw, compare_final in args.repetition:
        docs = acceptance.scan_documents(docs_folder)[0]
        decision = load(decision_path)
        rows = {"raw": read_rows(args.runs, raw_run), "final": read_rows(args.runs, final_run)}
        repetition = {"scored_runs": {"raw": raw_run, "final": final_run}, "verdict": decision.get("verdict"), "failed_items": decision.get("failed_items"), "not_established": decision.get("not_established"),
                      "new_large_omissions": (decision.get("quality_details") or {}).get("new_large_omissions"), "new_loops": (decision.get("quality_details") or {}).get("new_loops"),
                      "candidate_stop_reasons": (decision.get("quality_details") or {}).get("candidate_stop_reasons"), "facts": run_facts(docs, run_folder)}
        for kind in ("raw", "final"):
            repetition[kind] = table(rows[kind], groups)
            repetition[f"{kind}_paired_wer_against_baseline"] = {group: paired(rows[kind], baseline[kind], members) for group, members in groups.items() if group in ("all", "pilot_22_selection_influenced", "remaining_not_in_pilot")}
        for kind, path in (("raw", compare_raw), ("final", compare_final)):
            data = load(path)
            candidate = next(iter(data["candidates"].values()))
            repetition[f"{kind}_config_compare"] = {"paired_wer": candidate.get("paired_wer"), "large_omission_events_new": len(candidate["large_omission_events"]["new"]), "critical_flags": len(candidate.get("critical_flags") or []),
                                                    "by_language": candidate.get("by_language")}
        report["repetitions"][name] = repetition
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--write-languages", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--languages", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline-raw")
    parser.add_argument("--baseline-final")
    parser.add_argument("--repetition", nargs=8, action="append", default=[], metavar=("NAME", "RAW_RUN", "FINAL_RUN", "DOCS", "RUN_FOLDER", "DECISION", "COMPARE_RAW", "COMPARE_FINAL"))
    args = parser.parse_args(argv)
    if args.write_languages:
        if args.write_languages.exists() or args.dataset_root is None:
            raise SystemExit("--write-languages needs --dataset-root and a file that does not exist yet")
        args.write_languages.write_text(json.dumps(languages_from_dataset(args.dataset_root, acceptance.read_jobs(args.jobs)), indent=1) + "\n", encoding="utf-8")
        return 0
    if None in (args.pilot, args.languages, args.runs, args.out, args.baseline_raw, args.baseline_final) or not args.repetition:
        parser.error("a report needs --pilot --languages --runs --out --baseline-raw --baseline-final and at least one --repetition")
    if args.out.exists():
        raise SystemExit(f"report already exists, not overwriting it: {args.out}")
    report = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"verdict": r["verdict"], "raw_all": r["raw"]["all"]["wer"], "final_all": r["final"]["all"]["wer"]} for name, r in report["repetitions"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
