"""Score worker outputs for a checkpoint with the UNCHANGED engine scorer, and compare against the E00 baseline (CPU only, no inference).

    python -X utf8 sft_eval.py score   --docs <worker out folder of transcripts> --exp E01 --ckpt epoch-1 --out <eval folder> [--baseline <E00 eval folder>]

The three reported stages come from ONE set of generated transcripts (no extra ASR): raw (before formatting), percentage (formatter v2 applied to raw) and final (percentages-v2 + amount-v1 as written
by the worker, cross-checked against the vendored formatters). Each stage is imported as an engine run (provider meralion, allow-partial: 35 validation calls) and scored by scripts/score.py.
Outputs in --out: metrics.json (overall, by language, S/D/I, entity groups, per stage), spans_<stage>.csv, errors_vs_baseline_<stage>.csv (fixed / unresolved / regression / changed / maintained),
and the engine run ids. Inserted/hallucinated entities with no reference span are NOT in the span-level CSV in this version (the scorer attaches insertions only to selected runs).
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FYP_BE = HERE.parents[1]
WORKER = FYP_BE / "stt-service" / "meralion"
ENGINE = FYP_BE / "evaluation-engine"
STAGES = ("raw", "percentage", "final")


def _paths(worker_root, engine_root):
    for p in (str(Path(worker_root)), str(Path(worker_root) / "tools"), str(Path(engine_root)), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def make_predictions(docs, out, worker_root):
    """{stage: folder of CALL_x.json} from the worker's output documents."""
    _paths(worker_root, ENGINE)
    import to_predictions as tp
    from meralion_worker import formatters
    out = Path(out)
    folders = {}
    for stage, variant in (("raw", "raw"), ("final", "final")):
        folders[stage] = out / f"predictions_{stage}"
        tp.convert(docs, variant, folders[stage])
    folders["percentage"] = out / "predictions_percentage"
    folders["percentage"].mkdir(parents=True, exist_ok=True)
    mismatches = []
    for path in sorted(folders["raw"].glob("CALL_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        final = json.loads((folders["final"] / path.name).read_text(encoding="utf-8"))
        if formatters.apply_formatter("percentages-v2+amount-v1", row["transcript"]) != final["transcript"]:
            mismatches.append(path.stem)
        row["transcript"] = formatters.apply_formatter("percentages-v2", row["transcript"])
        row["settings"] = {**row["settings"], "transcript_variant": "percentage"}
        (folders["percentage"] / path.name).write_text(json.dumps(row, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return folders, mismatches


class EvaluationIncomplete(SystemExit):
    pass


def check_docs(docs_dir, valid_ids, summary_path=None):
    """The evaluation must cover EXACTLY the frozen validation calls: no missing, duplicate, extra or failed call, and every document has a non-empty raw transcript."""
    names = sorted(p.stem for p in Path(docs_dir).glob("*.json"))
    expected = sorted(valid_ids)
    problems = []
    if len(set(expected)) != len(expected):
        problems.append("the frozen validation id list has duplicates")
    if len(names) != len(set(names)):
        problems.append("duplicate documents")
    missing, extra = sorted(set(expected) - set(names)), sorted(set(names) - set(expected))
    if missing:
        problems.append(f"missing calls: {missing}")
    if extra:
        problems.append(f"calls outside the frozen validation set: {extra}")
    for name in names:
        doc = json.loads((Path(docs_dir) / f"{name}.json").read_text(encoding="utf-8"))
        if doc.get("jobName") != name or not doc["meralion"]["transcript_raw"].strip():
            problems.append(f"{name}: document name mismatch or empty transcript")
    if summary_path is not None:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        if summary.get("failed") != 0 or summary.get("done") != len(expected) or not summary.get("ok"):
            problems.append(f"worker summary: done {summary.get('done')}, failed {summary.get('failed')}, ok {summary.get('ok')}")
    if problems:
        raise EvaluationIncomplete("evaluation refused: " + "; ".join(problems))
    return names


CRITICAL = re.compile(r"\d|[$€£¥%]|\b(?:no|not|never|cannot|can't|don't|didn't|won't|yes|yeah|agree|agreed|accept|accepted|consent|confirm|confirmed|cancel|cancelled|million|thousand|hundred|billion|buy|sell|limit|market)\b|n't|"
                      r"[不没否别同意确认接受取消买卖万千百亿]", re.IGNORECASE)


def decode_health(docs_dir):
    """Windows whose decoding did not end normally (token limit / other), summed over the documents."""
    bad, windows, tokens = [], 0, 0
    for p in sorted(Path(docs_dir).glob("*.json")):
        for w in json.loads(p.read_text(encoding="utf-8"))["meralion"]["trace"]["windows"]:
            windows += 1
            tokens += int(w.get("tokens") or 0)
            if w.get("stop_reason") not in ("end_token", None):
                bad.append({"call": p.stem, "start_sample": w["start_sample"], "stop_reason": w["stop_reason"], "tokens": w.get("tokens")})
    return {"windows": windows, "abnormal_stops": len(bad), "examples": bad[:10], "token_limit_stops": sum(1 for b in bad if b["stop_reason"] == "max_new_tokens")}


def critical_insertions(run_dir, baseline_run, languages, out_path):
    """Units with inserted words whose text looks critical (numbers, amounts, negations, consent, order words): listed for review, and which of them are NEW relative to the baseline."""
    def collect(run):
        found = {}
        for path in sorted((Path(run) / "scores" / "units").glob("CALL_*.json")):
            for u in json.loads(path.read_text(encoding="utf-8"))["units"]:
                if u["insertions"] and CRITICAL.search(u["hyp_text"]):
                    found[(path.stem, u["turn_id"], u["ref_text"], u["hyp_text"])] = u["insertions"]
        return found
    cand, base = collect(run_dir), (collect(baseline_run) if baseline_run else {})
    new = {k: v for k, v in cand.items() if k not in base}
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["call_id", "language", "turn_id", "reference_text", "hypothesis_text", "inserted_words", "new_vs_baseline"])
        for k, v in sorted(cand.items()):
            writer.writerow([k[0], languages[k[0]], k[1], k[2], k[3], v, k not in base])
    return {"candidate_units": len(cand), "baseline_units": len(base), "new_units": len(new), "new_inserted_words": sum(new.values()), "resolved_vs_baseline": sum(1 for k in base if k not in cand)}


def import_and_score(folder, run_id, note, engine_root):
    sys.path.insert(0, str(engine_root))
    from scripts import import_predictions
    code = import_predictions.main(["--provider", "meralion", "--predictions", str(folder), "--run-id", run_id, "--allow-partial", "--allow-adapter", "--note", note, "--score"])
    if code:
        raise SystemExit(f"import/score of {run_id} failed with {code}")


def find_run(engine_root, run_id):
    """The preliminary run folder created for a partial import (`<id>-prelim-N`): the highest N."""
    runs = sorted((Path(engine_root) / "runs").glob(f"{run_id}-prelim-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if not runs:
        raise SystemExit(f"no run folder for {run_id}")
    return runs[-1]


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def aggregate(run_dir, languages):
    """Overall, per-language and per-entity-group counts from the scorer's own calls.jsonl / summary.json."""
    calls = read_jsonl(Path(run_dir) / "scores" / "calls.jsonl")
    keys = ("reference_words", "substitutions", "deletions", "insertions", "hits", "errors")
    totals = defaultdict(lambda: {"wer": dict.fromkeys(keys, 0), "entity": dict.fromkeys(keys, 0), "calls": 0})
    for c in calls:
        for bucket in ("ALL", languages[c["call_id"]]):
            t = totals[bucket]
            t["calls"] += 1
            for k in keys:
                t["wer"][k] += c["wer"][k]
                if c["entity_wer"]:
                    t["entity"][k] += c["entity_wer"][k]
    for t in totals.values():
        for part in ("wer", "entity"):
            t[part]["rate"] = t[part]["errors"] / t[part]["reference_words"] if t[part]["reference_words"] else None
    summary = json.loads((Path(run_dir) / "scores" / "summary.json").read_text(encoding="utf-8"))
    return {"by_scope": dict(totals), "entity_groups": summary["entity_wer"].get("groups", {}), "calls_scored": summary["calls_scored"], "normalizer": summary["normalizer_version"]}


def span_key(row):
    return (row["call_id"], row["turn_id"], row["start"], row["end"], row["group"], row["label"])


def error_csv(candidate_run, baseline_run, languages, out_path, experiment, checkpoint):
    cand = {span_key(r): r for r in read_jsonl(Path(candidate_run) / "scores" / "spans.jsonl")}
    base = {span_key(r): r for r in read_jsonl(Path(baseline_run) / "scores" / "spans.jsonl")} if baseline_run else {}
    counts = defaultdict(int)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "checkpoint", "call_id", "language", "span_id", "entity_category", "reference_text", "baseline_hypothesis", "candidate_hypothesis",
                         "baseline_errors", "candidate_errors", "candidate_S", "candidate_D", "status"])
        for key, row in sorted(cand.items()):
            b = base.get(key)
            e1 = row["errors"]
            if b is None:
                status = "no_baseline"
            else:
                e0 = b["errors"]
                same = b["hypothesis_tokens"] == row["hypothesis_tokens"]
                status = ("fixed" if e0 and not e1 else "regression" if not e0 and e1 else "unresolved" if e0 and e1 and same else "changed_error" if e0 and e1 else "maintained")
            counts[status] += 1
            writer.writerow([experiment, checkpoint, row["call_id"], languages[row["call_id"]], f"{key[0]}:{key[1]}:{key[2]}-{key[3]}:{key[5]}", row["group"], row["text"],
                             " ".join(b["hypothesis_tokens"]) if b else "", " ".join(row["hypothesis_tokens"]), b["errors"] if b else "", e1, row["substitutions"], row["deletions"], status])
    return dict(counts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--docs", type=Path, required=True)
    s.add_argument("--exp", required=True, help="experiment id, e.g. E00 / E01")
    s.add_argument("--ckpt", required=True, help="checkpoint id, e.g. base / epoch-1")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--baseline", type=Path, help="the E00 eval folder (its engine_runs.json is read)")
    s.add_argument("--split", type=Path, required=True, help="split.json of the SFT data package (call -> language)")
    s.add_argument("--valid", type=Path, required=True, help="validation_calls.json: the exact frozen set of 35 validation calls")
    s.add_argument("--worker-summary", type=Path, help="the worker's summary.json (must report done == 35, failed == 0)")
    s.add_argument("--worker-root", type=Path, default=WORKER)
    s.add_argument("--engine-root", type=Path, default=ENGINE)
    s.add_argument("--tag", default="", help="suffix that makes the engine run ids unique")
    args = p.parse_args(argv)
    languages = {c: v["language"] for c, v in json.loads(args.split.read_text(encoding="utf-8"))["per_call"].items()}
    args.out.mkdir(parents=True, exist_ok=True)
    valid_ids = json.loads(args.valid.read_text(encoding="utf-8"))["call_ids"]
    check_docs(args.docs, valid_ids, args.worker_summary)
    folders, mismatches = make_predictions(args.docs, args.out, args.worker_root)
    if mismatches:
        raise EvaluationIncomplete(f"evaluation refused: the worker's final transcript differs from the vendored formatters for {mismatches}")
    runs, metrics = {}, {"experiment": args.exp, "checkpoint": args.ckpt, "formatter_crosscheck_mismatches": mismatches, "validation_calls": len(valid_ids),
                       "decode_health": decode_health(args.docs), "stages": {}}
    baseline_runs = json.loads((args.baseline / "engine_runs.json").read_text(encoding="utf-8")) if args.baseline else {}
    for stage in STAGES:
        run_id = f"sft-{args.exp}-{args.ckpt}-{stage}{args.tag}".replace("_", "-")
        import_and_score(folders[stage], run_id, f"SFT validation: {args.exp} {args.ckpt} {stage} (35 development validation calls)", args.engine_root)
        run_dir = find_run(args.engine_root, run_id)
        runs[stage] = str(run_dir)
        metrics["stages"][stage] = aggregate(run_dir, languages)
        if metrics["stages"][stage]["calls_scored"] != len(valid_ids):
            raise EvaluationIncomplete(f"evaluation refused: {stage} scored {metrics['stages'][stage]['calls_scored']} calls, expected {len(valid_ids)}")
        base_run = baseline_runs.get(stage)
        metrics["stages"][stage]["critical_insertions"] = critical_insertions(run_dir, base_run, languages, args.out / f"critical_insertions_{stage}.csv")
        if stage in baseline_runs or args.baseline is None:
            metrics["stages"][stage]["errors_vs_baseline"] = error_csv(run_dir, baseline_runs.get(stage), languages, args.out / f"errors_vs_baseline_{stage}.csv", args.exp, args.ckpt)
    (args.out / "engine_runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    for stage in STAGES:
        a = metrics["stages"][stage]["by_scope"]["ALL"]
        print(f"{args.exp} {args.ckpt} {stage:10s} WER {100 * a['wer']['rate']:.3f}% ({a['wer']['errors']}/{a['wer']['reference_words']})  Entity WER {100 * a['entity']['rate']:.3f}% ({a['entity']['errors']}/{a['entity']['reference_words']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
