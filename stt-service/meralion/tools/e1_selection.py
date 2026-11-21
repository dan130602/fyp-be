"""E1 selection (Stage 3 follow-up E, section 8.4): which ONE vLLM configuration, if any, is promoted to the full pilot comparison (job E2). Verification only; never part of the image.

    python tools/e1_selection.py --manifest e1-manifest.json --runs-root <fetched run folders> --score-root <scorer runs folder> --compare-dir <config_compare outputs> \
        --reference-docs <TF reference documents of the three recordings> --out e1_report.json

The rule is fixed and applied mechanically (never chosen after looking at one good run):

  eligible run   exactly the expected recordings; every first-pass window present; the run's OWN persisted plan present and equal to the recomputed one; exit code 0 with every recording done; every
                 decode ends with the end token (a token-limit hit, `other` or an unavailable stop reason disqualifies); one clean worker commit and one allocation and physical GPU across ALL runs;
                 raw predictions bound to the documents; ZERO new large omissions against the exploratory Transformers reference (raw and formatted) and ZERO new detected loops.
  eligible arm   BOTH of its repetitions are eligible. A failed or missing run makes the arm ineligible; it never contributes a score.
  ranking        only among eligible arms, lower first: (1) the worst repetition's total raw errors on the recordings; (2) the sum of raw errors over both repetitions; (3) the number of recordings
                 whose raw transcript differs between the arm's two repetitions; (4) the fixed arm order C, A, D, B (fewest departures from the shipped package first). (4) always decides.

The report states whether the stock control (arm C) reproduced the historical runaway window in this context (a fresh engine with CALL_134 first, NOT the original pilot's engine history); a claim
that a candidate fixed it is allowed only if the control reproduced it here and no candidate run hit the token limit on that window.
Exit code: 0 when an arm is promoted, 1 when none is eligible.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance                                                         # noqa: E402
import compare_backends                                                   # noqa: E402
import decision                                                           # noqa: E402

ARM_ORDER = ("C", "A", "D", "B")
RUNAWAY = {"call": "CALL_134", "window": (24960000, 25440000)}
FIELDS = {"raw": "transcript_raw", "final": "transcript_final"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def loop_windows(docs):
    return {(kind, call, bounds) for kind, call, bounds, result in compare_backends.all_decodes(docs) if compare_backends.repeated_4gram(result["text"].split()) >= compare_backends.LOOP_REPEATS}


def run_folder(runs_root, run_id):
    return Path(runs_root) / f"worker-v0-{run_id}"


def read_run(args, manifest, arm, run_id, reference_loops, reference_docs):
    """Everything known about one run, and every reason it is not eligible."""
    reasons, info = [], {"run_id": run_id, "arm": arm}
    folder = run_folder(args.runs_root, run_id)
    expected = manifest["jobs"]
    if not folder.is_dir():
        return {**info, "present": False, "reasons": [f"the run folder {folder} is missing"], "eligible": False, "docs": {}}
    docs, doc_problems = acceptance.scan_documents(folder / "out" / "transcripts") if (folder / "out" / "transcripts").is_dir() else ({}, ["no documents folder"])
    reasons += doc_problems + acceptance.check_membership(expected, docs)
    reasons += [p for d in docs.values() for p in acceptance.check_window_coverage(d)]
    ids = acceptance.invocation_ids(docs) if docs else []
    if ids != [run_id.split("/")[-1]] and ids:
        reasons.append(f"the documents come from invocation(s) {ids}, not {run_id}")
    try:
        state = acceptance.load_state(folder / "state", ids[0]) if len(ids) == 1 else None
    except FileNotFoundError as error:
        state = None
        reasons.append(f"the persisted state cannot be read: {error}")
    if state is not None and docs:
        reasons += acceptance.check_persisted_plan(docs, state) + acceptance.check_recomputed_plan(docs, state)
    wall = load(folder / "wall.json") if (folder / "wall.json").is_file() else None
    summary = load(folder / "summary.json") if (folder / "summary.json").is_file() else None
    if wall is None or wall.get("exit_code") != 0:
        reasons.append(f"the worker did not exit 0 (wall.json: {wall})")
    if summary is None or summary.get("done") != len(expected) or summary.get("failed") != 0 or summary.get("exit_code") != 0:
        reasons.append(f"the summary does not show every recording done: {None if summary is None else {k: summary.get(k) for k in ('exit_code', 'done', 'failed', 'members')}}")
    meta = load(folder / "run-meta.json") if (folder / "run-meta.json").is_file() else {}
    gpu_memory = compare_backends.gpu_scope.memory_report(folder / "gpu.csv")                                     # an object: the number is None unless the scope is verified
    info.update(present=True, wall_seconds=None if wall is None else wall.get("wall_seconds"), gpu_memory=gpu_memory, gpu_peak_mib=gpu_memory["peak_mib"],
                commit=meta.get("worker_git_commit"), slurm_job_id=meta.get("slurm_job_id"), gpu_uuid=meta.get("gpu_uuid"), docs=docs)
    decodes = compare_backends.all_decodes(docs)
    stops = {}
    for _, _, _, result in decodes:
        stops[str(result["stop_reason"])] = stops.get(str(result["stop_reason"]), 0) + 1
    info["stop_reasons"] = stops
    if any(reason != "end_token" for reason in stops):
        reasons.append(f"a decode did not end with the end token: {stops}")
    info["token_limit_windows"] = [[kind, call, list(bounds)] for kind, call, bounds, result in decodes if result["stop_reason"] == "max_new_tokens"]
    own_loops = loop_windows(docs)
    info["new_loops"] = sorted([kind, call, list(bounds)] for kind, call, bounds in own_loops - reference_loops)
    if info["new_loops"]:
        reasons.append(f"{len(info['new_loops'])} new detected loop window(s) against the reference")
    scores = {}
    for kind, field in FIELDS.items():
        score_run = f"worker-v0-{run_id}-{'raw' if kind == 'raw' else 'amt1'}{getattr(args, 'score_suffix', '')}"
        evidence = decision.Evidence()
        rows = decision.read_scores(evidence, args.score_root, score_run, f"{run_id} {kind}", expected, docs, field) if docs else None
        if rows is None or evidence.problems:
            reasons.append(f"the {kind} scores are missing or not this run's own output: {evidence.problems[:2]}")
            continue
        scores[kind] = {"wer": decision.pooled(rows, "wer"), "entity_wer": decision.pooled(rows, "entity_wer"), "per_call_errors": {c: (r.get("wer") or {}).get("errors") for c, r in sorted(rows.items())}}
        compare_path = Path(args.compare_dir) / f"{run_id}-{kind}.json"
        if not compare_path.is_file():
            reasons.append(f"the config_compare output for the {kind} transcripts is missing ({compare_path.name})")
            continue
        candidates = load(compare_path).get("candidates", {})
        if score_run not in candidates:
            reasons.append(f"the config_compare output for {kind} does not contain {score_run}")
            continue
        new_events = candidates[score_run]["large_omission_events"]["new"]
        scores[kind]["new_large_omissions"] = len(new_events)
        if new_events:
            reasons.append(f"{len(new_events)} new large omission(s) in the {kind} transcripts against the reference")
    info["scores"] = scores
    info["per_call_changes_against_reference"] = sorted(c for c, d in docs.items() if c in reference_docs and d["meralion"]["transcript_raw"] != reference_docs[c]["meralion"]["transcript_raw"])
    info["reasons"], info["eligible"] = reasons, not reasons
    return info


def decide(args):
    manifest = load(args.manifest)
    reference_docs, reference_problems = acceptance.scan_documents(args.reference_docs)
    reference_loops = loop_windows(reference_docs)
    runs, arms = {}, {}
    for arm, entry in manifest["arms"].items():
        arms[arm] = {"settings_id": entry["settings_id"], "config": entry["config"], "runs": entry["run_ids"]}
        for run_id in entry["run_ids"]:
            runs[run_id] = read_run(args, manifest, arm, run_id, reference_loops, reference_docs)
    # one allocation, one clean commit, one physical GPU across every run that exists
    present = [r for r in runs.values() if r.get("present")]
    for key, label in (("commit", "worker commits"), ("gpu_uuid", "GPU UUIDs"), ("slurm_job_id", "allocations")):
        values = {r.get(key) for r in present}
        if len(values) > 1 or (values and (None in values)):
            for r in present:
                r["reasons"].append(f"the runs do not share one {label}: {sorted(map(str, values))}")
                r["eligible"] = False
    commits = {r.get("commit") for r in present}
    if any((c is None or str(c) in ("unknown", "None") or str(c).endswith("-dirty")) for c in commits):
        for r in present:
            r["reasons"].append(f"the worker commit is not one clean recorded commit: {sorted(map(str, commits))}")
            r["eligible"] = False
    ranking, report_arms = [], {}
    for arm, entry in arms.items():
        arm_runs = [runs[i] for i in entry["runs"]]
        eligible = len(arm_runs) == 2 and all(r["eligible"] for r in arm_runs)
        report = {**entry, "eligible": eligible, "reasons": {r["run_id"]: r["reasons"] for r in arm_runs if r["reasons"]}}
        if eligible:
            errors = [r["scores"]["raw"]["wer"][0] for r in arm_runs]
            texts = [{c: d["meralion"]["transcript_raw"] for c, d in r["docs"].items()} for r in arm_runs]
            differing = sum(1 for c in texts[0] if texts[0][c] != texts[1].get(c))
            report["rank_key"] = {"worst_raw_errors": max(errors), "sum_raw_errors": sum(errors), "recordings_differing_between_repetitions": differing, "arm_order": ARM_ORDER.index(arm)}
            ranking.append((max(errors), sum(errors), differing, ARM_ORDER.index(arm), arm))
        report_arms[arm] = report
    ranking.sort()
    promoted = ranking[0][-1] if ranking else None
    control = [r for r in runs.values() if r["arm"] == "C" and r.get("present")]
    control_reproduced = any(w[1] == RUNAWAY["call"] and tuple(w[2]) == RUNAWAY["window"] for r in control for w in r["token_limit_windows"])
    candidate_hit = any(w[1] == RUNAWAY["call"] and tuple(w[2]) == RUNAWAY["window"] for r in runs.values() if r["arm"] in ("A", "B") and r.get("present") for w in r["token_limit_windows"])
    run_reports = {run_id: {k: v for k, v in r.items() if k != "docs"} for run_id, r in runs.items()}
    return {"promoted": promoted, "ranking": [{"arm": r[-1], "key": list(r[:-1])} for r in ranking], "arms": report_arms, "runs": run_reports, "reference_problems": reference_problems,
            "stock_control_reproduces_runaway": control_reproduced, "historical_engine_history_reproduced": False,
            "claims_allowed": {"candidate_fixed_the_historical_runaway": bool(control_reproduced and not candidate_hit and control),
                               "note": "E1 starts CALL_134 in a fresh engine, which does not reproduce the original pilot's preceding engine history"},
            "rule": "eligibility first (completeness, termination, zero new large omissions and zero new loops in BOTH repetitions), then rank; ARM_ORDER=C,A,D,B decides any tie"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("manifest", "runs-root", "score-root", "compare-dir", "reference-docs", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--score-suffix", default="", help="appended to the scored run ids (the importer names a partial run <id>-prelim-1)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"report already exists, not overwriting it: {args.out}")
    result = decide(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"promoted": result["promoted"], "ranking": result["ranking"], "eligible": {a: r["eligible"] for a, r in result["arms"].items()}}))
    return 0 if result["promoted"] else 1


if __name__ == "__main__":
    sys.exit(main())
