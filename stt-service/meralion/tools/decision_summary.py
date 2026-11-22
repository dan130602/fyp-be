"""One verdict over the repeated candidate pilot runs of job E2 (Stage 3 follow-up E, section 8.5). Verification only; never part of the image.

    python tools/decision_summary.py --manifest e2-manifest.json --decisions decision-r1.json decision-r2.json --runs-root <fetched run folders> --out summary.json

`e2-manifest.json` is DECLARED by the job before any run (and a copy is kept at submission): the slurm job, the commit, the GPU, the promoted settings id and config file, the baseline and candidate
invocation ids. This tool requires EXACTLY the two declared candidate invocations, each with its own tools/decision.py verdict, and nothing else:

  NOT AN ACCEPTANCE VERDICT (exit 2)   a number of decisions other than two; a declared invocation missing, an unexpected one, or one given twice (two files, or one file twice); a decision for another
                                       settings id or commit; decisions that do not share the same baseline documents; a declared run folder without wall.json or gpu.csv; a decision that is itself
                                       NOT AN ACCEPTANCE VERDICT. Nothing is decided.
  otherwise                            ACCEPTED (exit 0) only if EVERY repetition is ACCEPTED (the worst repetition governs), else REJECTED (exit 1).

Two manifest schemas. `e2-manifest-v1` (matched allocation: the baseline was run in the same allocation, listed under baseline_invocation_ids). `full175-manifest-v1` (frozen-historical baseline): the
manifest also DECLARES the candidate's configuration fingerprint and settings digest and a `baseline` block (kind frozen-historical, the sha256 of the declared baseline source, the replay-generated baseline
documents' invocation id and sha256). Every decision must then be a frozen-historical decision bound to exactly those, to the declared clean commit, and to the declared allocation (slurm job and GPU); the
baseline has no run folder of its own (its documents are replay-generated). What a decision lists as NOT ESTABLISHED is carried into the summary unchanged.

The report lists, per run, the integer errors and reference counts (percentages are displayed only), new large omissions, new loops, stop-reason counts, wall time and GPU peak, and the minimum and
maximum over the repetitions. Nothing is selected as best.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_backends                                                   # noqa: E402

SCHEMA = "e2-manifest-v1"
FULL_SCHEMA = "full175-manifest-v1"
HISTORICAL = "frozen-historical"
NOT_A_VERDICT = "NOT AN ACCEPTANCE VERDICT"
METRICS = ("raw_wer", "raw_entity_wer", "final_wer", "final_entity_wer")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_folder(runs_root, invocation_id):
    return Path(runs_root) / f"worker-v0-{invocation_id}"


def percentage(errors, reference):
    return None if not reference else round(100.0 * errors / reference, 4)


def run_row(decision, runs_root, invocation_id):
    counts = (decision.get("quality_details") or {}).get("integer_counts") or {}
    folder = run_folder(runs_root, invocation_id)
    wall = load(folder / "wall.json").get("wall_seconds") if (folder / "wall.json").is_file() else None
    gpu_memory = compare_backends.gpu_scope.memory_report(folder / "gpu.csv")                                     # the number is None unless its scope is verified (REV 4.1 A3)
    row = {"invocation_id": invocation_id, "verdict": decision.get("verdict"), "failed_items": decision.get("failed_items"), "wall_seconds": wall, "gpu_memory": gpu_memory, "gpu_peak_mib": gpu_memory["peak_mib"],
           "new_large_omissions": (decision.get("quality_details") or {}).get("new_large_omissions"), "new_loops": (decision.get("quality_details") or {}).get("new_loops"),
           "candidate_stop_reasons": (decision.get("quality_details") or {}).get("candidate_stop_reasons")}
    for metric in METRICS:
        cell = counts.get(metric) or {}
        row[metric] = {"candidate_errors": cell.get("candidate_errors"), "baseline_errors": cell.get("baseline_errors"), "reference_tokens": cell.get("reference_tokens"),
                       "candidate_percent": percentage(cell.get("candidate_errors") or 0, cell.get("reference_tokens")), "baseline_percent": percentage(cell.get("baseline_errors") or 0, cell.get("reference_tokens"))}
    return row


def span(rows, getter):
    values = [getter(r) for r in rows if getter(r) is not None]
    return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}


def summarise(manifest, decisions, runs_root):
    problems = []
    historical = manifest.get("schema") == FULL_SCHEMA
    if manifest.get("schema") not in (SCHEMA, FULL_SCHEMA):
        problems.append(f"the manifest is not a {SCHEMA} document")
    baseline_declared = manifest.get("baseline") or {}
    baseline_ids = list(baseline_declared.get("invocation_ids") or []) if historical else [(manifest.get("baseline_invocation_ids") or {}).get("pilot")]
    if historical and (baseline_declared.get("kind") != HISTORICAL or len(baseline_ids) != 1 or not baseline_declared.get("source_sha256") or not baseline_declared.get("documents_sha256")
                       or not manifest.get("config_fingerprint") or not manifest.get("settings_digest")):
        problems.append("the manifest must declare a frozen-historical baseline (kind, one invocation id, source and documents sha256) and the candidate's config_fingerprint and settings_digest")
    declared = list(manifest.get("candidate_invocation_ids") or [])
    if len(declared) != 2 or len(set(declared)) != 2:
        problems.append(f"the manifest must declare exactly two distinct candidate invocations, found {declared}")
    if len(decisions) != 2:
        problems.append(f"exactly two decisions are required, {len(decisions)} given")
    seen = {}
    for label, decision in decisions:
        ids = (((decision.get("identity") or {}).get("candidate")) or {}).get("invocation_ids") or []
        if len(ids) != 1:
            problems.append(f"{label}: the decision does not name exactly one candidate invocation ({ids})")
            continue
        seen.setdefault(ids[0], []).append(label)
    for invocation_id, labels in sorted(seen.items()):
        if len(labels) > 1:
            problems.append(f"duplicate decision for the candidate invocation {invocation_id}: {labels}")
    missing, unexpected = sorted(set(declared) - set(seen)), sorted(set(seen) - set(declared))
    if missing:
        problems.append(f"missing decision for the declared candidate invocation(s) {missing}")
    if unexpected:
        problems.append(f"unexpected decision for undeclared candidate invocation(s) {unexpected}")
    baselines = set()
    for label, decision in decisions:
        identity = decision.get("identity") or {}
        candidate, baseline = identity.get("candidate") or {}, identity.get("baseline") or {}
        if decision.get("verdict") == NOT_A_VERDICT or decision.get("evidence_problems"):
            problems.append(f"{label}: the decision is itself not an acceptance verdict ({(decision.get('evidence_problems') or ['no verdict'])[:2]})")
        if decision.get("candidate_settings") != manifest.get("promoted_settings_id") or candidate.get("settings_ids") != [manifest.get("promoted_settings_id")]:
            problems.append(f"{label}: made for settings {decision.get('candidate_settings')!r} / {candidate.get('settings_ids')}, the manifest promotes {manifest.get('promoted_settings_id')!r}")
        if candidate.get("worker_commits") != [manifest.get("worker_commit")] or (not historical and baseline.get("worker_commits") != [manifest.get("worker_commit")]):
            problems.append(f"{label}: worker commits {candidate.get('worker_commits')} / {baseline.get('worker_commits')} differ from the declared {manifest.get('worker_commit')!r}")
        if (baseline.get("invocation_ids") or []) != baseline_ids:
            problems.append(f"{label}: its baseline is {baseline.get('invocation_ids')}, the manifest declares {baseline_ids[0] if len(baseline_ids) == 1 else baseline_ids!r}")
        baselines.add(baseline.get("documents_sha256"))
        if historical:
            full = decision.get("identity") or {}
            if full.get("baseline_kind") != HISTORICAL or full.get("baseline_source_sha256") != baseline_declared.get("source_sha256"):
                problems.append(f"{label}: not a decision against the declared frozen-historical baseline source (kind {full.get('baseline_kind')!r}, source {full.get('baseline_source_sha256')!r})")
            if baseline.get("documents_sha256") != baseline_declared.get("documents_sha256"):
                problems.append(f"{label}: its baseline documents are not the declared ones")
            if candidate.get("config_fingerprints") != [manifest.get("config_fingerprint")]:
                problems.append(f"{label}: configuration fingerprints {candidate.get('config_fingerprints')} differ from the declared {manifest.get('config_fingerprint')!r}")
            if full.get("candidate_frozen_digests") != [manifest.get("settings_digest")]:
                problems.append(f"{label}: settings digests {full.get('candidate_frozen_digests')} differ from the declared {manifest.get('settings_digest')!r}")
            allocation = full.get("candidate_allocation") or {}
            if allocation.get("slurm_job_id") != manifest.get("slurm_job_id") or allocation.get("gpu_uuid") != manifest.get("gpu_uuid") or not manifest.get("gpu_uuid"):
                problems.append(f"{label}: allocation {allocation.get('slurm_job_id')!r} / {allocation.get('gpu_uuid')!r} differs from the declared {manifest.get('slurm_job_id')!r} / {manifest.get('gpu_uuid')!r}")
    if len(baselines) > 1:
        problems.append("the decisions were not made against the same baseline documents")
    declared_runs = list(declared) if historical else [*declared, baseline_ids[0]]
    for invocation_id in declared_runs:
        folder = run_folder(runs_root, invocation_id)
        for name in ("wall.json", "gpu.csv"):
            if not (folder / name).is_file():
                problems.append(f"the declared run {invocation_id} has no {name} in {folder}")
    by_invocation = {((d.get("identity") or {}).get("candidate") or {}).get("invocation_ids", [None])[0]: d for _, d in decisions}
    rows = [run_row(by_invocation[i], runs_root, i) for i in declared if i in by_invocation]
    verdicts = [d.get("verdict") for _, d in decisions]
    if problems:
        verdict, code = NOT_A_VERDICT, 2
    elif all(v == "ACCEPTED" for v in verdicts):
        verdict, code = "ACCEPTED", 0
    else:
        verdict, code = "REJECTED", 1
    ranges = {metric: span(rows, lambda r, m=metric: r[m]["candidate_errors"]) for metric in METRICS}
    ranges.update(wall_seconds=span(rows, lambda r: r["wall_seconds"]), gpu_peak_mib=span(rows, lambda r: r["gpu_peak_mib"]),
                  gpu_scopes=[{"invocation_id": r["invocation_id"], "scope": r["gpu_memory"]["scope"], "reason": r["gpu_memory"]["reason"], "peak_mib": r["gpu_peak_mib"],
                               "device_reported_peak_mib": r["gpu_memory"]["device_reported_peak_mib"]} for r in rows])
    not_established = sorted({line for _, d in decisions for line in (d.get("not_established") or [])})
    return {"verdict": verdict, "exit_code": code, "problems": problems, "runs": rows, "ranges_over_repetitions": ranges, "baseline": baseline_ids[0] if len(baseline_ids) == 1 else baseline_ids,
            "baseline_kind": HISTORICAL if historical else "matched-allocation", "not_established": not_established,
            "settings_id": manifest.get("promoted_settings_id"), "note": "the worst repetition governs; nothing here is a best-of selection; percentages are displayed only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, nargs="+", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"summary already exists, not overwriting it: {args.out}")
    manifest = load(args.manifest)
    decisions = [(str(path), load(path)) for path in args.decisions]
    result = summarise(manifest, decisions, args.runs_root)
    result["inputs"] = [{"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in (args.manifest, *args.decisions)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "problems": result["problems"][:6]}, indent=1, ensure_ascii=False))
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
