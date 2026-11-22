"""Replay parity (Stage 2 acceptance): run the WORKER PIPELINE on the 175 saved development calls with the saved decodes, and compare through the written output documents.

    python tools/replay_parity.py [--runs <evaluation-engine/runs>] [--workdir <new empty folder>] [--skip-crash] [--report <file>] [--config <runtime config>]

Verification only (never part of the image): it reads the git-ignored saved artefacts below, writes only into its work directory and its report, and uses no GPU.
  inputs    fg175-all/overlap                      the raw FG windows (4,080) of the 175 calls     -> the first-pass decodes
            fg175-lm-retries-all + fg175-lm-plan-all.json   the 507 saved retry decodes, keyed by (call, window)
  frozen    fg175-all/final_m2ax_seq_ap_lmonly_m5          the frozen raw transcripts
            fg175-all/final_m2ax_seq_ap_lmonly_m5_fee2     the legacy formatted transcripts (formatter percentages-v2), and its fee_rewrites.json (13 calls)
            fg175-all/final_m2ax_seq_ap_lmonly_m5_fee2_amt1  the selected formatted transcripts (percentages-v2+amount-v1), and its fee_rewrites.json (29 calls)
            fg175-lm-plan-lmonly.json                      the frozen retry plan (502 windows)
--config picks the runtime config (default: the packaged default, amt1); its formatter picks the formatted target folder (FORMATTED_TARGETS). The raw transcripts, the plan and the retry
counts are the same for every config. One invocation of 175 members with the frozen config (invocation-wide run budget). Then the same replay with simulated crashes, each resumed until it finishes.
Exit code 0 only if every acceptance check holds.
"""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meralion_worker import __version__, config as cfg, model_check, pipeline, strategy_fg          # noqa: E402
from meralion_worker.audio import ReplayRecording                                       # noqa: E402
from meralion_worker.decoders.base import STOP_REASONS, DecodeResult, Decoder           # noqa: E402
from meralion_worker.decoders.replay_decoder import ReplayDecoder, build_table          # noqa: E402
from meralion_worker.state import stable_view                                           # noqa: E402
from meralion_worker.storage import LocalStorage                                        # noqa: E402

REPO = ROOT.parents[1]
EXPECTED = {"windows_total": 4080, "run_budget": 612, "selected": 502, "skipped": 5, "inserted": 8, "adopted": 353, "failed": 1, "calls": 175}
# formatter name -> (the saved formatted folder under fg175-all, how many calls it changes relative to the raw transcripts)
FORMATTED_TARGETS = {
    "percentages-v2": ("final_m2ax_seq_ap_lmonly_m5_fee2", 13),
    "percentages-v2+amount-v1": ("final_m2ax_seq_ap_lmonly_m5_fee2_amt1", 29),
}


class SimulatedCrash(Exception):
    pass


class CrashingDecoder(Decoder):
    def __init__(self, inner, after):
        self.inner, self.after, self.count = inner, after, 0

    def decode(self, piece, *, recording_id, start_sample, end_sample):
        self.count += 1
        if self.count > self.after:
            raise SimulatedCrash(f"decoder died on call {self.count}")
        return self.inner.decode(piece, recording_id=recording_id, start_sample=start_sample, end_sample=end_sample)

    def describe(self):
        return self.inner.describe()


class FlakyStorage(LocalStorage):
    def __init__(self, root, predicate, nth):
        super().__init__(root)
        self.predicate, self.nth, self.seen = predicate, nth, 0

    def write_exclusive(self, key, data):
        if self.predicate(key):
            self.seen += 1
            if self.seen == self.nth:
                raise SimulatedCrash(f"storage died writing {key}")
        return super().write_exclusive(key, data)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): stable_view(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(root.rglob("*.json"))}


def load_inputs(runs):
    raw_dir = runs / "fg175-all" / "overlap"
    calls, entries = {}, []
    for path in sorted(raw_dir.glob("CALL_*.json")):
        raw = load_json(path)
        call = raw["call_id"]
        calls[call] = raw["loaded_samples"]
        for w in raw["windows"]:
            stop = w["stop_reason"] if w["stop_reason"] in STOP_REASONS else None
            entries.append(((call, w["start_sample"], w["end_sample"]), DecodeResult(text=w["text"], tokens=w["tokens"], stop_reason=stop, decode_seconds=w["decode_seconds"])))
    plan_all = load_json(runs / "fg175-lm-plan-all.json")
    for group in plan_all["selected"]:
        decode = load_json(runs / "fg175-lm-retries-all" / f"{group['retry_id']}.json")
        entries.append(((group["call_id"], group["window"][0], group["window"][1]), DecodeResult(text=decode["text"], tokens=decode["tokens"], stop_reason=None, decode_seconds=decode["decode_seconds"])))
    return calls, build_table(entries)


def make_context(effective):
    manifest = model_check.load_model_manifest(effective.model_manifest_id, effective.config_root)
    digest = model_check.canonical_digest(manifest)
    vendored = model_check.verify_vendored()
    return pipeline.RunContext(model={"manifest": manifest["id"], "manifest_sha256": digest, "revision": manifest["revision"]},
                               worker={"version": __version__, "git_commit": "replay", "vendored": vendored}, vendored_sources_sha256=vendored["sources_sha256"], model_manifest_sha256=digest)


def run_once(calls, table, work, *, decoder=None, output=None, state=None, effective, context):
    jobs = [pipeline.Job(call, f"replay://{call}") for call in sorted(calls)]
    return pipeline.run_invocation(
        invocation_id="replay-175", effective=effective, jobs=jobs, decoder=decoder or ReplayDecoder(table), open_recording=lambda job: ReplayRecording(job.job_name, calls[job.job_name]),
        output_storage=output or LocalStorage(work / "out"), state_storage=state or LocalStorage(work / "state"), context=context)


def compare_to_frozen(runs, calls, work, report, formatter):
    folder, expected_changed = FORMATTED_TARGETS[formatter]
    raw_dir, fee_dir = runs / "fg175-all" / "final_m2ax_seq_ap_lmonly_m5", runs / "fg175-all" / folder
    listed = {r["call_id"] for r in load_json(fee_dir / "fee_rewrites.json")["rewrites"]}
    # the expected changed set comes from the saved transcripts themselves (formatted differs from raw) and must agree with the folder's own rewrite list
    changed_saved = {call for call in calls if load_json(fee_dir / f"{call}.json")["transcript"] != load_json(raw_dir / f"{call}.json")["transcript"]}
    report.update(formatted_target=folder, expected_formatter_changed=expected_changed, saved_changed=len(changed_saved), saved_changed_agrees_with_fee_rewrites=changed_saved == listed)
    raw_equal = final_equal = block_final_equal = 0
    changed, totals, mismatches, more_than_seams = set(), Counter(), [], []
    for call in sorted(calls):
        document = load_json(work / "out" / "transcripts" / f"{call}.json")
        block = document["meralion"]
        want_raw, want_final = load_json(raw_dir / f"{call}.json")["transcript"], load_json(fee_dir / f"{call}.json")["transcript"]
        ok_raw = block["transcript_raw"] == want_raw
        ok_final = document["results"]["transcripts"][0]["transcript"] == want_final                     # D4: the adapter's own field
        ok_block = block["transcript_final"] == want_final
        raw_equal += ok_raw
        final_equal += ok_final
        block_final_equal += ok_block
        if not (ok_raw and ok_final and ok_block):
            mismatches.append({"call_id": call, "raw": ok_raw, "results_transcripts": ok_final, "transcript_final": ok_block})
        if block["retries"]["selected"] > len(block["trace"]["seams"]):
            more_than_seams.append(call)
        if block["formatter"]["changed"]:
            changed.add(call)
        for key in ("inserted", "adopted", "failed"):
            totals[key] += block["retries"][key]
    report.update(calls_with_more_retries_than_seam_records=more_than_seams, raw_equal=raw_equal, results_transcripts_equal_target=final_equal, transcript_final_equal_target=block_final_equal, mismatches=mismatches[:20],
                  formatter_changed=len(changed), formatter_changed_equals_saved=changed == changed_saved, retry_totals=dict(totals))


def reference_run(runs, calls, table, work, effective, context):
    started = time.monotonic()
    summary = run_once(calls, table, work, effective=effective, context=context)
    report = {"seconds": round(time.monotonic() - started, 1), "summary": summary.as_dict()}
    plan = load_json(work / "state" / "replay-175" / "plan.json")
    lmonly = {(g["call_id"], tuple(g["window"])) for g in load_json(runs / "fg175-lm-plan-lmonly.json")["selected"]}
    mine = {(g["call_id"], tuple(g["window"])) for g in plan["selected"]}
    report.update(calls=len(calls), windows_total=plan["windows_total"], run_budget=plan["run_budget"], run_budget_bound=plan["run_budget_bound"], selected=len(plan["selected"]),
                  skipped=len(plan["skipped"]), skipped_reasons=dict(Counter(s["reason"] for s in plan["skipped"])), plan_equals_lmonly=mine == lmonly)
    compare_to_frozen(runs, calls, work, report, effective.formatter)
    return report


def standalone_check(runs, calls, table, base, effective, context, reference, call="CALL_064"):
    """One recording alone through the same pipeline: does it select the same retries and give the same transcripts as inside the 175-call invocation? Retry ids are numbered by position in the
    invocation, so they are not compared; bounds, trigger semantics, decoded texts and both transcripts are."""
    work = base / f"standalone-{call}"
    run_once({call: calls[call]}, table, work, effective=effective, context=context)
    alone = load_json(work / "out" / "transcripts" / f"{call}.json")["meralion"]
    batch = load_json(base / "reference" / "out" / "transcripts" / f"{call}.json")["meralion"]
    windows = lambda block: sorted(tuple(d["window"]) for d in block["retries"]["decodes"])
    plan = load_json(work / "state" / "replay-175" / "plan.json")
    result = {"call": call, "retries_alone": len(alone["retries"]["decodes"]), "retries_in_the_175_call_run": len(batch["retries"]["decodes"]), "run_budget_alone": plan["run_budget"],
              "run_budget_bound_alone": plan["run_budget_bound"], "same_retry_bounds": windows(alone) == windows(batch),
              "same_retry_triggers": sorted(json.dumps([o["window"], o["triggers"]], sort_keys=True) for o in alone["retries"]["outcomes"]) == sorted(json.dumps([o["window"], o["triggers"]], sort_keys=True) for o in batch["retries"]["outcomes"]),
              "same_retry_texts": sorted((tuple(d["window"]), d["result"]["text"]) for d in alone["retries"]["decodes"]) == sorted((tuple(d["window"]), d["result"]["text"]) for d in batch["retries"]["decodes"]),
              "same_transcripts": alone["transcript_raw"] == batch["transcript_raw"] and alone["transcript_final"] == batch["transcript_final"]}
    result["identical"] = all(result[k] for k in ("same_retry_bounds", "same_retry_triggers", "same_retry_texts", "same_transcripts"))
    return result


def expected_resume_calls(work, calls, reference_plan):
    """What a resume must decode, computed from the CRASHED state and the plan: the first-pass windows of recordings that have no first pass, plus the planned retries of recordings that are
    not finished and lack their retry file. A recording with a done marker contributes nothing. Returns (sorted expected calls, names of the finished recordings)."""
    state = Path(work) / "state" / "replay-175"
    by_call = {}
    for group in reference_plan["selected"]:
        by_call.setdefault(group["call_id"], []).append(group)
    expected, finished = [], set()
    for name in sorted(calls):
        if (state / "done" / f"{name}.json").exists():
            finished.add(name)
            continue
        if not (state / "first_pass" / f"{name}.json").exists():
            expected += [(name, a, b) for a, b in strategy_fg.windows_for(calls[name])]
        for group in by_call.get(name, []):
            if not (state / "retries" / f"{group['retry_id']}.json").exists():
                expected.append((name, group["window"][0], group["window"][1]))
    return sorted(expected), finished


def crash_scenarios(runs, calls, table, base, reference_snapshot, effective, context, log):
    windows = EXPECTED["windows_total"]
    reference_plan = load_json(base / "reference" / "state" / "replay-175" / "plan.json")
    scenarios = [
        ("first-pass-mid", dict(decoder_after=1500)),
        ("after-plan", dict(decoder_after=windows)),
        ("retries-mid", dict(decoder_after=windows + 200)),
        ("outputs-mid", dict(output_fail_nth=60)),
        ("output-before-marker", dict(state_fail_nth=60)),
    ]
    results = []
    for name, spec in scenarios:
        work = base / f"crash-{name}"
        out, state = LocalStorage(work / "out"), LocalStorage(work / "state")
        if "output_fail_nth" in spec:
            out = FlakyStorage(work / "out", lambda key: key.startswith("transcripts/"), spec["output_fail_nth"])
        if "state_fail_nth" in spec:
            state = FlakyStorage(work / "state", lambda key: "/done/" in key, spec["state_fail_nth"])
        decoder = CrashingDecoder(ReplayDecoder(table), spec["decoder_after"]) if "decoder_after" in spec else ReplayDecoder(table)
        crashed = False
        log(f"  crash scenario {name}: first run")
        try:
            run_once(calls, table, work, decoder=decoder, output=out, state=state, effective=effective, context=context)
        except SimulatedCrash:
            crashed = True
        expected_calls, finished_before = expected_resume_calls(work, calls, reference_plan)
        resumes, resumed_decodes, resume_calls, summary = 0, 0, [], None
        while summary is None and resumes < 4:
            resumes += 1
            resume_decoder = ReplayDecoder(table)
            summary = run_once(calls, table, work, decoder=resume_decoder, effective=effective, context=context)
            resumed_decodes += len(resume_decoder.calls)
            resume_calls += resume_decoder.calls
        redecoded = sorted({call[0] for call in resume_calls if call[0] in finished_before})
        same_out = snapshot(work / "out") == reference_snapshot["out"]
        same_state = snapshot(work / "state") == reference_snapshot["state"]
        differences = [] if (same_out and same_state) else [k for k in set(snapshot(work / "out")) | set(reference_snapshot["out"]) if snapshot(work / "out").get(k) != reference_snapshot["out"].get(k)][:10]
        results.append({"scenario": name, "crashed": crashed, "resume_runs": resumes, "resume_exit_code": summary.exit_code if summary else None, "resumed_decode_calls": resumed_decodes,
                        "finished_before_resume": len(finished_before), "finished_recordings_redecoded": redecoded,
                        "expected_resume_decode_calls": len(expected_calls), "resume_decodes_match_expected": sorted(resume_calls) == expected_calls,
                        "outputs_identical": same_out, "state_identical": same_state, "differences": differences})
        log(f"  crash scenario {name}: crashed={crashed} finished before resume={len(finished_before)} resumed decodes={resumed_decodes} (expected {len(expected_calls)}) identical={same_out and same_state}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO / "evaluation-engine" / "runs")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-crash", action="store_true")
    parser.add_argument("--config", type=Path, help="runtime config file (default: the packaged default)")
    args = parser.parse_args(argv)
    runs = args.runs
    base = args.workdir or runs / "worker-v0-replay" / f"work-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if base.exists() and any(base.iterdir()):
        raise SystemExit(f"work directory is not empty: {base}")
    base.mkdir(parents=True, exist_ok=True)
    effective = cfg.load_runtime_config(args.config, env={})
    if effective.formatter not in FORMATTED_TARGETS:
        raise SystemExit(f"no saved formatted target for formatter {effective.formatter!r}; known: {sorted(FORMATTED_TARGETS)}")
    report_path = args.report or runs / "worker-v0-replay" / f"report-{effective.data['config_id']}.json"
    if report_path.exists():
        raise SystemExit(f"report already exists, not overwriting it: {report_path}")
    log = lambda message: print(message, flush=True)
    log(f"loading the saved decodes from {runs}")
    calls, table = load_inputs(runs)
    context = make_context(effective)
    log(f"{len(calls)} calls, {len(table)} saved decodes; reference run")
    reference = reference_run(runs, calls, table, base / "reference", effective, context)
    log(json.dumps({k: v for k, v in reference.items() if k not in ("summary", "mismatches")}, sort_keys=True))
    report = {"tool": "replay_parity", "worker_version": __version__, "config_id": effective.data["config_id"], "formatter": effective.formatter, "config_fingerprint": effective.fingerprint, "runs_dir": str(runs), "workdir": str(base), "expected": EXPECTED, "reference": reference}
    report["standalone"] = standalone_check(runs, calls, table, base, effective, context, reference)
    if not args.skip_crash:
        snap = {"out": snapshot(base / "reference" / "out"), "state": snapshot(base / "reference" / "state")}
        report["crash_scenarios"] = crash_scenarios(runs, calls, table, base, snap, effective, context, log)
    expected_budget = EXPECTED["run_budget"] if effective.data["strategy"]["run_cap"] is not None else None
    skipped_ok = reference["skipped"] == EXPECTED["skipped"] and reference["skipped_reasons"] == {"unresolved per-call budget": EXPECTED["skipped"]}
    checks = {
        "calls_175": reference["calls"] == EXPECTED["calls"],
        "windows_4080": reference["windows_total"] == EXPECTED["windows_total"],
        "run_budget_as_expected_and_not_bound": reference["run_budget"] == expected_budget and reference["run_budget_bound"] is False,
        "retry_groups_never_exceed_seam_records": reference["calls_with_more_retries_than_seam_records"] == [],
        "selected_502": reference["selected"] == EXPECTED["selected"],
        "skipped_5_unresolved_per_call": skipped_ok,
        "plan_equals_lmonly": reference["plan_equals_lmonly"],
        "run_exit_0_no_missing_decodes": reference["summary"]["exit_code"] == 0 and reference["summary"]["done"] == EXPECTED["calls"],
        "raw_175_of_175": reference["raw_equal"] == EXPECTED["calls"],
        "results_transcripts_equal_target_175_of_175": reference["results_transcripts_equal_target"] == EXPECTED["calls"],
        "transcript_final_equal_target_175_of_175": reference["transcript_final_equal_target"] == EXPECTED["calls"],
        "retry_totals_8_353_1": reference["retry_totals"] == {"inserted": EXPECTED["inserted"], "adopted": EXPECTED["adopted"], "failed": EXPECTED["failed"]},
        "formatter_changed_equals_saved": reference["formatter_changed"] == reference["expected_formatter_changed"] == reference["saved_changed"] and reference["formatter_changed_equals_saved"] and reference["saved_changed_agrees_with_fee_rewrites"],
    }
    if not args.skip_crash:
        scenarios = {s["scenario"]: s for s in report["crash_scenarios"]}
        checks["every_crash_scenario_crashed_and_resumed_identically"] = all(s["crashed"] and s["resume_exit_code"] == 0 and s["outputs_identical"] and s["state_identical"] for s in scenarios.values())
        checks["after_plan_resume_decodes_only_the_502_retries"] = scenarios["after-plan"]["resumed_decode_calls"] == EXPECTED["selected"]
        checks["finished_recordings_never_redecoded"] = all(not s["finished_recordings_redecoded"] for s in scenarios.values()) and any(s["finished_before_resume"] > 0 for s in scenarios.values())
        checks["resume_decodes_exactly_the_unfinished_work"] = all(s["resume_decodes_match_expected"] for s in scenarios.values())
    if effective.data["strategy"]["run_cap"] is None:                                              # a strategy without a budget must give a recording the same result alone as in the batch
        checks["standalone_CALL_064_equals_the_same_recording_in_the_175_call_run"] = report["standalone"]["identical"] and report["standalone"]["retries_alone"] == report["standalone"]["retries_in_the_175_call_run"] == 3
    report["checks"], report["passed"] = checks, all(checks.values())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, ok in checks.items():
        log(f"  {'PASS' if ok else 'FAIL'}  {name}")
    log(f"report: {report_path}\nPASSED" if report["passed"] else f"report: {report_path}\nFAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
