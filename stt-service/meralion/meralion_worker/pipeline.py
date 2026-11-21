"""One invocation: audio -> windows -> decode -> first merge -> retry plan -> retry decodes -> reconcile -> formatter -> output document -> storage, with persisted progress.

Phases, in order (every start runs them again and only does what is missing):
  1  invocation identity       create invocation.json or require an identical one (incompatible: ResumeError)
  2  outputs of another run    an existing transcripts/<job>.json that this run cannot prove it wrote blocks the whole invocation before any work
  3  inputs of a resumed run   every unfinished recording that has a recordings file is reloaded and its sha256 must be unchanged
  4  first pass                per recording: load, decode every FG window, persist; a final input failure marks the recording failed (the others continue)
  5  plan                      the retry selection over the recordings of THIS invocation, written once and never recomputed
  6  retries and outputs       per recording, in job order: decode the planned retries it lacks and persist them, then reconcile, format, build the document, write it exclusively and
                               write the done marker, BEFORE the next recording is touched (a ready recording never waits for another recording's retries)

The retry budget is invocation-wide (D3): the recordings of one invocation are planned together, and the plan survives a restart unchanged.
Audio is reloaded per recording in each phase, so memory is bounded by one recording (R3); the input hash is checked on every load.
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from meralion_worker import __version__, output, strategy_fg
from meralion_worker.audio import InputError
from meralion_worker.decoders.base import DecodeRequest, DecodeResult
from meralion_worker.formatters import apply_formatter
from meralion_worker.retry_selection import select_retries
from meralion_worker.state import InvocationState, ResumeError, sha256_bytes
from meralion_worker.storage import AlreadyExists

RETRY_KINDS = ("unresolved", "low_margin_conflict")
UNRESOLVED_CALL_CAP = 0.10
RUN_CAP = 0.15


@dataclass(frozen=True)
class Job:
    job_name: str
    input: str


@dataclass
class RunContext:
    model: dict                       # {"manifest", "manifest_sha256", "revision"}
    worker: dict                      # {"version", "git_commit", "vendored": {...}}
    vendored_sources_sha256: str
    model_manifest_sha256: str
    timing: dict = field(default_factory=lambda: {"model_check_s": 0.0, "model_load_s": 0.0})


@dataclass
class Summary:
    exit_code: int
    invocation_id: str
    resumed: bool
    members: int
    done: int
    failed: int
    windows_total: int
    run_budget: object                                           # an int, or None for a strategy without an invocation-wide budget
    run_budget_bound: bool
    retries_selected: int
    jobs: dict

    def as_dict(self):
        return dict(self.__dict__)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _output_key(name):
    return f"transcripts/{name}.json"


def _same_run(document, invocation_id, fingerprint, audio_sha256):
    block = (document or {}).get("meralion") or {}
    return (block.get("invocation") or {}).get("id") == invocation_id and block.get("config_fingerprint") == fingerprint and (block.get("audio") or {}).get("sha256") == audio_sha256


def _blocked_by_foreign_outputs(state, output_storage, jobs, effective):
    """Phase 2. Repairs a crash between an output and its done marker; collects every output this run cannot prove it wrote."""
    blocked = []
    for job in jobs:
        name = job.job_name
        if state.exists("done", f"{name}.json") or not output_storage.exists(_output_key(name)):
            continue
        data = output_storage.read(_output_key(name))
        try:
            document = json.loads(data.decode("utf-8"))
        except ValueError:
            blocked.append(name)
            continue
        recorded = state.read("recordings", f"{name}.json")["audio_sha256"] if state.exists("recordings", f"{name}.json") else None
        if recorded is not None and _same_run(document, state.invocation_id, effective.fingerprint, recorded):
            state.put_once("done", f"{name}.json", document={"job_name": name, "output_key": _output_key(name), "output_sha256": sha256_bytes(data)})
        else:
            blocked.append(name)
    return blocked


def _load_checked(open_recording, state, job, *, what):
    """Reload a recording and require the stored hash (R3). An input error on a recording that loaded before means the input changed."""
    try:
        recording = open_recording(job)
    except InputError as error:
        raise ResumeError(f"{what}: input of {job.job_name!r} can no longer be used ({error}); it loaded before") from error
    stored = state.read("recordings", f"{job.job_name}.json")
    if recording.sha256 != stored["audio_sha256"] or recording.num_samples != stored["num_samples"]:
        raise ResumeError(f"{what}: input of {job.job_name!r} changed since it was first read (sha256 or length differs)")
    return recording


def _pred_from_first_pass(document):
    windows = [{"start_sample": w["start_sample"], "end_sample": w["end_sample"], "text": w["result"]["text"]} for w in document["windows"]]
    return strategy_fg.first_merge(document["job_name"], document["num_samples"], windows)


def run_invocation(*, invocation_id, effective, jobs, decoder, open_recording, output_storage, state_storage, context, log=None):
    started = time.monotonic()
    log = log or (lambda message: None)
    names = [job.job_name for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError("job names must be unique")
    by_name = {job.job_name: job for job in jobs}
    state = InvocationState(state_storage, invocation_id)

    # phase 2 for a NEW invocation comes first: nothing can be proved to be this run's output, so any existing output refuses the run before a single state file is written
    if not state.exists("invocation.json"):
        foreign = [n for n in names if output_storage.exists(_output_key(n))]
        if foreign:
            raise ResumeError(f"outputs already exist that this invocation did not write: {foreign}; nothing was changed")

    # phase 1
    resumed = state.begin({
        "invocation_id": invocation_id, "jobs": [{"job_name": j.job_name, "input": j.input} for j in jobs], "config": effective.data, "config_fingerprint": effective.fingerprint,
        "model_manifest_sha256": context.model_manifest_sha256, "vendored_sources_sha256": context.vendored_sources_sha256, "decoder": decoder.identity(),
        "worker_version": __version__, "created_at": _now()})

    # phase 2
    blocked = _blocked_by_foreign_outputs(state, output_storage, jobs, effective)
    if blocked:
        raise ResumeError(f"outputs already exist that this invocation did not write: {blocked}; nothing was changed")
    failed = {n for n in names if state.exists("failed", f"{n}.json")}
    done = {n for n in names if state.exists("done", f"{n}.json")}

    # phase 3
    for name in names:
        if name not in done and name not in failed and state.exists("recordings", f"{name}.json"):
            _load_checked(open_recording, state, by_name[name], what="resume check")

    # phase 4
    for name in names:
        if name in done or name in failed:
            continue
        if state.exists("first_pass", f"{name}.json"):
            stored = state.read("recordings", f"{name}.json")
            existing = state.read("first_pass", f"{name}.json")
            if [(w["start_sample"], w["end_sample"]) for w in existing["windows"]] != strategy_fg.windows_for(stored["num_samples"]):
                raise ResumeError(f"first-pass windows of {name!r} do not match the FG plan for its length")
            continue
        try:
            recording = open_recording(by_name[name])
        except InputError as error:
            state.put_once("failed", f"{name}.json", document={"job_name": name, "reason": str(error)})
            failed.add(name)
            log(f"{name}: final input failure: {error}")
            continue
        record = {"job_name": name, "input": by_name[name].input, "audio_sha256": recording.sha256, "num_samples": recording.num_samples, "source_sample_rate": recording.source_sample_rate, "seconds": recording.seconds}
        created, _ = state.put_once("recordings", f"{name}.json", document=record)
        if not created:
            stored = state.read("recordings", f"{name}.json")
            if stored["audio_sha256"] != recording.sha256 or stored["num_samples"] != recording.num_samples:
                raise ResumeError(f"input of {name!r} changed since it was first read")
        windows = strategy_fg.windows_for(recording.num_samples)
        results = decoder.decode_many([DecodeRequest(name, a, b, recording.get(a, b)) for a, b in windows])
        state.put_once("first_pass", f"{name}.json", document={"job_name": name, "num_samples": recording.num_samples,
                                                              "windows": [{"start_sample": a, "end_sample": b, "result": r.to_dict()} for (a, b), r in zip(windows, results)]})
        log(f"{name}: first pass, {len(windows)} windows")

    members = [n for n in names if n not in failed]
    summary_jobs = {n: ("failed" if n in failed else "pending") for n in names}
    if not members:
        return Summary(2 if failed else 0, invocation_id, resumed, 0, 0, len(failed), 0, 0, False, 0, summary_jobs)

    # phase 5
    for name in members:
        if not state.exists("first_pass", f"{name}.json"):
            raise ResumeError(f"first pass of {name!r} is missing")
    first_pass_sha = {n: sha256_bytes(state.read_bytes("first_pass", f"{n}.json")) for n in members}
    if not state.exists("plan.json"):
        preds = {n: _pred_from_first_pass(state.read("first_pass", f"{n}.json")) for n in members}
        strategy = effective.data["strategy"]
        selection = select_retries(preds, kinds=tuple(strategy["retry_kinds"]), unresolved_call_cap=strategy["unresolved_call_cap"], run_cap=strategy["run_cap"])
        selected = [{"retry_id": f"{g['call_id']}_r{n:03d}", "call_id": g["call_id"], "window": list(g["window"]), "priority": g["priority"], "triggers": g["triggers"]} for n, g in enumerate(selection.selected)]
        state.put_once("plan.json", document={"members": members, "windows_total": selection.windows_total, "run_budget": selection.run_budget, "run_budget_bound": selection.run_budget_bound,
                                              "groups_total": selection.groups_total, "selected": selected, "skipped": selection.skipped, "first_pass_sha256": first_pass_sha})
    plan_bytes = state.read_bytes("plan.json")
    plan, plan_sha256 = json.loads(plan_bytes.decode("utf-8")), sha256_bytes(plan_bytes)
    if plan["members"] != members or plan["first_pass_sha256"] != first_pass_sha:
        raise ResumeError("plan.json was made from different recordings or first-pass results than the ones on disk")
    by_call = {}
    for group in plan["selected"]:
        by_call.setdefault(group["call_id"], []).append(group)

    def finish(name):
        """Reconcile, format, build the document, write it exclusively, then the done marker. Needs the recording's first pass and every one of its planned retries on disk."""
        first_pass = state.read("first_pass", f"{name}.json")
        stored = state.read("recordings", f"{name}.json")
        pred = _pred_from_first_pass(first_pass)
        retries, retry_decodes = [], []
        for group in by_call.get(name, []):
            document = state.read("retries", f"{group['retry_id']}.json")
            result = DecodeResult.from_dict(document["result"])
            retries.append({"retry_id": group["retry_id"], "window": group["window"], "triggers": group["triggers"], "text": result.text, "tokens": result.tokens, "route": result.stop_reason})
            retry_decodes.append({"retry_id": group["retry_id"], "window": group["window"], "result": document["result"]})
        retries.sort(key=lambda r: r["window"][0])
        reconciled = strategy_fg.reconcile(pred, retries)
        transcript_final = apply_formatter(effective.formatter, reconciled["transcript_raw"])
        decode_seconds = sum(w["result"]["decode_seconds"] for w in first_pass["windows"]) + sum(d["result"]["decode_seconds"] for d in retry_decodes)
        document = output.build_document(
            job_name=name, transcript_raw=reconciled["transcript_raw"], transcript_final=transcript_final, effective=effective.as_record(),
            invocation={"id": invocation_id, "members": len(members), "windows_total": plan["windows_total"], "run_budget": plan["run_budget"], "run_budget_bound": plan["run_budget_bound"], "plan_sha256": plan_sha256},
            decoder=decoder.describe(), model=context.model, worker=context.worker,
            audio={"input": by_name[name].input, "sha256": stored["audio_sha256"], "source_sample_rate": stored["source_sample_rate"], "seconds": stored["seconds"], "num_samples": stored["num_samples"]},
            retries={"selected": len(retry_decodes), "skipped": [s for s in plan["skipped"] if s["call_id"] == name], **reconciled["counts"], "outcomes": reconciled["outcomes"], "decodes": retry_decodes},
            trace={"windows": [{"start_sample": w["start_sample"], "end_sample": w["end_sample"], **w["result"]} for w in first_pass["windows"]], "seams": reconciled["seams"],
                   "provenance": reconciled["provenance"], "chain": reconciled["chain"]},
            timing={**context.timing, "decode_s": round(decode_seconds, 3), "total_s": round(time.monotonic() - started, 3)})
        data = output.serialize(document)
        try:
            output_storage.write_exclusive(_output_key(name), data)
        except AlreadyExists:                                         # a concurrent writer: acceptable only if it is this very run's output
            data = output_storage.read(_output_key(name))
            if not _same_run(json.loads(data.decode("utf-8")), invocation_id, effective.fingerprint, stored["audio_sha256"]):
                raise ResumeError(f"output for {name!r} appeared from another run while this one was writing it")
        state.put_once("done", f"{name}.json", document={"job_name": name, "output_key": _output_key(name), "output_sha256": sha256_bytes(data)})
        done.add(name)
        summary_jobs[name] = "done"
        log(f"{name}: written")

    # phase 6: per recording, in job order. Decode the retries it lacks, persist them, then finish it (output and done marker) BEFORE the next recording is touched, so a recording that is
    # ready is never held back by another recording's retries. The plan is invocation-wide and already on disk, so every recording's result is the same whatever the order or a restart.
    for name in members:
        if name in done:
            summary_jobs[name] = "done"
            continue
        missing = [g for g in by_call.get(name, []) if not state.exists("retries", f"{g['retry_id']}.json")]
        if missing:
            recording = _load_checked(open_recording, state, by_name[name], what="retry decoding")
            results = decoder.decode_many([DecodeRequest(name, g["window"][0], g["window"][1], recording.get(g["window"][0], g["window"][1])) for g in missing])
            for group, result in zip(missing, results):
                state.put_once("retries", f"{group['retry_id']}.json", document={"retry_id": group["retry_id"], "call_id": name, "window": group["window"], "result": result.to_dict()})
            log(f"{name}: {len(missing)} retry window(s)")
        finish(name)

    return Summary(2 if failed else 0, invocation_id, resumed, len(members), len(done & set(members)), len(failed), plan["windows_total"], plan["run_budget"], plan["run_budget_bound"],
                   len(plan["selected"]), summary_jobs)
