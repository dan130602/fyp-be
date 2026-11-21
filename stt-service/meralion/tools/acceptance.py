"""Independent acceptance checks shared by live_parity.py and compare_backends.py (Stage 3 follow-up; verification only, never part of the image; reads no reference text).

Every check returns a list of problems (empty means it holds). The checks never trust one source with itself: the expected recordings come from the JOB MANIFEST, the expected first-pass
bounds from the audio length, and the retries from each backend's OWN persisted state (plan.json and the first_pass / retries files), which is also re-derived from the persisted first pass.

  scan_documents      reads a folder of output documents and finds duplicates BEFORE any dictionary is built
  check_membership    missing, unexpected and duplicate outputs against the job manifest
  check_window_coverage   the first-pass windows of a document equal the windows of its audio length: complete, unique, in order
  load_state / check_persisted_plan   decoded retries equal the persisted plan by retry id AND bounds; first-pass and retry files agree with the document; plan_sha256 matches
  check_recomputed_plan   the plan is recomputed with the vendored selection from the persisted first pass and the config's caps and must equal the persisted plan
  compare_same_gpu    a standalone run against the same recording inside another invocation: bounds, triggers, texts and transcripts (never retry ids or timing)
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meralion_worker import retry_selection, strategy_fg          # noqa: E402

NO_BUDGET = 1e9                                                    # a run cap large enough that no group is ever skipped for "run budget" (policies without an invocation-wide budget)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---- evidence: bind a report to the exact run it describes ---------------------------------------------------------------------------------------------------------------------

def document_sha256(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def evidence_of(docs):
    """Identity of a set of output documents, read from INSIDE them. A report carries this block; decision.py recomputes it from the documents it is given and rejects a mismatch (a stale report, or a
    report of another run, configuration or baseline)."""
    blocks = {name: document["meralion"] for name, document in docs.items()}
    return {"jobs": sorted(docs), "documents_sha256": hashlib.sha256(json.dumps({name: document_sha256(document) for name, document in sorted(docs.items())}, sort_keys=True).encode("utf-8")).hexdigest(),
            "invocation_ids": sorted({b["invocation"]["id"] for b in blocks.values()}), "plan_sha256": sorted({b["invocation"]["plan_sha256"] for b in blocks.values()}),
            "config_ids": sorted({b["config"]["config_id"] for b in blocks.values()}), "config_fingerprints": sorted({b["config_fingerprint"] for b in blocks.values()}),
            "strategy_ids": sorted({b["config"]["strategy"]["id"] for b in blocks.values()}), "formatters": sorted({b["formatter"]["name"] for b in blocks.values()}),
            "settings_ids": sorted({b["decoder"]["settings_id"] for b in blocks.values()}), "backends": sorted({b["decoder"]["backend"] for b in blocks.values()}),
            "worker_commits": sorted({str(b["worker"]["git_commit"]) for b in blocks.values()}), "model_manifest_sha256": sorted({b["model"]["manifest_sha256"] for b in blocks.values()}),
            "engine_modes": sorted({(b["decoder"].get("execution") or {}).get("mode") or "none" for b in blocks.values()}),
            "engine_core_clients": sorted({(b["decoder"].get("execution") or {}).get("engine_core_client") or "none" for b in blocks.values()}),
            "audio_sha256": {name: b["audio"]["sha256"] for name, b in sorted(blocks.items())}}


# ---- documents and membership -----------------------------------------------------------------------------------------------------------------------------------------------

def read_jobs(path):
    """The ordered job names of a job manifest ({"jobs": [{"job_name": ...}]}); duplicates in the manifest are an error."""
    names = [job["job_name"] for job in load_json(path)["jobs"]]
    if len(names) != len(set(names)):
        raise ValueError("the job manifest lists a recording twice")
    return names


def scan_documents(folder):
    """-> (docs by jobName, problems). A second file for the same jobName, or a file whose name is not its jobName, is a problem; the first file wins so later checks still run."""
    docs, problems = {}, []
    for path in sorted(Path(folder).glob("*.json")):
        document = load_json(path)
        name = document.get("jobName")
        if path.stem != name:
            problems.append(f"{path.name}: the file name does not match jobName {name!r}")
        if name in docs:
            problems.append(f"duplicate output for {name!r}: {path.name}")
            continue
        docs[name] = document
    return docs, problems


def check_membership(expected, docs):
    problems = []
    missing = [name for name in expected if name not in docs]
    unexpected = sorted(name for name in docs if name not in set(expected))
    if missing:
        problems.append(f"missing outputs for {len(missing)} expected recordings: {missing[:5]}")
    if unexpected:
        problems.append(f"unexpected outputs for {len(unexpected)} recordings not in the job manifest: {unexpected[:5]}")
    return problems


def check_window_coverage(document):
    block, name = document["meralion"], document["jobName"]
    want = [tuple(bounds) for bounds in strategy_fg.windows_for(block["audio"]["num_samples"])]
    got = [(w["start_sample"], w["end_sample"]) for w in block["trace"]["windows"]]
    if got == want:
        return []
    duplicated = sorted({b for b in got if got.count(b) > 1})
    return [f"{name}: first-pass windows are not the {len(want)} windows of the audio: got {len(got)}, missing {len([b for b in want if b not in got])}, "
            f"unexpected {len([b for b in got if b not in want])}, duplicated {len(duplicated)}"]


# ---- the persisted state of one invocation ----------------------------------------------------------------------------------------------------------------------------------

def invocation_ids(docs):
    return sorted({document["meralion"]["invocation"]["id"] for document in docs.values()})


def load_state(state_root, invocation_id):
    root = Path(state_root) / invocation_id
    if not (root / "plan.json").is_file():
        raise FileNotFoundError(f"no persisted plan at {root / 'plan.json'}")
    plan_bytes = (root / "plan.json").read_bytes()
    first_pass = {}
    for path in sorted((root / "first_pass").glob("*.json")):
        data = path.read_bytes()
        first_pass[path.stem] = (data, json.loads(data.decode("utf-8")))
    retries = {path.stem: load_json(path) for path in sorted((root / "retries").glob("*.json"))} if (root / "retries").is_dir() else {}
    return {"root": root, "plan_bytes": plan_bytes, "plan": json.loads(plan_bytes.decode("utf-8")), "first_pass": first_pass, "retries": retries, "invocation": load_json(root / "invocation.json")}


def check_persisted_plan(docs, state):
    """Each document against the invocation's own persisted plan, first-pass files and retry files."""
    problems, plan = [], state["plan"]
    if sorted(plan["members"]) != sorted(docs):
        problems.append(f"the persisted plan's members differ from the documents: plan-only {sorted(set(plan['members']) - set(docs))[:5]}, documents-only {sorted(set(docs) - set(plan['members']))[:5]}")
    by_call = {}
    for group in plan["selected"]:
        by_call.setdefault(group["call_id"], []).append(group)
    plan_sha = sha256_bytes(state["plan_bytes"])
    for name, document in sorted(docs.items()):
        block = document["meralion"]
        if block["invocation"]["plan_sha256"] != plan_sha:
            problems.append(f"{name}: plan_sha256 in the document is not the sha256 of the persisted plan.json")
        if name not in state["first_pass"]:
            problems.append(f"{name}: no persisted first-pass file")
        else:
            data, first = state["first_pass"][name]
            if plan["first_pass_sha256"].get(name) != sha256_bytes(data):
                problems.append(f"{name}: the first-pass file is not the one the plan was made from")
            persisted = [(w["start_sample"], w["end_sample"], w["result"]["text"]) for w in first["windows"]]
            shown = [(w["start_sample"], w["end_sample"], w["text"]) for w in block["trace"]["windows"]]
            if persisted != shown:
                problems.append(f"{name}: the document's first-pass windows differ from the persisted first pass")
        planned = [(g["retry_id"], tuple(g["window"])) for g in by_call.get(name, [])]
        decoded = [(d["retry_id"], tuple(d["window"])) for d in block["retries"]["decodes"]]
        if sorted(planned) != sorted(decoded):
            problems.append(f"{name}: decoded retries {sorted(decoded)} differ from the persisted plan's {sorted(planned)} (by retry id and bounds)")
        if block["retries"]["selected"] != len(planned):
            problems.append(f"{name}: retries.selected is {block['retries']['selected']} but the persisted plan selects {len(planned)}")
        for decode in block["retries"]["decodes"]:
            saved = state["retries"].get(decode["retry_id"])
            if saved is None:
                problems.append(f"{name}: no persisted retry file for {decode['retry_id']}")
            elif saved["result"]["text"] != decode["result"]["text"] or saved["window"] != decode["window"]:
                problems.append(f"{name}: retry {decode['retry_id']} differs from its persisted file")
    return problems


def recompute_selection(state, strategy):
    """The vendored selection run again on the persisted first pass with the config's caps: -> (selected [(retry_id, call, window)], skipped [(call, window, reason)], bound, budget)."""
    preds = {}
    for name in state["plan"]["members"]:
        first = state["first_pass"][name][1]
        windows = [{"start_sample": w["start_sample"], "end_sample": w["end_sample"], "text": w["result"]["text"]} for w in first["windows"]]
        preds[name] = strategy_fg.first_merge(name, first["num_samples"], windows)
    has_budget = strategy.get("run_cap") is not None
    selection = retry_selection.select_retries(preds, kinds=set(strategy["retry_kinds"]), unresolved_call_cap=strategy["unresolved_call_cap"], run_cap=strategy["run_cap"] if has_budget else NO_BUDGET)
    selected = [(f"{g['call_id']}_r{n:03d}", g["call_id"], tuple(g["window"])) for n, g in enumerate(selection.selected)]
    skipped = [(s["call_id"], tuple(s["window"]), s["reason"]) for s in selection.skipped]
    return selected, skipped, selection.run_budget_bound, (selection.run_budget if has_budget else None)


def check_recomputed_plan(docs, state):
    document = next(iter(docs.values()))
    selected, skipped, bound, budget = recompute_selection(state, document["meralion"]["config"]["strategy"])
    persisted_selected = [(g["retry_id"], g["call_id"], tuple(g["window"])) for g in state["plan"]["selected"]]
    persisted_skipped = [(s["call_id"], tuple(s["window"]), s["reason"]) for s in state["plan"]["skipped"]]
    problems = []
    if selected != persisted_selected:
        problems.append(f"the persisted plan is not what the selection gives for the persisted first pass: recomputed {len(selected)} retries, persisted {len(persisted_selected)}; "
                        f"only recomputed {sorted(set(selected) - set(persisted_selected))[:3]}, only persisted {sorted(set(persisted_selected) - set(selected))[:3]}")
    if skipped != persisted_skipped:
        problems.append("the persisted plan's skipped groups differ from the recomputed ones")
    if bool(state["plan"]["run_budget_bound"]) != bool(bound) or state["plan"]["run_budget"] != budget:
        problems.append(f"the persisted run budget ({state['plan']['run_budget']}, bound {state['plan']['run_budget_bound']}) is not the recomputed one ({budget}, bound {bound})")
    return problems


# ---- the same recording alone and inside another invocation (same GPU) ----------------------------------------------------------------------------------------------------------

def _triggers_by_window(block):
    return {tuple(o["window"]): json.dumps(o["triggers"], sort_keys=True) for o in block["retries"]["outcomes"]}


def compare_same_gpu(alone, reference):
    """alone / reference: {jobName: document}. Compared per recording, excluding retry ids, plan ids and timing: first-pass bounds and texts, selected retry bounds, trigger semantics, decoded
    retry texts, and both transcripts. Any difference is a problem."""
    problems = []
    for name in sorted(alone):
        if name not in reference:
            problems.append(f"{name}: not in the reference run")
            continue
        a, b = alone[name]["meralion"], reference[name]["meralion"]
        if [(w["start_sample"], w["end_sample"], w["text"]) for w in a["trace"]["windows"]] != [(w["start_sample"], w["end_sample"], w["text"]) for w in b["trace"]["windows"]]:
            problems.append(f"{name}: first-pass windows differ")
        a_retries = {tuple(d["window"]): d["result"]["text"] for d in a["retries"]["decodes"]}
        b_retries = {tuple(d["window"]): d["result"]["text"] for d in b["retries"]["decodes"]}
        if sorted(a_retries) != sorted(b_retries):
            problems.append(f"{name}: selected retry bounds differ: alone {sorted(a_retries)}, reference {sorted(b_retries)}")
        elif a_retries != b_retries:
            problems.append(f"{name}: decoded retry texts differ")
        if _triggers_by_window(a) != _triggers_by_window(b):
            problems.append(f"{name}: trigger semantics of the retries differ")
        if a["transcript_raw"] != b["transcript_raw"] or alone[name]["results"]["transcripts"][0]["transcript"] != reference[name]["results"]["transcripts"][0]["transcript"]:
            problems.append(f"{name}: transcripts differ")
    return problems
