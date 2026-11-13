"""Paired clean-versus-environment_v2 comparison of two SCORED runs (offline: reads runs/<id>/, never AWS).

A condition's raw WER is not comparable with another condition's, because they contain different calls and call difficulty differs. The
valid comparison is PAIRED: the same call transcribed from its clean audio and from its environment_v2 audio, then the difference.

Clean results can stand in for the clean (C1) calls that a V2 run skipped (--skip-clean-controls) only when the two runs are equivalent
(same handler file, TRANSCRIBE_* settings, Transcribe job settings and normaliser). Otherwise `check_equivalent` says so and the clean
controls must be transcribed again.
"""

import json
import random
from datetime import datetime
from pathlib import Path

from .config import FYP_BE_ROOT

# Parts of a Transcribe job record that identify the job or report its outcome, not how it was requested; everything else is a request
# parameter (language setup, speaker settings, vocabularies, models ...). These are what decide whether two runs are comparable.
VOLATILE_JOB_KEYS = {"TranscriptionJobName", "TranscriptionJobStatus", "StartTime", "CreationTime", "CompletionTime", "Transcript", "Media",
                     "MediaFormat", "MediaSampleRateHertz", "LanguageCodes", "IdentifiedLanguageScore", "FailureReason"}
IGNORED_ENVIRONMENT = {"TRANSCRIBE_OUTPUT_BUCKET"}      # where the files land; it does not change a transcript
MAX_RUN_GAP_DAYS = 30
BOOTSTRAP = 2000
SEED = 20261112


class CompareError(RuntimeError):
    pass


def load_run(runs_root, run_id, scores="scores"):
    folder = Path(runs_root) / run_id
    if not (folder / "metadata.json").is_file() or not (folder / scores / "calls.jsonl").is_file():
        raise CompareError(f"{run_id}: needs metadata.json and a scored {scores}/ folder (run score.py first)")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / scores / "summary.json").read_text(encoding="utf-8"))
    rows = {}
    for line in (folder / scores / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["call_id"]] = row
    return {"id": run_id, "metadata": metadata, "summary": summary, "rows": rows}


def load_metadata_only(runs_root, run_id):
    """A run that is collected but not scored yet (the assembler needs no scores)."""
    path = Path(runs_root) / run_id / "metadata.json"
    if not path.is_file():
        raise CompareError(f"{run_id}: no metadata.json (collect it first)")
    return {"id": run_id, "metadata": json.loads(path.read_text(encoding="utf-8")), "summary": {}}


def _job_parameters(run):
    """The distinct sets of request parameters across every completed job of a run (normally exactly one)."""
    found = set()
    for entry in run["metadata"]["calls"]:
        if entry.get("status") == "COMPLETED" and entry.get("job"):
            found.add(json.dumps({k: v for k, v in entry["job"].items() if k not in VOLATILE_JOB_KEYS}, sort_keys=True, ensure_ascii=False))
    return found


def _environment(run):
    return {k: v for k, v in (run["metadata"].get("handler_environment") or {}).items() if k not in IGNORED_ENVIRONMENT}


def check_equivalent(clean, v2):
    """(equivalent?, reasons it is not, warnings). Equivalent means the clean results may stand in for clean controls of the V2 run.

    Decided by what was actually REQUESTED from Transcribe (the recorded job parameters of every job) and the TRANSCRIBE_* settings, not
    by the handler file hash: the handler can only influence a transcript through those parameters. A changed handler file alone is a warning.
    """
    reasons, warnings = [], []
    a, b = clean["metadata"], v2["metadata"]
    if a.get("dataset_id") != b.get("dataset_id"):
        reasons.append("different datasets")
    if _environment(clean) != _environment(v2):
        reasons.append("the TRANSCRIBE_* settings differ")
    params_a, params_b = _job_parameters(clean), _job_parameters(v2)
    if not params_a or not params_b:
        reasons.append("a run has no completed Transcribe job record to compare")
    elif params_a != params_b:
        reasons.append("the Transcribe job parameters differ")
    if a["handler"]["sha256"] != b["handler"]["sha256"]:
        warnings.append("the production handler file differs from the one the clean run recorded; reuse rests on identical Transcribe job parameters")
    if clean["summary"] and v2["summary"] and clean["summary"].get("normalizer_version") != v2["summary"].get("normalizer_version"):
        reasons.append("the scoring normaliser version differs")
    try:
        gap = abs((datetime.fromisoformat(b["started_at"]) - datetime.fromisoformat(a["started_at"])).days)
        if gap > MAX_RUN_GAP_DAYS:
            warnings.append(f"the runs are {gap} days apart; Amazon Transcribe models can change silently, so reuse is risky")
    except (KeyError, ValueError):
        warnings.append("run dates could not be compared")
    return not reasons, reasons, warnings


def language_of(dataset_root, call_id):
    truth = json.loads((Path(dataset_root) / "calls" / call_id / "ground_truth.json").read_text(encoding="utf-8"))
    return truth["metadata"]["language_profile"]


def _totals(rows, metric):
    if metric == "speaker":
        top, bottom = sum(r["speaker"]["correct_items"] for r in rows), sum(r["speaker"]["eligible_items"] for r in rows)
    else:
        block = "wer" if metric == "wer" else "entity_wer"
        top, bottom = sum(r[block]["errors"] for r in rows), sum(r[block]["reference_words"] for r in rows)
    return top, bottom


def rate(rows, metric):
    top, bottom = _totals(rows, metric)
    return None if not bottom else top / bottom


def _call_wer(row):
    block = row["wer"]
    return block["errors"] / block["reference_words"] if block["reference_words"] else 0.0


def bootstrap_ci(values):
    """95% interval of the mean per-call WER difference (seeded, so the report is reproducible); None for fewer than 2 calls."""
    if len(values) < 2:
        return None
    rng = random.Random(SEED)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(BOOTSTRAP))
    return [round(100 * means[int(0.025 * BOOTSTRAP)], 3), round(100 * means[int(0.975 * BOOTSTRAP) - 1], 3)]


def _pp(after, before):
    return None if after is None or before is None else round(100 * (after - before), 3)


def _cell(pairs):
    clean_rows, v2_rows = [p[0] for p in pairs], [p[1] for p in pairs]
    cell = {"calls": len(pairs)}
    for metric in ("wer", "entity", "speaker"):
        before, after = rate(clean_rows, metric), rate(v2_rows, metric)
        cell[metric] = {"clean": None if before is None else round(100 * before, 3), "v2": None if after is None else round(100 * after, 3),
                        "delta_pp": _pp(after, before)}
    deltas = [_call_wer(b) - _call_wer(a) for a, b in pairs]
    cell["wer"]["mean_call_delta_pp"] = round(100 * sum(deltas) / len(deltas), 3) if deltas else None
    cell["wer"]["ci95_pp"] = bootstrap_ci(deltas)
    return cell


def _plain(rows):
    return {"calls": len(rows), "wer": None if not rows else round(100 * rate(rows, "wer"), 3),
            "entity": None if not rows or rate(rows, "entity") is None else round(100 * rate(rows, "entity"), 3)}


def compare(clean, v2, dataset_root=None):
    """The paired comparison. Raises CompareError when skipped clean controls cannot be taken from the clean run."""
    if v2["metadata"].get("audio_condition") != "environment_v2":
        raise CompareError(f"{v2['id']} is not an environment_v2 run")
    if clean["metadata"].get("audio_condition", "clean") != "clean":
        raise CompareError(f"{clean['id']} is not a clean run")
    equivalent, reasons, warnings = check_equivalent(clean, v2)
    skipped = list(v2["metadata"].get("skipped_clean_controls") or [])
    copied = [e["call_id"] for e in v2["metadata"]["calls"] if e.get("reused_from_run")]     # an assembled run carries them as entries
    if (skipped or copied) and not equivalent:
        raise CompareError("clean controls must be re-transcribed: " + "; ".join(reasons))
    root = Path(dataset_root) if dataset_root else FYP_BE_ROOT / "evaluation-dataset" / "datasets" / v2["metadata"]["dataset_id"]
    conditions = {e["call_id"]: e.get("v2_condition") for e in v2["metadata"]["calls"]}
    pairs, missing = [], []
    for call_id, row in sorted(v2["rows"].items()):
        if call_id in copied:
            continue
        before = clean["rows"].get(call_id)
        if before is None:
            missing.append(call_id)
        else:
            pairs.append((before, row, conditions.get(call_id) or "?", language_of(root, call_id)))
    by_condition, by_language = {}, {}
    for key in sorted({p[2] for p in pairs}):
        by_condition[key] = _cell([(a, b) for a, b, c, _ in pairs if c == key])
        for lang in sorted({p[3] for p in pairs if p[2] == key}):
            by_language[f"{key}|{lang}"] = _cell([(a, b) for a, b, c, l in pairs if c == key and l == lang])
    reused = [clean["rows"][c] for c in skipped if c in clean["rows"]] + [v2["rows"][c] for c in copied if c in v2["rows"]]
    dataset = {}
    for key in sorted({p[2] for p in pairs}):
        dataset[key] = _plain([b for _, b, c, _ in pairs if c == key])
    if reused:
        dataset["C1"] = dict(_plain(reused), source="reused_from_clean_run")
    return {"clean_run": clean["id"], "v2_run": v2["id"], "equivalent": equivalent, "reasons": reasons, "warnings": warnings,
            "reused_clean_calls": len(reused), "skipped_clean_controls_missing_in_clean_run": [c for c in skipped if c not in clean["rows"]],
            "calls_missing_in_clean_run": missing, "paired_by_condition": by_condition, "paired_by_condition_language": by_language,
            "v2_dataset_by_condition": dataset,
            "note": "Compare conditions through the PAIRED delta only; raw WER differs across conditions because the calls differ."}
