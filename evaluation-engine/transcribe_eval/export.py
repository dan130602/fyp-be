"""Export a scored run as a self-contained local bundle for the fyp-ai dashboard (view only).

A plain copy: it needs a scored run, never a commit or a clean working tree (provenance is recorded as found), writes no absolute
machine paths and no MP3s. Audio is referenced by dataset / call / render ids plus its sha256, so the dashboard can find and verify the
original file on the same machine when `STT_AUDIO_ROOT` is configured.
"""

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from .collect import handler_provenance
from .config import FYP_BE_ROOT
from .reference import load_turn_times, load_turns

SCHEMA_VERSION = "1.0"
DEFAULT_DESTINATION = FYP_BE_ROOT.parent / "fyp-ai" / "evaluation" / "stt_runs"
V2_AUDIO = "environment_v2"
# A noisy / channel-degraded run is published under its own dataset id, so the dashboard (and Supabase) show it as a separate card
# next to the clean baseline. The audio reference keeps the REAL dataset id: the recordings live in that dataset's folder.
V2_DATASET_SUFFIX = "__environment_v2"


class ExportError(RuntimeError):
    pass


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(FYP_BE_ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _context(units, position, width=5):
    """The reference words just before and after a unit, within the same turn (for reading an error in place)."""
    turn = units[position]["turn_id"]
    before = [u["ref_text"] for u in units[max(0, position - width * 2):position] if u["turn_id"] == turn and u["ref_text"]][-width:]
    after = [u["ref_text"] for u in units[position + 1:position + 1 + width * 2] if u["turn_id"] == turn and u["ref_text"]][:width]
    return {"before": " ".join(before), "after": " ".join(after)}


def frequent_errors(calls_units):
    """Error groups across a run: {op, ref, hyp, count, entity_count, occurrences[], examples[]}, entity-related first.
    Keyed by the normalised tokens the scoring compared, with the raw forms kept as examples."""
    groups = defaultdict(lambda: {"count": 0, "entity_count": 0, "occurrences": [], "examples": []})
    for call_id, units in calls_units.items():
        for position, unit in enumerate(units):
            for part in unit["parts"]:
                if part["op"] == "equal":
                    continue
                group = groups[(part["op"], part["ref"], part["hyp"])]
                group["count"] += 1
                group["entity_count"] += bool(part["entity"])
                group["occurrences"].append({"call_id": call_id, "unit": unit["id"], "turn_id": unit["turn_id"], "entity": bool(part["entity"]),
                                             "context": _context(units, position)})
                example = {"ref_text": unit["ref_text"], "hyp_text": unit["hyp_text"]}
                if example not in group["examples"] and len(group["examples"]) < 3:
                    group["examples"].append(example)
    rows = [{"op": op, "ref": ref, "hyp": hyp, **data} for (op, ref, hyp), data in groups.items()]
    return sorted(rows, key=lambda row: (-row["entity_count"], -row["count"], row["op"], row["ref"] or "", row["hyp"] or ""))


def _language(dataset_root, call_id):
    """The call's language profile (ENGLISH, MANDARIN, EN_ZH_MIXED, SINGLISH) from its ground truth, or None for older datasets."""
    try:
        return _read_json(Path(dataset_root) / "calls" / call_id / "ground_truth.json").get("metadata", {}).get("language_profile")
    except (OSError, ValueError):
        return None


def _audio_reference(dataset_id, call_id, entry, metadata):
    """Which recording was evaluated. A changed (non-C1) environment_v2 call points at its variant file; every other call, clean runs
    and the C1 calls of a V2 run included, points at the original render exactly as before."""
    reference = {"dataset_id": dataset_id, "call_id": call_id, "render_id": entry.get("render_id") or metadata.get("render_id"),
                 "audio_sha256": entry.get("audio_sha256")}
    if entry.get("audio_condition") == V2_AUDIO and entry.get("v2_condition") not in (None, "C1"):
        reference["variant"] = V2_AUDIO
    return reference


def _comparison(run_dir):
    """The paired clean-versus-changed result that scripts/compare_conditions.py saved next to the run, if there is one."""
    found = sorted(Path(run_dir).glob("comparison_vs_*.json"))
    return _read_json(found[-1]) if found else None


def export_run(run_dir, destination=DEFAULT_DESTINATION, *, overwrite=False, dataset_root=None):
    """Write `<destination>/<run id>/` from a scored run; returns the bundle folder."""
    run_dir = Path(run_dir)
    scores = run_dir / "scores"
    if not (scores / "summary.json").is_file() or not (scores / "units").is_dir():
        raise ExportError(f"{run_dir.name} is not scored yet (run scripts/score.py --run {run_dir.name} first)")
    metadata, summary = _read_json(run_dir / "metadata.json"), _read_json(scores / "summary.json")
    dataset_id = metadata["dataset_id"]
    v2 = metadata.get("audio_condition") == V2_AUDIO
    dataset_root = Path(dataset_root) if dataset_root else FYP_BE_ROOT / "evaluation-dataset" / "datasets" / dataset_id
    bundle = Path(destination) / run_dir.name
    if bundle.exists():
        if not overwrite:
            raise ExportError(f"{bundle} already exists; pass --overwrite to replace it")
        shutil.rmtree(bundle)

    scored = {json.loads(line)["call_id"]: json.loads(line) for line in (scores / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line}
    spans = defaultdict(list)
    for line in (scores / "spans.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            spans[row.pop("call_id")].append(row)
    calls_units = {path.stem: _read_json(path)["units"] for path in sorted((scores / "units").glob("CALL_*.json"))}
    entries = {e["call_id"]: e for e in metadata["calls"]}
    manifest = dataset_root / "dataset_manifest.json"
    job = next((e["job"] for e in metadata["calls"] if e.get("job")), None) or {}
    job_settings = {"language_code": job.get("LanguageCode"), "settings": job.get("Settings"), "media_format": job.get("MediaFormat"),
                    "media_sample_rate_hertz": job.get("MediaSampleRateHertz"), "vocabulary_name": (job.get("Settings") or {}).get("VocabularyName"),
                    "language_model_name": (job.get("ModelSettings") or {}).get("LanguageModelName"),
                    "language_options": job.get("LanguageOptions"), "identify_multiple_languages": job.get("IdentifyMultipleLanguages")}

    (bundle / "calls").mkdir(parents=True)
    call_rows = []
    for call_id, units in calls_units.items():
        row, entry = scored[call_id], entries[call_id]
        turns_ref = load_turns(dataset_root, call_id)
        times = load_turn_times(dataset_root, call_id)
        call = {
            "schema_version": SCHEMA_VERSION, "call_id": call_id,
            "audio": _audio_reference(dataset_id, call_id, entry, metadata),
            "reference_fingerprints": {"ground_truth_sha256": _sha256(dataset_root / "calls" / call_id / "ground_truth.json"),
                                       "transcript_sha256": _sha256(dataset_root / "calls" / call_id / "transcript.json")},
            "metrics": {"wer": row["wer"], "entity_wer": row["entity_wer"], "speaker": row["speaker"], "confidence": row["confidence"]},
            "turns": [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"],
                       "start": times.get(t["turn_id"], (None, None))[0], "end": times.get(t["turn_id"], (None, None))[1]} for t in turns_ref],
            "units": units, "entity_spans": spans.get(call_id, []),
            "raw_reference_text": row["raw_reference_text"], "raw_hypothesis_text": row["raw_hypothesis_text"],
        }
        call_row = {"call_id": call_id, "language": _language(dataset_root, call_id), "wer": row["wer"], "entity_wer": row["entity_wer"], "speaker": row["speaker"],
                    "mean_confidence": round(row["confidence"]["sum"] / row["confidence"]["items"], 4) if row["confidence"]["items"] else None,
                    "audio_seconds": (times[max(times)][1] if times else None)}
        if v2:
            call["condition"] = call_row["condition"] = entry.get("v2_condition")
        (bundle / "calls" / f"{call_id}.json").write_text(json.dumps(call, ensure_ascii=False) + "\n", encoding="utf-8")
        call_rows.append(call_row)

    run = {
        "schema_version": SCHEMA_VERSION, "run_id": run_dir.name, "dataset_id": dataset_id + V2_DATASET_SUFFIX if v2 else dataset_id,
        "render_id": metadata.get("render_id"),
        "dataset_call_ids": _read_json(manifest).get("call_ids", []) if manifest.is_file() else [],
        "collected_at": metadata.get("started_at"), "scored_at": summary.get("scored_at"),
        "versions": {"normalizer": summary["normalizer_version"], "jiwer": summary["jiwer_version"]},
        "provenance": {"fyp_be_commit": _git("rev-parse", "HEAD"), "fyp_be_uncommitted_changes": bool(_git("status", "--porcelain") or ""),
                       "handler": metadata.get("handler"), "handler_environment": metadata.get("handler_environment"),
                       "region": metadata.get("region")},
        "transcribe_job": job_settings,
        "summary": summary, "calls": call_rows,
    }
    if v2:
        run["audio_condition"], run["base_dataset_id"] = V2_AUDIO, dataset_id
        comparison = _comparison(run_dir)
        if comparison is not None:
            run["comparison"] = comparison
    (bundle / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (bundle / "errors.json").write_text(json.dumps(frequent_errors(calls_units), ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(scores / "unnormalised.json", bundle / "unnormalised.json")
    shutil.copy2(scores / "confusions.json", bundle / "confusions.json")
    if (scores / "changes_from_1.2.0.json").is_file():  # what the scoring-rule change did to this run (scripts/compare_scores.py)
        shutil.copy2(scores / "changes_from_1.2.0.json", bundle / "changes_from_1.2.0.json")
    return bundle
