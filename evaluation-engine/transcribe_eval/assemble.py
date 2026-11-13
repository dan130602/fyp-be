"""Assemble ONE complete environment_v2 run from (a) a V2 run that skipped the clean controls and (b) the earlier clean run.

The V2 run transcribed only the changed audio (collect.py --audio-condition environment_v2 --skip-clean-controls). The clean (C1) calls
are the original recordings, so their existing clean transcripts are copied into the new run folder and score.py can score all calls
together. Offline: reads and copies local run folders only, never AWS.

Honest provenance: the assembled run keeps the V2 run's own start and finish times (the real run date). Each reused call keeps its ORIGINAL
Transcribe job record and times and gains reused_from_run, so nobody can mistake a copied transcript for a fresh one.
Reuse is refused unless the handler file, the TRANSCRIBE_* settings and the Transcribe job settings match: otherwise the clean controls
must be transcribed again.
"""

import json
import shutil
from pathlib import Path

from .compare import CompareError, check_equivalent, load_metadata_only


def assemble(runs_root, v2_run_id, clean_run_id, out_run_id):
    runs_root = Path(runs_root)
    v2, clean = load_metadata_only(runs_root, v2_run_id), load_metadata_only(runs_root, clean_run_id)
    if v2["metadata"].get("audio_condition") != "environment_v2" or not v2["metadata"].get("skipped_clean_controls"):
        raise CompareError(f"{v2_run_id} is not an environment_v2 run that skipped the clean controls")
    if clean["metadata"].get("audio_condition", "clean") != "clean":
        raise CompareError(f"{clean_run_id} is not a clean run")
    equivalent, reasons, warnings = check_equivalent(clean, v2)
    if reasons:
        raise CompareError("clean controls must be re-transcribed: " + "; ".join(reasons))
    out = runs_root / out_run_id
    if out.exists():
        raise CompareError(f"run folder already exists: {out}")
    clean_entries = {e["call_id"]: e for e in clean["metadata"]["calls"]}
    missing = [c for c in v2["metadata"]["skipped_clean_controls"]
               if clean_entries.get(c, {}).get("status") != "COMPLETED" or not (runs_root / clean_run_id / clean_entries[c]["transcript_file"]).is_file()]
    if missing:
        raise CompareError(f"the clean run has no completed transcript for: {', '.join(missing)}; transcribe those calls instead")
    (out / "raw" / "transcripts").mkdir(parents=True)
    entries = {}
    for entry in v2["metadata"]["calls"]:
        entries[entry["call_id"]] = dict(entry)
        if entry.get("transcript_file"):
            shutil.copyfile(runs_root / v2_run_id / entry["transcript_file"], out / entry["transcript_file"])
    reused = []
    for call_id in v2["metadata"]["skipped_clean_controls"]:
        source = clean_entries[call_id]
        target = f"raw/transcripts/{call_id}.json"
        shutil.copyfile(runs_root / clean_run_id / source["transcript_file"], out / target)
        entries[call_id] = dict(source, transcript_file=target, audio_condition="clean", v2_condition="C1", reused_from_run=clean_run_id,
                                original_run_started_at=clean["metadata"].get("started_at"))
        reused.append(call_id)
    metadata = dict(v2["metadata"], run_id=out_run_id, skipped_clean_controls=[], skip_clean_controls=False, reused_clean_controls=sorted(reused),
                    assembled_from={"v2_run": v2_run_id, "clean_run": clean_run_id}, calls=[entries[c] for c in sorted(entries)])
    done = all(e["status"] == "COMPLETED" for e in metadata["calls"])
    metadata["status"] = "completed" if done else "completed_with_failures"
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return {"run_id": out_run_id, "calls": len(entries), "reused_clean_calls": len(reused), "new_calls": len(entries) - len(reused),
            "warnings": warnings, "status": metadata["status"]}
