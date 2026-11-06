"""Collect phase: send the dataset's audio through the production handler and bring Transcribe's output back to this machine.

For every call: keep a master copy of the audio in S3 (uploaded once, re-uploaded only when its sha256 changes), copy it inside S3 to this run's
own key, start the job by calling the UNMODIFIED production handler with an S3 event, poll the job with `get_transcription_job`, download the
transcript to `runs/<run id>/raw/transcripts/CALL_xxx.json` and delete the run's temporary copies. `metadata.json` records what ran: the settings
Transcribe reports for each job, the handler's git commit, the audio hashes and the timestamps. Nothing secret is written.

Scoring never needs AWS: it reads the saved `runs/<run id>/` folder.
"""

import hashlib
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

from .config import ENGINE_ROOT, FYP_BE_ROOT
from .events import master_key, run_key, s3_event

TERMINAL = {"COMPLETED", "FAILED"}
RUNS_ROOT = ENGINE_ROOT / "runs"


class CollectError(RuntimeError):
    pass


def new_run_id(now=None):
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_id_of(dataset_root):
    manifest = Path(dataset_root) / "dataset_manifest.json"
    if not manifest.is_file():
        raise CollectError(f"no dataset_manifest.json in {dataset_root}")
    return json.loads(manifest.read_text(encoding="utf-8"))["dataset_id"]


def discover_calls(dataset_root):
    return sorted(path.name for path in (Path(dataset_root) / "calls").glob("CALL_*") if path.is_dir())


AUTO_RENDER = "auto"  # use the call's own single render folder (a release mixes Polly mp3 with Gemini / OmniVoice wav renders)


def local_audio(dataset_root, call_id, render_id):
    renders = Path(dataset_root) / "calls" / call_id / "renders"
    if render_id == AUTO_RENDER:
        found = sorted(p for p in renders.glob("*/audio.*") if p.suffix in (".mp3", ".wav")) if renders.is_dir() else []
        if len(found) != 1:
            raise CollectError(f"{call_id}: --render-id auto needs exactly one audio.mp3 / audio.wav under {renders}, found {len(found)}")
        return found[0]
    path = renders / render_id / "audio.mp3"
    if not path.is_file():
        raise CollectError(f"{call_id}: no audio at {path} (render it first; the mp3 files are not committed)")
    return path


def _render_and_extension(path):
    return path.parent.name, path.suffix.lstrip(".")


def handler_provenance(handler_path):
    """Which handler ran: its git commit, whether it has uncommitted edits, and a hash of the exact file."""
    info = {"path": str(Path(handler_path).relative_to(FYP_BE_ROOT)) if Path(handler_path).is_relative_to(FYP_BE_ROOT) else str(handler_path),
            "sha256": sha256_file(handler_path), "git_commit": None, "uncommitted_changes": None}
    try:
        folder = str(Path(handler_path).parent)
        info["git_commit"] = subprocess.run(["git", "-C", str(FYP_BE_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(FYP_BE_ROOT), "status", "--porcelain", "--", folder], capture_output=True, text=True, check=True).stdout
        info["uncommitted_changes"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def plan(dataset_root, render_id, call_ids, bucket):
    """What a collect would do, without any AWS access: keys, sizes, hashes and audio minutes."""
    dataset_id = dataset_id_of(dataset_root)
    rows = []
    for call_id in call_ids:
        path = local_audio(dataset_root, call_id, render_id)
        timings = path.with_name("timings.json")
        seconds = json.loads(timings.read_text(encoding="utf-8")).get("duration_seconds") if timings.is_file() else None
        call_render, extension = _render_and_extension(path)
        rows.append({"call_id": call_id, "render_id": call_render, "bytes": path.stat().st_size, "audio_seconds": seconds,
                     "sha256": sha256_file(path), "master_key": master_key(dataset_id, call_render, call_id, extension)})
    return {"dataset_id": dataset_id, "render_id": render_id, "bucket": bucket, "calls": rows,
            "audio_minutes": round(sum(r["audio_seconds"] or 0 for r in rows) / 60, 1)}


def _ensure_master(s3, bucket, key, path, digest):
    content_type = "audio/wav" if path.suffix == ".wav" else "audio/mpeg"
    try:
        if s3.head_object(Bucket=bucket, Key=key).get("Metadata", {}).get("sha256") == digest:
            return "reused"
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
    s3.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}, "ContentType": content_type})
    return "uploaded"


def _job_record(job):
    return json.loads(json.dumps(job, default=str))


def collect(settings, handler, s3, *, dataset_root, render_id, call_ids, run_id=None, runs_root=RUNS_ROOT, poll_seconds=5,
            timeout_seconds=1800, sleep=time.sleep, clock=time.monotonic, progress=print, keep_uploads=False):
    """Run the collect phase; returns the metadata dict (also saved as `metadata.json`). Raises CollectError before anything is uploaded
    when an input is missing. Per-call job failures are recorded, not raised."""
    dataset_root, bucket = Path(dataset_root), settings.bucket
    dataset_id = dataset_id_of(dataset_root)
    run_id = run_id or new_run_id()
    run_dir = Path(runs_root) / run_id
    if run_dir.exists():
        raise CollectError(f"run folder already exists: {run_dir}")
    audio = {call_id: local_audio(dataset_root, call_id, render_id) for call_id in call_ids}
    digests = {call_id: sha256_file(path) for call_id, path in audio.items()}
    (run_dir / "raw" / "transcripts").mkdir(parents=True)

    transcribe = handler.transcribe
    keys = {call_id: run_key(run_id, dataset_id, call_id, _render_and_extension(audio[call_id])[1]) for call_id in call_ids}
    entries = {call_id: {"call_id": call_id, "render_id": _render_and_extension(audio[call_id])[0], "audio_bytes": audio[call_id].stat().st_size, "audio_sha256": digests[call_id],
                         "master_key": master_key(dataset_id, _render_and_extension(audio[call_id])[0], call_id, _render_and_extension(audio[call_id])[1]), "run_key": keys[call_id], "job_name": None,
                         "status": "NOT_STARTED", "failure_reason": None, "transcript_file": None, "job": None} for call_id in call_ids}
    metadata = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None, "status": "running", "error": None,
                "region": settings.region, "bucket": bucket, "dataset_id": dataset_id, "render_id": render_id,
                "handler": handler_provenance(settings.handler_path), "handler_environment": settings.handler_environment(),
                "calls": list(entries.values())}
    still_running = False
    try:
        for call_id in call_ids:
            state = _ensure_master(s3, bucket, entries[call_id]["master_key"], audio[call_id], digests[call_id])
            s3.copy_object(Bucket=bucket, Key=keys[call_id], CopySource={"Bucket": bucket, "Key": entries[call_id]["master_key"]},
                           MetadataDirective="COPY")
            progress(f"{call_id}: master audio {state}; run copy made")

        started = handler.handler(s3_event(bucket, [keys[c] for c in call_ids]), None)["started_jobs"]
        by_key = {key: call_id for call_id, key in keys.items()}
        for name in started:  # match each job to its call through the media address Transcribe reports
            job = transcribe.get_transcription_job(TranscriptionJobName=name)["TranscriptionJob"]
            uri = job["Media"]["MediaFileUri"]
            call_id = next((by_key[key] for key in by_key if uri.endswith("/" + key)), None)
            if call_id is None:
                raise CollectError(f"job {name} points at an unexpected file: {uri}")
            entries[call_id].update(job_name=name, status=job["TranscriptionJobStatus"], job=_job_record(job))
        if missing := [c for c, e in entries.items() if e["job_name"] is None]:
            raise CollectError(f"the handler started no job for: {', '.join(missing)}")
        progress(f"started {len(started)} Transcribe job(s) through the production handler")

        deadline = clock() + timeout_seconds
        while True:
            pending = [e for e in entries.values() if e["status"] not in TERMINAL]
            if not pending:
                break
            if clock() > deadline:
                still_running = True
                for entry in pending:
                    entry["status"] = "TIMEOUT"
                break
            sleep(poll_seconds)
            for entry in pending:
                job = transcribe.get_transcription_job(TranscriptionJobName=entry["job_name"])["TranscriptionJob"]
                entry.update(status=job["TranscriptionJobStatus"], job=_job_record(job))
                if entry["status"] in TERMINAL:
                    progress(f"{entry['call_id']}: {entry['status']}")

        for entry in entries.values():
            if entry["status"] == "FAILED":
                entry["failure_reason"] = (entry["job"] or {}).get("FailureReason")
            elif entry["status"] == "COMPLETED":
                body = s3.get_object(Bucket=bucket, Key=f"transcripts/{entry['job_name']}.json")["Body"].read()
                json.loads(body)  # refuse to save something that is not JSON
                target = run_dir / "raw" / "transcripts" / f"{entry['call_id']}.json"
                target.write_bytes(body)
                entry["transcript_file"] = str(target.relative_to(run_dir))
        done = sum(e["status"] == "COMPLETED" for e in entries.values())
        metadata["status"] = "completed" if done == len(entries) else "completed_with_failures"
    except Exception as error:  # noqa: BLE001 - recorded in metadata, then re-raised
        metadata["status"], metadata["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        if not keep_uploads and not still_running:
            for key in keys.values():
                try:
                    s3.delete_object(Bucket=bucket, Key=key)
                except ClientError:
                    pass
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["calls"] = list(entries.values())
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return metadata
