"""Gitignored operational store: attempts, canonical requests, raw provider responses, batch submissions, run lock.

Layout under `<audit_root>/<dataset_id>/<stage>/`:

    .lock
    attempts/<CALL_ID>/<A001>/{attempt.json, request.json, raw_response.json}
    submissions/SUB_0001.json            submission record (state history, batch snapshot, anomalies)
    submissions/SUB_0001.input.jsonl     exact bytes uploaded to the provider
    submissions/SUB_0001.output.jsonl    exact provider output file, as downloaded (when one exists)
    submissions/SUB_0001.error.jsonl     exact provider error file, as downloaded (when one exists)

This is working state and a debugging archive. Everything needed to reproduce or audit an accepted artifact is
also committed in the stage's provenance file, so losing this directory never invalidates an accepted result.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json

from .fsutil import write_bytes_atomic, write_text_atomic


class LockHeldError(ValueError):
    """Another run holds the lock (or a crashed run left it behind and it must be removed manually)."""


class StageWriteLockBusy(LockHeldError):
    """The short dataset write lock stayed busy for the whole wait: contention, or a crash leftover (see the message)."""


STALE_LOCK_ADVICE = ("Confirm that no G3/G4 process is active, and only then delete that file; "
                     "deleting a lock that is in use can corrupt a concurrent write.")


@contextmanager
def _exclusive_file(path, *, wait_seconds=0.0, poll_seconds=0.1, error=LockHeldError):
    """Hold an exclusively created file for the duration of the block; never follows a symlink; removes it afterwards."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing symlink lock path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            handle = path.open("x", encoding="utf-8")
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise error(f"Another run holds {path}. {STALE_LOCK_ADVICE}") from None
            time.sleep(poll_seconds)
    with handle:
        handle.write(f"pid {os.getpid()}\n")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


WRITE_LOCK_NAME = ".stage_write.lock"


def write_lock_path(dataset_root):
    """The shared write lock of one dataset, keyed by its RESOLVED root (independent of any --audit-dir)."""
    return Path(dataset_root).resolve() / WRITE_LOCK_NAME


@contextmanager
def dataset_write_lock(dataset_root, *, wait_seconds=10.0, poll_seconds=0.1):
    """Short cross-stage lock around 'verify upstream -> validate -> write accepted artifact'. Never held across provider calls.

    Lock order everywhere: the stage's own audit lock first, then this lock. Raises StageWriteLockBusy if it stays busy.
    """
    with _exclusive_file(write_lock_path(dataset_root), wait_seconds=wait_seconds, poll_seconds=poll_seconds, error=StageWriteLockBusy):
        yield


def attempt_label(number):
    return f"A{number:03d}"


def submission_label(number):
    return f"SUB_{number:04d}"


class AttemptStore:
    def __init__(self, audit_root, dataset_id, stage):
        self.root = Path(audit_root) / dataset_id / stage
        self.stage = stage

    # ---- lock ----------------------------------------------------------------------------------------
    @contextmanager
    def lock(self):
        with _exclusive_file(self.root / ".lock"):
            yield

    # ---- attempts ------------------------------------------------------------------------------------
    def attempt_dir(self, call_id, attempt_id):
        return self.root / "attempts" / call_id / attempt_id

    def attempt_ids(self, call_id):
        directory = self.root / "attempts" / call_id
        if not directory.is_dir():
            return []
        return sorted(entry.name for entry in directory.iterdir() if entry.is_dir() and (entry / "attempt.json").is_file())

    def read_attempt(self, call_id, attempt_id):
        return parse_json_text((self.attempt_dir(call_id, attempt_id) / "attempt.json").read_text(encoding="utf-8"))

    def write_attempt(self, attempt):
        write_text_atomic(self.attempt_dir(attempt["call_id"], attempt["attempt_id"]) / "attempt.json", serialize_json(attempt))

    def write_request(self, call_id, attempt_id, text):
        write_text_atomic(self.attempt_dir(call_id, attempt_id) / "request.json", text)

    def read_request(self, call_id, attempt_id):
        path = self.attempt_dir(call_id, attempt_id) / "request.json"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def write_raw_response(self, call_id, attempt_id, value):
        write_text_atomic(self.attempt_dir(call_id, attempt_id) / "raw_response.json", serialize_json(value))

    def read_raw_response(self, call_id, attempt_id):
        path = self.attempt_dir(call_id, attempt_id) / "raw_response.json"
        return parse_json_text(path.read_text(encoding="utf-8")) if path.is_file() else None

    # ---- submissions ---------------------------------------------------------------------------------
    def submission_ids(self):
        directory = self.root / "submissions"
        if not directory.is_dir():
            return []
        return sorted(entry.stem for entry in directory.iterdir() if entry.suffix == ".json")

    def next_submission_number(self):
        return max((int(name.split("_")[1]) for name in self.submission_ids()), default=0) + 1

    def read_submission(self, submission_id):
        return parse_json_text((self.root / "submissions" / f"{submission_id}.json").read_text(encoding="utf-8"))

    def write_submission(self, submission):
        write_text_atomic(self.root / "submissions" / f"{submission['submission_id']}.json", serialize_json(submission))

    def write_input_jsonl(self, submission_id, text):
        write_text_atomic(self.root / "submissions" / f"{submission_id}.input.jsonl", text)

    def read_input_jsonl(self, submission_id):
        return (self.root / "submissions" / f"{submission_id}.input.jsonl").read_text(encoding="utf-8")

    def result_file_path(self, submission_id, kind):
        """Exact provider result file for a submission: kind is 'output' or 'error'."""
        return self.root / "submissions" / f"{submission_id}.{kind}.jsonl"

    def write_result_file(self, submission_id, kind, data):
        """Retain the downloaded provider file byte for byte, before anything is parsed from it."""
        write_bytes_atomic(self.result_file_path(submission_id, kind), data)


def canonical_request_text(body):
    """Canonical request bytes: sorted keys, compact separators, UTF-8. The request hash is taken over this text."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
