"""The GPU-time ledger of the Stage 3 follow-up (standard library only; runs on the Origami login node and locally in tests).

Every submitted job's ALLOCATED time limit is recorded and counted in full, even if the job fails or ends early. A job whose limit exceeds the remaining allowance is refused.

    python3 ledger.py check  <ledger.json> <cap_hours> <HH:MM:SS>
    python3 ledger.py record <ledger.json> <cap_hours> <HH:MM:SS> <job id> <script> <gpu>
    python3 ledger.py show   <ledger.json> <cap_hours>
    python3 ledger.py annotate <ledger.json> <job id> <status> <reason...>

`annotate` adds status, reason and a timestamp to the entry of a job that is already recorded (for example `cancelled`). It NEVER changes allocated_seconds and never removes the entry: a submission
that was cancelled after it was recorded still counts in full.
"""
import json
import sys
import time
from pathlib import Path


def seconds_of(limit):
    """'HH:MM:SS' or 'D-HH:MM:SS' or 'MM:SS' -> seconds."""
    days = 0
    if "-" in limit:
        day_part, limit = limit.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in limit.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, secs = parts
    return days * 86400 + hours * 3600 + minutes * 60 + secs


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"jobs": []}


def spent_seconds(ledger):
    return sum(job["allocated_seconds"] for job in ledger["jobs"])


def remaining_seconds(ledger, cap_hours):
    return int(cap_hours * 3600) - spent_seconds(ledger)


def check(path, cap_hours, limit):
    """-> (ok, remaining seconds, requested seconds)."""
    requested = seconds_of(limit)
    remaining = remaining_seconds(load(path), cap_hours)
    return requested <= remaining, remaining, requested


def record(path, cap_hours, limit, job_id, script, gpu):
    ok, remaining, requested = check(path, cap_hours, limit)
    if not ok:
        raise SystemExit(f"refused: {limit} exceeds the remaining allowance of {remaining} s")
    ledger = load(path)
    ledger["jobs"].append({"job": job_id, "script": script, "gpu": gpu, "allocated_seconds": requested, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    Path(path).write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")
    return ledger


def annotate(path, job_id, status, reason):
    """Add status, reason and `annotated` to the recorded entry of `job_id`. allocated_seconds is untouched; an unknown job id is an error (nothing is written)."""
    ledger = load(path)
    entries = [job for job in ledger["jobs"] if job["job"] == job_id]
    if len(entries) != 1:
        raise SystemExit(f"cannot annotate job {job_id!r}: {len(entries)} entries with that id in the ledger")
    entries[0].update(status=status, reason=reason, annotated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    Path(path).write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")
    return ledger


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0]
    if command == "annotate":
        annotate(argv[1], argv[2], argv[3], " ".join(argv[4:]))
        return 0
    path, cap = argv[1], float(argv[2])
    if command == "check":
        ok, remaining, requested = check(path, cap, argv[3])
        print(f"remaining {remaining} s, requested {requested} s: {'ok' if ok else 'REFUSED'}")
        return 0 if ok else 3
    if command == "record":
        record(path, cap, argv[3], argv[4], argv[5], argv[6])
        return 0
    if command == "show":
        ledger = load(path)
        print(json.dumps({"spent_hours": round(spent_seconds(ledger) / 3600, 3), "remaining_hours": round(remaining_seconds(ledger, cap) / 3600, 3), "jobs": ledger["jobs"]}, indent=1))
        return 0
    raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    sys.exit(main())
