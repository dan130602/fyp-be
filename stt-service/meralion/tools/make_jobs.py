"""Write the jobs files for the Stage 3 live runs (runs on the Origami login node; CPU, standard library only; never part of the image).

    python make_jobs.py --batch ~/meralion-pilot/batch --pilot pilot_calls_overlap_v1.json --out <dir>

  jobs-pilot22.json   {"jobs": [{"job_name": <call id>, "input": <absolute audio path>}]}  the 22 development calls of the pilot list, in its order
  jobs-c4.json        the shortest of the 22 (the like-for-like one-recording reference, plan C4)

    python make_jobs.py --batch ~/meralion-pilot/batch --all-calls --out <dir>          (full development validation)
  jobs-dev175.json    every call of batch/calls.json in the manifest's own order (the order of the frozen development run), audio sha256 checked first
    python make_jobs.py --batch <dir with calls.json> --audit-predictions <frozen run's raw/predictions>      (CPU, no audio: every manifest audio hash equals the one the frozen run recorded)

Reads batch/calls.json (call id, audio path, audio sha256) and the pilot list's call ids only: no reference text, no scores. Every audio file's sha256 is checked against the manifest first.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


class JobsError(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pilot_call_ids(pilot):
    per_call = pilot.get("per_call")
    if not isinstance(per_call, dict) or not per_call:
        raise JobsError("the pilot list has no per_call table")
    ids = list(per_call)
    if len(ids) != len(set(ids)):
        raise JobsError("duplicate call ids in the pilot list")
    return ids


def build_jobs(batch_dir, pilot, *, verify=True):
    """Returns (jobs for the pilot calls in pilot order, the shortest call id, {call: audio_seconds from the pilot list})."""
    batch_dir = Path(batch_dir)
    manifest = json.loads((batch_dir / "calls.json").read_text(encoding="utf-8"))
    by_id = {row["call_id"]: row for row in manifest["calls"]}
    jobs = []
    for call_id in pilot_call_ids(pilot):
        if call_id not in by_id:
            raise JobsError(f"{call_id} is not in {batch_dir / 'calls.json'}")
        row = by_id[call_id]
        path = (batch_dir / row["audio"]).resolve()
        if not path.is_file():
            raise JobsError(f"missing audio file {path}")
        if verify and sha256_file(path) != row["audio_sha256"]:
            raise JobsError(f"{call_id}: the audio file does not match the manifest sha256")
        jobs.append({"job_name": call_id, "input": str(path)})
    seconds = {call: float(info["audio_seconds"]) for call, info in pilot["per_call"].items()}
    shortest = min(seconds, key=lambda call: (seconds[call], call))
    return jobs, shortest, seconds


def build_all_jobs(batch_dir, *, verify=True):
    """Every call of batch/calls.json, in the manifest's order, as jobs (audio sha256 checked unless verify is False)."""
    batch_dir = Path(batch_dir)
    manifest = json.loads((batch_dir / "calls.json").read_text(encoding="utf-8"))
    ids = [row["call_id"] for row in manifest["calls"]]
    if len(ids) != len(set(ids)):
        raise JobsError("duplicate call ids in calls.json")
    jobs = []
    for row in manifest["calls"]:
        path = (batch_dir / row["audio"]).resolve()
        if not path.is_file():
            raise JobsError(f"missing audio file {path}")
        if verify and sha256_file(path) != row["audio_sha256"]:
            raise JobsError(f"{row['call_id']}: the audio file does not match the manifest sha256")
        jobs.append({"job_name": row["call_id"], "input": str(path)})
    return jobs


def audit_predictions(batch_dir, predictions_dir):
    """Problems (empty means none) between the manifest's audio hashes and the hashes a scored run recorded in its predictions (audio itself is not read)."""
    manifest = json.loads((Path(batch_dir) / "calls.json").read_text(encoding="utf-8"))
    expected = {row["call_id"]: row["audio_sha256"] for row in manifest["calls"]}
    seen, problems = {}, []
    for path in sorted(Path(predictions_dir).glob("*.json")):
        prediction = json.loads(path.read_text(encoding="utf-8"))
        seen[prediction["call_id"]] = (prediction.get("settings") or {}).get("audio_sha256")
    if sorted(seen) != sorted(expected):
        problems.append(f"the recordings differ: manifest-only {sorted(set(expected) - set(seen))[:5]}, predictions-only {sorted(set(seen) - set(expected))[:5]}")
    for call, digest in sorted(seen.items()):
        if call in expected and digest != expected[call]:
            problems.append(f"{call}: the run recorded audio {digest!r}, the manifest has {expected[call]!r}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--all-calls", action="store_true", help="write jobs-dev175.json (every call of calls.json, manifest order)")
    parser.add_argument("--audit-predictions", type=Path, help="only compare the manifest's audio hashes with a scored run's predictions")
    parser.add_argument("--no-verify", action="store_true", help="skip the audio sha256 check (tests only)")
    args = parser.parse_args(argv)
    if args.audit_predictions:
        problems = audit_predictions(args.batch, args.audit_predictions)
        print(json.dumps({"problems": problems[:10], "count": len(problems)}))
        return 3 if problems else 0
    if args.out is None or (args.pilot is None) == (not args.all_calls):
        print("make_jobs: --out and exactly one of --pilot or --all-calls are required", file=sys.stderr)
        return 3
    if args.all_calls:
        try:
            jobs = build_all_jobs(args.batch, verify=not args.no_verify)
        except (JobsError, OSError, KeyError, ValueError) as error:
            print(f"make_jobs: {error}", file=sys.stderr)
            return 3
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "jobs-dev175.json").write_text(json.dumps({"jobs": jobs}, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"calls": len(jobs)}))
        return 0
    try:
        pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
        jobs, shortest, seconds = build_jobs(args.batch, pilot, verify=not args.no_verify)
    except (JobsError, OSError, KeyError, ValueError) as error:
        print(f"make_jobs: {error}", file=sys.stderr)
        return 3
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "jobs-pilot22.json").write_text(json.dumps({"jobs": jobs}, indent=1) + "\n", encoding="utf-8")
    (args.out / "jobs-c4.json").write_text(json.dumps({"jobs": [job for job in jobs if job["job_name"] == shortest]}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pilot_calls": len(jobs), "shortest": shortest, "shortest_seconds": seconds[shortest], "total_audio_hours": round(sum(seconds.values()) / 3600, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
