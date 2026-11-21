"""Exercise the worker's real S3Storage against a TEST bucket WITHOUT constructing a decoder (so it needs no GPU and loads no model). Verification only; never part of the image's entrypoint.

    python s3_storage_check.py --input-uri s3://<bucket>/<allowed input key> --state-bucket <bucket> --state-prefix meralion-state/s3-check-<unique id> [--region ap-southeast-1] [--out report.json]

Run it INSIDE the image so the worker code and boto3 are exactly the shipped ones, with the Batch job role's credentials (or the instance role):

    docker run --rm -e AWS_REGION=ap-southeast-1 -v <this file>:/checks/s3_storage_check.py:ro --entrypoint python -e PYTHONPATH=/opt/meralion <image> /checks/s3_storage_check.py ...

What it checks, in this order (each is a named result; the exit code is 0 only when all hold):
  download_input         the input object is downloaded (s3:GetObject on the input prefix) and is non-empty
  missing_key_is_absent  `exists` on a key that is not there answers False (a 404; this needs s3:ListBucket, otherwise S3 answers 403 and the worker raises, which is the safe behaviour)
  conditional_write      `write_exclusive` creates a new object under the STATE prefix (s3:PutObject with If-None-Match)
  read_back_equals       the object read back (s3:GetObject) equals what was written
  exists_after_write     `exists` answers True
  second_write_refused   a second `write_exclusive` of the same key raises AlreadyExists (the conditional write holds; nothing is overwritten)
Only keys under --state-prefix are written (the prefix the job role may write). It writes nothing under transcripts/ and deletes nothing (the worker role has no delete permission by design): the objects it
creates are tiny and are removed by the test bucket's lifecycle rule. It cannot say anything about the decoder, the GPU or the transcripts.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path


def run_checks(storage, input_uri, state_storage, check_key):
    """-> {name: {"ok": bool, "detail": str}}. `storage` downloads the input; `state_storage` is an S3Storage rooted at the state prefix."""
    from meralion_worker import storage as worker_storage
    results = {}

    def record(name, ok, detail=""):
        results[name] = {"ok": bool(ok), "detail": str(detail)[:300]}

    try:
        with tempfile.TemporaryDirectory() as folder:
            path = storage.download(input_uri, folder)
            size = Path(path).stat().st_size
            record("download_input", size > 0, f"{size} bytes, sha256 {hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]}")
    except Exception as error:                                                                                   # noqa: BLE001 - every failure is a recorded fact
        record("download_input", False, f"{type(error).__name__}: {error}")
    missing = f"{check_key}.absent"
    try:
        record("missing_key_is_absent", state_storage.exists(missing) is False, "exists() on a key that was never written")
    except Exception as error:                                                                                   # noqa: BLE001
        record("missing_key_is_absent", False, f"{type(error).__name__}: {error} (s3:ListBucket on the bucket is probably missing)")
    payload = json.dumps({"check": "meralion s3 storage", "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}).encode("utf-8")
    try:
        state_storage.write_exclusive(check_key, payload)
        record("conditional_write", True, f"created {check_key}")
    except Exception as error:                                                                                   # noqa: BLE001
        record("conditional_write", False, f"{type(error).__name__}: {error}")
    try:
        record("read_back_equals", state_storage.read(check_key) == payload, "get_object of the object just written")
    except Exception as error:                                                                                   # noqa: BLE001
        record("read_back_equals", False, f"{type(error).__name__}: {error}")
    try:
        record("exists_after_write", state_storage.exists(check_key) is True)
    except Exception as error:                                                                                   # noqa: BLE001
        record("exists_after_write", False, f"{type(error).__name__}: {error}")
    try:
        state_storage.write_exclusive(check_key, b"a second, different body")
        record("second_write_refused", False, "the second write was ACCEPTED: the conditional write does not protect existing objects")
    except worker_storage.AlreadyExists:
        record("second_write_refused", True, "AlreadyExists, as required")
    except Exception as error:                                                                                   # noqa: BLE001
        record("second_write_refused", False, f"{type(error).__name__}: {error}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--state-bucket", required=True)
    parser.add_argument("--state-prefix", required=True, help="e.g. meralion-state/s3-check-<unique id>; must be a prefix the role may write")
    parser.add_argument("--region", default=None)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    from meralion_worker import storage as worker_storage
    bucket, _ = worker_storage.parse_s3_uri(args.input_uri)
    results = run_checks(worker_storage.S3Storage(bucket, "", region=args.region), args.input_uri, worker_storage.S3Storage(args.state_bucket, args.state_prefix, region=args.region), "check.json")
    report = {"input_uri": args.input_uri, "state": f"s3://{args.state_bucket}/{args.state_prefix.strip('/')}/", "results": results, "passed": all(item["ok"] for item in results.values()),
              "scope": "S3Storage only: no decoder, no GPU, no model; says nothing about transcripts or the worker's end-to-end run"}
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
