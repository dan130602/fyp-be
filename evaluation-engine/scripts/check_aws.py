"""Read-only connection check: who am I, can I reach the evaluation bucket, may I use Transcribe. Prints no secrets and changes nothing.

    python -B scripts/check_aws.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from transcribe_eval.config import ConfigError, load_settings


def _check(label, call):
    try:
        detail = call()
    except ClientError as error:
        print(f"FAIL  {label}: {error.response['Error']['Code']}")
        return False
    except BotoCoreError as error:
        print(f"FAIL  {label}: {type(error).__name__}")
        return False
    print(f"ok    {label}{': ' + detail if detail else ''}")
    return True


def main():
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Configuration problem: {error}")
        return 2
    settings.apply_to_environ()
    session = boto3.Session(region_name=settings.region)
    print(f"region {settings.region}, bucket {settings.bucket}")

    def identity():
        who = session.client("sts").get_caller_identity()
        return f"account {who['Account']}, {who['Arn']}"

    def bucket_reachable():
        session.client("s3").head_bucket(Bucket=settings.bucket)

    def list_objects():
        found = session.client("s3").list_objects_v2(Bucket=settings.bucket, MaxKeys=1)["KeyCount"]
        return f"{found} object(s) seen"

    def transcribe_access():
        session.client("transcribe").list_transcription_jobs(MaxResults=1)

    results = [
        _check("AWS identity", identity),
        _check("bucket reachable", bucket_reachable),
        _check("list objects", list_objects),
        _check("Transcribe access", transcribe_access),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
