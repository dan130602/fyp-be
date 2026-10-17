"""Nightly batch enqueuer (EventBridge Scheduler target). Only enqueues - it never processes or
deletes anything. Safe to re-run: a row is claimed PENDING -> QUEUED before its message is sent,
so a second run (or a Scheduler retry) skips everything already queued.

Manual run (dry_run lists what would be enqueued without changing anything):
    aws lambda invoke --function-name <enqueuer> --payload '{"dry_run": true}' \
        --cli-binary-format raw-in-base64-out out.json
"""
import json
import logging
import os
import uuid

import boto3

from app import db

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def _record_id_from_key(key):
    """Batch keys are "{record_id}/{filename}"; anything else in the bucket is ignored."""
    prefix, sep, rest = key.partition("/")
    if not sep or not rest:
        return None
    try:
        return str(uuid.UUID(prefix))
    except ValueError:
        return None


def handler(event, context):
    dry_run = bool((event or {}).get("dry_run"))
    bucket = os.environ["BATCH_BUCKET"]
    queue_url = os.environ["BATCH_QUEUE_URL"]

    s3 = boto3.client("s3")
    sqs = boto3.client("sqs")
    stats = {"listed": 0, "enqueued": 0, "skipped": 0, "failed": 0, "dry_run": dry_run}

    conn = db.connect()
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                stats["listed"] += 1

                record_id = _record_id_from_key(key)
                if record_id is None:
                    logger.warning("Ignoring s3://%s/%s: key is not '<record_id>/<filename>'", bucket, key)
                    stats["skipped"] += 1
                    continue

                if dry_run:
                    if db.get_status(conn, record_id) == "PENDING":
                        stats["enqueued"] += 1
                    else:
                        stats["skipped"] += 1
                    continue

                if not db.claim_pending(conn, record_id):
                    stats["skipped"] += 1
                    continue

                try:
                    sqs.send_message(
                        QueueUrl=queue_url,
                        MessageBody=json.dumps(
                            {"bucket": bucket, "key": key, "record_id": record_id, "source": "batch"}
                        ),
                    )
                    stats["enqueued"] += 1
                except Exception:
                    # Undo the claim so tonight's failure does not strand the row as QUEUED forever.
                    logger.exception("Failed to enqueue %s", key)
                    db.release_queued(conn, record_id)
                    stats["failed"] += 1
    finally:
        conn.close()

    logger.info("Batch enqueue finished: %s", stats)
    if stats["failed"]:
        # Surface it (Lambda error metric / Scheduler retry); re-running is idempotent.
        raise RuntimeError(f"Failed to enqueue {stats['failed']} object(s): {stats}")
    return stats
