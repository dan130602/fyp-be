"""Consumer for the batch SQS queue. Runs the SAME job-start logic as the immediate path
(handler._start_job), adding only the batch bookkeeping around it.

The Transcribe job outlives this invocation, so "completed" and the batch-bucket cleanup happen
later, in media-processing-service, when the job's completion event arrives (see
MediaAssetService.applyTranscribeStatusUpdate). What this worker owns is QUEUED -> PROCESSING,
idempotency, and FAILED once retries are exhausted.
"""
import json
import logging
import os

from botocore.exceptions import ClientError

from app import db, handler

logger = logging.getLogger()


def _start_batch_job(bucket, key):
    job_name = handler._batch_job_name_for_key(key)
    try:
        return handler._start_job(bucket, key, job_name=job_name)
    except ClientError as exc:
        # Redelivery after the job was already started (e.g. we crashed before acking): the
        # deterministic job name makes this a duplicate, not a failure.
        if exc.response.get("Error", {}).get("Code") == "ConflictException":
            logger.info("Transcribe job %s already exists for s3://%s/%s; treating as started", job_name, bucket, key)
            return job_name
        raise


def _process(conn, record):
    body = json.loads(record["body"])
    bucket, key, record_id = body["bucket"], body["key"], body["record_id"]

    if body.get("source") != "batch":
        logger.error("Dropping message %s: unexpected source %r", record["messageId"], body.get("source"))
        return

    if not db.mark_processing(conn, record_id):
        logger.info("Skipping %s: not found or already COMPLETED (duplicate delivery)", record_id)
        return

    try:
        job_name = _start_batch_job(bucket, key)
    except Exception as exc:
        # Must match the queue's maxReceiveCount (both come from the same Terraform variable):
        # on the last allowed receive this message goes to the DLQ next, so record the failure now.
        max_receives = int(os.environ.get("BATCH_MAX_RECEIVE_COUNT", "5"))
        receives = int(record.get("attributes", {}).get("ApproximateReceiveCount", "1"))
        if receives >= max_receives:
            db.mark_failed(conn, record_id, f"Failed to start transcription after {receives} attempts: {exc}"[:1000])
        raise

    if job_name is None:
        # Unsupported media format: retrying cannot help, so fail it now instead of burning retries.
        db.mark_failed(conn, record_id, "Unsupported or unknown media format")


def handle(event):
    """Returns SQS partial-batch-failure format, so one bad record does not re-run its siblings."""
    failures = []
    conn = db.connect()
    try:
        for record in event["Records"]:
            try:
                _process(conn, record)
            except Exception:
                logger.exception("Batch record %s failed", record.get("messageId"))
                failures.append({"itemIdentifier": record["messageId"]})
    finally:
        conn.close()
    return {"batchItemFailures": failures}
