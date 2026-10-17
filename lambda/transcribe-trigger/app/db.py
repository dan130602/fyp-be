"""Minimal Postgres access for the batch Lambdas (enqueuer + batch worker).

Only the batch path touches the DB - the immediate S3-triggered Lambda deliberately stays
DB-free. All status writes are conditional UPDATEs so concurrent/duplicate runs are safe.
"""
import json
import os
import ssl

import boto3


def connect():
    # Imported here so unit tests and the immediate path don't need the driver installed.
    import pg8000.native

    secret = json.loads(
        boto3.client("secretsmanager").get_secret_value(SecretId=os.environ["DB_SECRET_ARN"])["SecretString"]
    )

    # Encrypts the connection without verifying the server cert (equivalent to sslmode=require,
    # which is also what the Spring service's default JDBC sslmode amounts to). The RDS CA bundle
    # isn't in the Lambda base image; to verify, bake it into the image and load it here.
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    return pg8000.native.Connection(
        user=secret["username"],
        password=secret["password"],
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ["DB_NAME"],
        ssl_context=ssl_context,
        timeout=10,
    )


def get_status(conn, record_id):
    """Status of a BATCH row, or None if there is no such row."""
    rows = conn.run(
        "SELECT status FROM t_media_asset WHERE id = :id AND processing_mode = 'BATCH'",
        id=record_id,
    )
    return rows[0][0] if rows else None


def claim_pending(conn, record_id):
    """PENDING -> QUEUED. True only for the caller that actually flipped it, so two overlapping
    enqueuer runs can never both send a message for the same record."""
    rows = conn.run(
        "UPDATE t_media_asset SET status = 'QUEUED', updated_at = now() "
        "WHERE id = :id AND processing_mode = 'BATCH' AND status = 'PENDING' RETURNING id",
        id=record_id,
    )
    return bool(rows)


def release_queued(conn, record_id):
    """QUEUED -> PENDING, undoing claim_pending when the SQS send failed."""
    conn.run(
        "UPDATE t_media_asset SET status = 'PENDING', updated_at = now() "
        "WHERE id = :id AND processing_mode = 'BATCH' AND status = 'QUEUED'",
        id=record_id,
    )


def mark_processing(conn, record_id):
    """-> PROCESSING for anything not already COMPLETED. False means skip the message: the row is
    COMPLETED (SQS is at-least-once, so duplicates are expected) or does not exist."""
    rows = conn.run(
        "UPDATE t_media_asset SET status = 'PROCESSING', failure_reason = NULL, updated_at = now() "
        "WHERE id = :id AND processing_mode = 'BATCH' AND status <> 'COMPLETED' RETURNING id",
        id=record_id,
    )
    return bool(rows)


def mark_failed(conn, record_id, reason):
    conn.run(
        "UPDATE t_media_asset SET status = 'FAILED', failure_reason = :reason, updated_at = now() "
        "WHERE id = :id AND processing_mode = 'BATCH' AND status <> 'COMPLETED'",
        id=record_id,
        reason=reason,
    )
