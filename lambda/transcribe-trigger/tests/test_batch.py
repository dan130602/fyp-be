import hashlib
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app import batch_worker, enqueuer, handler

RECORD_ID = str(uuid.uuid4())
KEY = f"{RECORD_ID}/call.mp3"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_BUCKET", "output-bucket")
    monkeypatch.setenv("BATCH_BUCKET", "batch-bucket")
    monkeypatch.setenv("BATCH_QUEUE_URL", "https://sqs.example/batch")
    monkeypatch.setenv("BATCH_MAX_RECEIVE_COUNT", "3")


@pytest.fixture
def fake_db(monkeypatch):
    conn = MagicMock(name="conn")
    mock = MagicMock(name="db")
    mock.connect.return_value = conn
    monkeypatch.setattr(batch_worker, "db", mock)
    monkeypatch.setattr(enqueuer, "db", mock)
    return mock


def sqs_record(key=KEY, record_id=RECORD_ID, source="batch", receive_count=1, message_id="m1"):
    return {
        "messageId": message_id,
        "eventSource": "aws:sqs",
        "attributes": {"ApproximateReceiveCount": str(receive_count)},
        "body": json.dumps({"bucket": "batch-bucket", "key": key, "record_id": record_id, "source": source}),
    }


# -- job name -----------------------------------------------------------------

def test_batch_job_name_uses_record_id_segment_and_key_digest():
    digest = hashlib.sha256(KEY.encode("utf-8")).hexdigest()[:16]

    assert handler._batch_job_name_for_key(KEY) == f"transcribe-{RECORD_ID}-{digest}"


# -- handler dispatch -----------------------------------------------------------

def test_handler_routes_sqs_events_to_batch_worker():
    event = {"Records": [sqs_record()]}

    with patch.object(batch_worker, "handle", return_value={"batchItemFailures": []}) as mock_handle:
        result = handler.handler(event, context=None)

    mock_handle.assert_called_once_with(event)
    assert result == {"batchItemFailures": []}


# -- batch worker ---------------------------------------------------------------

def test_worker_marks_processing_then_starts_job_with_batch_job_name(fake_db):
    fake_db.mark_processing.return_value = True

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = batch_worker.handle({"Records": [sqs_record()]})

    assert result == {"batchItemFailures": []}
    fake_db.mark_processing.assert_called_once_with(fake_db.connect.return_value, RECORD_ID)
    kwargs = mock_transcribe.start_transcription_job.call_args.kwargs
    assert kwargs["TranscriptionJobName"] == handler._batch_job_name_for_key(KEY)
    assert kwargs["Media"]["MediaFileUri"] == f"s3://batch-bucket/{KEY}"
    fake_db.mark_failed.assert_not_called()
    fake_db.connect.return_value.close.assert_called_once()


def test_worker_skips_already_completed_record(fake_db):
    fake_db.mark_processing.return_value = False

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = batch_worker.handle({"Records": [sqs_record()]})

    assert result == {"batchItemFailures": []}
    mock_transcribe.start_transcription_job.assert_not_called()


def test_worker_treats_existing_job_as_already_started(fake_db):
    fake_db.mark_processing.return_value = True
    conflict = ClientError({"Error": {"Code": "ConflictException", "Message": "exists"}}, "StartTranscriptionJob")

    with patch.object(handler, "transcribe") as mock_transcribe:
        mock_transcribe.start_transcription_job.side_effect = conflict
        result = batch_worker.handle({"Records": [sqs_record()]})

    assert result == {"batchItemFailures": []}
    fake_db.mark_failed.assert_not_called()


def test_worker_failure_before_last_attempt_retries_without_marking_failed(fake_db):
    fake_db.mark_processing.return_value = True

    with patch.object(handler, "transcribe") as mock_transcribe:
        mock_transcribe.start_transcription_job.side_effect = Exception("throttled")
        result = batch_worker.handle({"Records": [sqs_record(receive_count=2)]})

    assert result == {"batchItemFailures": [{"itemIdentifier": "m1"}]}
    fake_db.mark_failed.assert_not_called()


def test_worker_failure_on_last_attempt_marks_failed_and_still_reports_failure(fake_db):
    fake_db.mark_processing.return_value = True

    with patch.object(handler, "transcribe") as mock_transcribe:
        mock_transcribe.start_transcription_job.side_effect = Exception("throttled")
        result = batch_worker.handle({"Records": [sqs_record(receive_count=3)]})

    # Reported as a failure so SQS moves it to the DLQ.
    assert result == {"batchItemFailures": [{"itemIdentifier": "m1"}]}
    fake_db.mark_failed.assert_called_once()
    assert fake_db.mark_failed.call_args.args[1] == RECORD_ID


def test_worker_unsupported_format_fails_immediately_without_retry(fake_db):
    key = f"{RECORD_ID}/notes.txt"
    fake_db.mark_processing.return_value = True

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = batch_worker.handle({"Records": [sqs_record(key=key)]})

    assert result == {"batchItemFailures": []}
    mock_transcribe.start_transcription_job.assert_not_called()
    fake_db.mark_failed.assert_called_once()


def test_worker_one_bad_record_does_not_fail_its_siblings(fake_db):
    fake_db.mark_processing.return_value = True
    good = sqs_record(message_id="good")
    bad = sqs_record(message_id="bad")

    with patch.object(handler, "transcribe") as mock_transcribe:
        mock_transcribe.start_transcription_job.side_effect = [None, Exception("boom")]
        result = batch_worker.handle({"Records": [good, bad]})

    assert result == {"batchItemFailures": [{"itemIdentifier": "bad"}]}


def test_worker_drops_messages_not_from_batch_source(fake_db):
    with patch.object(handler, "transcribe") as mock_transcribe:
        result = batch_worker.handle({"Records": [sqs_record(source="other")]})

    assert result == {"batchItemFailures": []}
    fake_db.mark_processing.assert_not_called()
    mock_transcribe.start_transcription_job.assert_not_called()


# -- enqueuer -------------------------------------------------------------------

@pytest.fixture
def aws(monkeypatch):
    s3, sqs = MagicMock(name="s3"), MagicMock(name="sqs")
    monkeypatch.setattr(enqueuer.boto3, "client", lambda name: {"s3": s3, "sqs": sqs}[name])
    return s3, sqs


def pages(*page_keys):
    return [{"Contents": [{"Key": k} for k in keys]} for keys in page_keys]


def test_enqueuer_paginates_and_enqueues_each_pending_object(fake_db, aws):
    s3, sqs = aws
    other = str(uuid.uuid4())
    s3.get_paginator.return_value.paginate.return_value = pages([KEY], [f"{other}/b.wav"])
    fake_db.claim_pending.return_value = True

    stats = enqueuer.handler({}, None)

    assert stats["enqueued"] == 2
    assert sqs.send_message.call_count == 2
    first = sqs.send_message.call_args_list[0].kwargs
    assert first["QueueUrl"] == "https://sqs.example/batch"
    assert json.loads(first["MessageBody"]) == {
        "bucket": "batch-bucket", "key": KEY, "record_id": RECORD_ID, "source": "batch",
    }
    s3.get_paginator.assert_called_once_with("list_objects_v2")


def test_enqueuer_skips_rows_that_are_not_pending(fake_db, aws):
    s3, sqs = aws
    s3.get_paginator.return_value.paginate.return_value = pages([KEY])
    fake_db.claim_pending.return_value = False

    stats = enqueuer.handler({}, None)

    assert stats["enqueued"] == 0 and stats["skipped"] == 1
    sqs.send_message.assert_not_called()


def test_enqueuer_ignores_keys_that_are_not_record_id_slash_filename(fake_db, aws):
    s3, sqs = aws
    s3.get_paginator.return_value.paginate.return_value = pages(["stray.mp3", "not-a-uuid/a.mp3", f"{RECORD_ID}/"])

    stats = enqueuer.handler({}, None)

    assert stats["skipped"] == 3
    fake_db.claim_pending.assert_not_called()
    sqs.send_message.assert_not_called()


def test_enqueuer_releases_claim_and_raises_when_send_fails(fake_db, aws):
    s3, sqs = aws
    s3.get_paginator.return_value.paginate.return_value = pages([KEY])
    fake_db.claim_pending.return_value = True
    sqs.send_message.side_effect = Exception("sqs down")

    with pytest.raises(RuntimeError, match="Failed to enqueue 1"):
        enqueuer.handler({}, None)

    fake_db.release_queued.assert_called_once_with(fake_db.connect.return_value, RECORD_ID)


def test_enqueuer_dry_run_changes_nothing(fake_db, aws):
    s3, sqs = aws
    s3.get_paginator.return_value.paginate.return_value = pages([KEY])
    fake_db.get_status.return_value = "PENDING"

    stats = enqueuer.handler({"dry_run": True}, None)

    assert stats["enqueued"] == 1 and stats["dry_run"] is True
    fake_db.claim_pending.assert_not_called()
    sqs.send_message.assert_not_called()
