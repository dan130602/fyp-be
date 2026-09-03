from unittest.mock import patch

import pytest

from app import handler


def s3_event(records):
    return {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {"key": key},
                }
            }
            for bucket, key in records
        ]
    }


@pytest.fixture(autouse=True)
def transcribe_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_BUCKET", "output-bucket")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_CODE", "en-US")


@pytest.fixture
def fixed_uuid(monkeypatch):
    class _FixedUUID:
        hex = "deadbeef00000000"

    monkeypatch.setattr(handler.uuid, "uuid4", lambda: _FixedUUID())
    return _FixedUUID.hex[:8]


# -- _media_format_for_key ---------------------------------------------------

@pytest.mark.parametrize(
    "key,expected",
    [
        ("uploads/a.mp3", "mp3"),
        ("uploads/a.MP3", "mp3"),
        ("uploads/a.wav", "wav"),
        ("uploads/nested/dir/a.flac", "flac"),
        ("uploads/a.txt", None),
        ("uploads/no-extension", None),
    ],
)
def test_media_format_for_key(key, expected):
    assert handler._media_format_for_key(key) == expected


# -- _job_name_for_key --------------------------------------------------------

def test_job_name_for_key_sanitizes_and_suffixes(fixed_uuid):
    job_name = handler._job_name_for_key("uploads/My Recording (final)!.mp3")

    assert job_name.startswith("My-Recording--final--")
    assert job_name.endswith(fixed_uuid)


def test_job_name_for_key_truncates_long_names(fixed_uuid):
    long_stem = "a" * 300
    job_name = handler._job_name_for_key(f"uploads/{long_stem}.mp3")

    prefix, _, suffix = job_name.rpartition("-")
    assert len(prefix) == 150
    assert suffix == fixed_uuid


# -- _start_job ----------------------------------------------------------------

def test_start_job_calls_transcribe_with_expected_params(fixed_uuid):
    with patch.object(handler, "transcribe") as mock_transcribe:
        job_name = handler._start_job("input-bucket", "uploads/audio.mp3")

    assert job_name == f"audio-{fixed_uuid}"
    mock_transcribe.start_transcription_job.assert_called_once_with(
        TranscriptionJobName=job_name,
        LanguageCode="en-US",
        MediaFormat="mp3",
        Media={"MediaFileUri": "s3://input-bucket/uploads/audio.mp3"},
        OutputBucketName="output-bucket",
        OutputKey=f"transcripts/{job_name}.json",
    )


def test_start_job_skips_unsupported_format():
    with patch.object(handler, "transcribe") as mock_transcribe:
        job_name = handler._start_job("input-bucket", "uploads/notes.txt")

    assert job_name is None
    mock_transcribe.start_transcription_job.assert_not_called()


# -- handler ---------------------------------------------------------------

def test_handler_starts_job_for_valid_record(fixed_uuid):
    event = s3_event([("input-bucket", "uploads/audio.mp3")])

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = handler.handler(event, context=None)

    assert result == {"started_jobs": [f"audio-{fixed_uuid}"]}
    mock_transcribe.start_transcription_job.assert_called_once()


def test_handler_url_decodes_key(fixed_uuid):
    event = s3_event([("input-bucket", "uploads/my+recording.mp3")])

    with patch.object(handler, "transcribe") as mock_transcribe:
        handler.handler(event, context=None)

    call_kwargs = mock_transcribe.start_transcription_job.call_args.kwargs
    assert call_kwargs["Media"]["MediaFileUri"] == "s3://input-bucket/uploads/my recording.mp3"


def test_handler_skips_directory_markers_and_unsupported_formats(fixed_uuid):
    event = s3_event(
        [
            ("input-bucket", "uploads/"),
            ("input-bucket", "uploads/readme.txt"),
            ("input-bucket", "uploads/audio.wav"),
        ]
    )

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = handler.handler(event, context=None)

    assert result == {"started_jobs": [f"audio-{fixed_uuid}"]}
    mock_transcribe.start_transcription_job.assert_called_once()


def test_handler_ignores_records_missing_bucket_or_key():
    event = {"Records": [{"s3": {"bucket": {}, "object": {"key": "uploads/a.mp3"}}}]}

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = handler.handler(event, context=None)

    assert result == {"started_jobs": []}
    mock_transcribe.start_transcription_job.assert_not_called()


def test_handler_raises_after_processing_all_records_when_one_fails(fixed_uuid):
    event = s3_event(
        [
            ("input-bucket", "uploads/good.mp3"),
            ("input-bucket", "uploads/bad.wav"),
        ]
    )

    with patch.object(handler, "transcribe") as mock_transcribe:
        mock_transcribe.start_transcription_job.side_effect = [None, Exception("boom")]

        with pytest.raises(RuntimeError, match="Failed to start 1 Transcribe job"):
            handler.handler(event, context=None)

    assert mock_transcribe.start_transcription_job.call_count == 2
