import hashlib
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


def expected_job_name(key):
    user_id = key.split("/")[1]
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"transcribe-{user_id}-{digest}"


@pytest.fixture(autouse=True)
def transcribe_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_BUCKET", "output-bucket")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_CODE", "en-US")


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

def test_job_name_for_key_is_deterministic():
    key = "uploads/My Recording (final)!.mp3"

    assert handler._job_name_for_key(key) == expected_job_name(key)


def test_job_name_for_key_digest_suffix_has_constant_length():
    long_stem = "a" * 300
    job_name = handler._job_name_for_key(f"uploads/{long_stem}.mp3")

    digest_suffix = job_name.rpartition("-")[2]
    assert len(digest_suffix) == 16


# -- _start_job ----------------------------------------------------------------

def test_start_job_calls_transcribe_with_expected_params():
    key = "uploads/audio.mp3"
    job_name = expected_job_name(key)

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = handler._start_job("input-bucket", key)

    assert result == job_name
    mock_transcribe.start_transcription_job.assert_called_once_with(
        TranscriptionJobName=job_name,
        LanguageCode="en-US",
        MediaFormat="mp3",
        Media={"MediaFileUri": "s3://input-bucket/uploads/audio.mp3"},
        OutputBucketName="output-bucket",
        OutputKey=f"transcripts/{job_name}.json",
        Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": 2},
    )


def test_start_job_skips_unsupported_format():
    with patch.object(handler, "transcribe") as mock_transcribe:
        job_name = handler._start_job("input-bucket", "uploads/notes.txt")

    assert job_name is None
    mock_transcribe.start_transcription_job.assert_not_called()


# -- handler ---------------------------------------------------------------

def test_handler_starts_job_for_valid_record():
    key = "uploads/audio.mp3"
    event = s3_event([("input-bucket", key)])

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = handler.handler(event, context=None)

    assert result == {"started_jobs": [expected_job_name(key)]}
    mock_transcribe.start_transcription_job.assert_called_once()


def test_handler_url_decodes_key():
    event = s3_event([("input-bucket", "uploads/my+recording.mp3")])

    with patch.object(handler, "transcribe") as mock_transcribe:
        handler.handler(event, context=None)

    call_kwargs = mock_transcribe.start_transcription_job.call_args.kwargs
    assert call_kwargs["Media"]["MediaFileUri"] == "s3://input-bucket/uploads/my recording.mp3"


def test_handler_skips_directory_markers_and_unsupported_formats():
    key = "uploads/audio.wav"
    event = s3_event(
        [
            ("input-bucket", "uploads/"),
            ("input-bucket", "uploads/readme.txt"),
            ("input-bucket", key),
        ]
    )

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = handler.handler(event, context=None)

    assert result == {"started_jobs": [expected_job_name(key)]}
    mock_transcribe.start_transcription_job.assert_called_once()


def test_handler_ignores_records_missing_bucket_or_key():
    event = {"Records": [{"s3": {"bucket": {}, "object": {"key": "uploads/a.mp3"}}}]}

    with patch.object(handler, "transcribe") as mock_transcribe:
        result = handler.handler(event, context=None)

    assert result == {"started_jobs": []}
    mock_transcribe.start_transcription_job.assert_not_called()


def test_handler_raises_after_processing_all_records_when_one_fails():
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


# -- language selection ------------------------------------------------------

def _start_kwargs(monkeypatch, **env):
    for name in ("TRANSCRIBE_LANGUAGE_OPTIONS", "TRANSCRIBE_LANGUAGE_ID_MODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with patch.object(handler, "transcribe") as mock_transcribe:
        handler.handler(s3_event([("input-bucket", "uploads/u1/audio.wav")]), context=None)
    return mock_transcribe.start_transcription_job.call_args.kwargs


def test_default_language_is_the_single_configured_code(monkeypatch):
    kwargs = _start_kwargs(monkeypatch)
    assert kwargs["LanguageCode"] == "en-US" and "LanguageOptions" not in kwargs and "IdentifyMultipleLanguages" not in kwargs


def test_language_options_identify_multiple_languages_by_default(monkeypatch):
    kwargs = _start_kwargs(monkeypatch, TRANSCRIBE_LANGUAGE_OPTIONS="en-US, zh-CN")
    assert kwargs["IdentifyMultipleLanguages"] is True and kwargs["LanguageOptions"] == ["en-US", "zh-CN"] and "LanguageCode" not in kwargs
    assert kwargs["Settings"]["ShowSpeakerLabels"] is True


def test_single_language_identification_mode(monkeypatch):
    kwargs = _start_kwargs(monkeypatch, TRANSCRIBE_LANGUAGE_OPTIONS="en-US,zh-CN", TRANSCRIBE_LANGUAGE_ID_MODE="single")
    assert kwargs["IdentifyLanguage"] is True and "IdentifyMultipleLanguages" not in kwargs


def test_one_language_option_or_unknown_mode_is_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="at least two"):
        _start_kwargs(monkeypatch, TRANSCRIBE_LANGUAGE_OPTIONS="en-US")
    with pytest.raises(RuntimeError, match="multiple' or 'single"):
        _start_kwargs(monkeypatch, TRANSCRIBE_LANGUAGE_OPTIONS="en-US,zh-CN", TRANSCRIBE_LANGUAGE_ID_MODE="bogus")
