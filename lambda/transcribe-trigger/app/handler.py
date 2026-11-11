import hashlib
import logging
import os
import re
import urllib.parse

import boto3

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

transcribe = boto3.client("transcribe")

SUPPORTED_MEDIA_FORMATS = {
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "ogg": "ogg",
    "amr": "amr",
    "webm": "webm",
}


def _media_format_for_key(key):
    _, _, ext = key.rpartition(".")
    return SUPPORTED_MEDIA_FORMATS.get(ext.lower())


def _job_name_for_key(key):
    """Deterministic - must match PresignedUrlService.jobNameForObjectKey (Java) exactly for
    the same key, since that's the only correlation between a MediaAsset row and the Transcribe
    job started here, with neither side ever calling the other directly.

    Includes the user id (key's "uploads/{userId}/..." segment) so job names are recognizable
    at a glance in the Transcribe console, rather than a bare hash."""
    user_id = key.split("/")[1]
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"transcribe-{user_id}-{digest}"


def _batch_job_name_for_key(key):
    """Batch keys are "{record_id}/{filename}" (no "uploads/{userId}" prefix), so the id segment
    is index 0 and is the MediaAsset id. Must match PresignedUrlService.jobNameForObjectKey (Java)
    for ProcessingMode.BATCH."""
    record_id = key.split("/")[0]
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"transcribe-{record_id}-{digest}"


def _language_parameters():
    """How Transcribe should pick the language.

    Default (nothing but TRANSCRIBE_LANGUAGE_CODE set): one fixed language, exactly as before.
    With TRANSCRIBE_LANGUAGE_OPTIONS (comma separated, at least two codes, e.g. "en-US,zh-CN") the language is identified from the audio:
    TRANSCRIBE_LANGUAGE_ID_MODE "multiple" (default) lets one file switch language between segments (IdentifyMultipleLanguages);
    "single" picks one language for the whole file (IdentifyLanguage).
    """
    options = [code.strip() for code in os.environ.get("TRANSCRIBE_LANGUAGE_OPTIONS", "").split(",") if code.strip()]
    if not options:
        return {"LanguageCode": os.environ.get("TRANSCRIBE_LANGUAGE_CODE", "en-US")}
    if len(options) < 2:
        raise ValueError("TRANSCRIBE_LANGUAGE_OPTIONS needs at least two language codes")
    mode = os.environ.get("TRANSCRIBE_LANGUAGE_ID_MODE", "multiple").strip().lower()
    if mode == "multiple":
        return {"IdentifyMultipleLanguages": True, "LanguageOptions": options}
    if mode == "single":
        return {"IdentifyLanguage": True, "LanguageOptions": options}
    raise ValueError("TRANSCRIBE_LANGUAGE_ID_MODE must be 'multiple' or 'single'")


def _customization_parameters(language, settings):
    """Attach configured resources using the request shape for the selected language mode."""
    parameters = {}
    model = os.environ.get("TRANSCRIBE_LANGUAGE_MODEL_NAME", "").strip()
    if model:
        if language.get("LanguageCode") != "en-US":
            raise ValueError("TRANSCRIBE_LANGUAGE_MODEL_NAME requires fixed en-US with no language options")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", model):
            raise ValueError("TRANSCRIBE_LANGUAGE_MODEL_NAME is not a valid Transcribe resource name")
        parameters["ModelSettings"] = {"LanguageModelName": model}

    configured = os.environ.get("TRANSCRIBE_VOCABULARIES", "").strip()
    if not configured:
        return parameters

    vocabularies = {}
    for entry in configured.split(","):
        code, _, name = (part.strip() for part in entry.partition("="))
        if not code or code in vocabularies or not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", name):
            raise ValueError("TRANSCRIBE_VOCABULARIES needs unique locale=name entries")
        vocabularies[code] = name

    if "LanguageCode" in language:
        code = language["LanguageCode"]
        if set(vocabularies) != {code}:
            raise ValueError("TRANSCRIBE_VOCABULARIES must match the fixed language code")
        settings["VocabularyName"] = vocabularies[code]
    else:
        options = language["LanguageOptions"]
        if not 2 <= len(options) <= 5 or len(set(options)) != len(options):
            raise ValueError("Custom vocabularies with language identification need two to five distinct LanguageOptions")
        if not 2 <= len(vocabularies) <= 5 or not vocabularies.keys() <= set(options):
            raise ValueError("TRANSCRIBE_VOCABULARIES needs two to five entries matching LanguageOptions")
        parameters["LanguageIdSettings"] = {
            code: {"VocabularyName": name} for code, name in vocabularies.items()
        }
    return parameters


def _start_job(bucket, key, job_name=None):
    media_format = _media_format_for_key(key)
    if media_format is None:
        logger.warning("Skipping s3://%s/%s: unsupported or unknown media format", bucket, key)
        return None

    job_name = job_name or _job_name_for_key(key)
    output_bucket = os.environ["TRANSCRIBE_OUTPUT_BUCKET"]
    max_speaker_labels = int(os.environ.get("TRANSCRIBE_MAX_SPEAKER_LABELS", "2"))
    language = _language_parameters()
    # Diarization adds results.speaker_labels.segments to the output JSON.
    settings = {"ShowSpeakerLabels": True, "MaxSpeakerLabels": max_speaker_labels}
    customization = _customization_parameters(language, settings)

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        **language,
        **customization,
        MediaFormat=media_format,
        Media={"MediaFileUri": f"s3://{bucket}/{key}"},
        OutputBucketName=output_bucket,
        OutputKey=f"transcripts/{job_name}.json",
        Settings=settings,
    )
    logger.info("Started Transcribe job %s for s3://%s/%s", job_name, bucket, key)
    return job_name


def handler(event, context):
    records = event.get("Records", [])
    if records and records[0].get("eventSource") == "aws:sqs":
        # Batch queue consumer. Imported lazily so the immediate (S3-triggered) path never loads
        # the DB driver or needs any DB configuration.
        from app import batch_worker

        return batch_worker.handle(event)

    started = []
    errors = []

    for record in event.get("Records", []):
        s3_info = record.get("s3", {})
        bucket = s3_info.get("bucket", {}).get("name")
        raw_key = s3_info.get("object", {}).get("key")
        if not bucket or not raw_key:
            continue

        key = urllib.parse.unquote_plus(raw_key)
        if key.endswith("/"):
            continue

        try:
            job_name = _start_job(bucket, key)
            if job_name:
                started.append(job_name)
        except Exception as exc:
            logger.exception("Failed to start Transcribe job for s3://%s/%s", bucket, key)
            errors.append({"bucket": bucket, "key": key, "error": str(exc)})

    if errors:
        raise RuntimeError(f"Failed to start {len(errors)} Transcribe job(s): {errors}")

    return {"started_jobs": started}
