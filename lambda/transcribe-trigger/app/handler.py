import hashlib
import logging
import os
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
    job started here, with neither side ever calling the other directly."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"transcribe-{digest}"


def _start_job(bucket, key):
    media_format = _media_format_for_key(key)
    if media_format is None:
        logger.warning("Skipping s3://%s/%s: unsupported or unknown media format", bucket, key)
        return None

    job_name = _job_name_for_key(key)
    output_bucket = os.environ["TRANSCRIBE_OUTPUT_BUCKET"]
    language_code = os.environ.get("TRANSCRIBE_LANGUAGE_CODE", "en-US")

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode=language_code,
        MediaFormat=media_format,
        Media={"MediaFileUri": f"s3://{bucket}/{key}"},
        OutputBucketName=output_bucket,
        OutputKey=f"transcripts/{job_name}.json",
    )
    logger.info("Started Transcribe job %s for s3://%s/%s", job_name, bucket, key)
    return job_name


def handler(event, context):
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
