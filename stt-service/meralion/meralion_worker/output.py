"""The output document `transcripts/<job_name>.json`: Transcribe-shaped `results.transcripts` only, plus a `meralion` block that Java never forwards (it forwards `results`).

v0 emits NO items, speaker labels, audio segments, timestamps or confidence: the model provides none and none is invented. Raw and final transcripts are both recorded.
"""
import json
import re

from meralion_worker.config import fingerprint_of
from meralion_worker.formatters import REGISTRY, apply_formatter, formatter_version

SCHEMA_VERSION = "meralion-worker-output-v0"
JOB_NAME = re.compile(r"[0-9A-Za-z._-]{1,200}")          # always used with fullmatch(): "$" would also accept a trailing newline
# The v0 shape is closed: `results` is exactly {"transcripts": [{"transcript": <str>}]}. Any other key (items, speaker_labels, audio_segments, channel_labels, start_time,
# end_time, confidence, alternatives ...) therefore cannot appear anywhere inside it, at any depth, and validate_document rejects the document if it does.


class OutputError(ValueError):
    pass


def build_document(*, job_name, transcript_raw, transcript_final, effective, invocation, decoder, model, worker, audio, retries, trace, timing):
    """effective: config.EffectiveConfig.as_record() output ({"config", "config_fingerprint"}).

    The formatter identity (name, version) is derived from effective["config"]["formatter"], never passed separately, so it cannot contradict the recorded configuration.
    """
    formatter_name = effective["config"]["formatter"]
    document = {
        "jobName": job_name,
        "status": "COMPLETED",
        "results": {"transcripts": [{"transcript": transcript_final}]},
        "meralion": {
            "schema_version": SCHEMA_VERSION,
            "transcript_raw": transcript_raw,
            "transcript_final": transcript_final,
            "formatter": {"name": formatter_name, "version": formatter_version(formatter_name) if formatter_name in REGISTRY else None, "changed": transcript_raw != transcript_final},
            "config": effective["config"],
            "config_fingerprint": effective["config_fingerprint"],
            "invocation": invocation,
            "decoder": decoder,
            "model": model,
            "worker": worker,
            "audio": audio,
            "retries": retries,
            "trace": trace,
            "timing": timing,
        },
    }
    validate_document(document)
    return document


def validate_document(document):
    if not isinstance(document, dict):
        raise OutputError("the document must be an object")
    name = document.get("jobName")
    if not isinstance(name, str) or not JOB_NAME.fullmatch(name):
        raise OutputError(f"jobName must fully match {JOB_NAME.pattern}")
    if document.get("status") != "COMPLETED":
        raise OutputError("status must be COMPLETED")
    results = document.get("results")
    if not isinstance(results, dict) or set(results) != {"transcripts"}:
        raise OutputError("results must contain only 'transcripts'")
    transcripts = results["transcripts"]
    if not (isinstance(transcripts, list) and len(transcripts) == 1 and isinstance(transcripts[0], dict) and set(transcripts[0]) == {"transcript"} and isinstance(transcripts[0]["transcript"], str)):
        raise OutputError("results.transcripts must be [{'transcript': <str>}]")
    block = document.get("meralion")
    if not isinstance(block, dict) or block.get("schema_version") != SCHEMA_VERSION:
        raise OutputError("missing or unknown meralion.schema_version")
    for key in ("transcript_raw", "transcript_final"):
        if not isinstance(block.get(key), str):
            raise OutputError(f"meralion.{key} must be a string")
    if block["transcript_final"] != transcripts[0]["transcript"]:
        raise OutputError("results.transcripts[0].transcript must equal meralion.transcript_final")
    config = block.get("config")
    configured = config.get("formatter") if isinstance(config, dict) else None
    if configured not in REGISTRY:
        raise OutputError(f"unknown formatter {configured!r} in meralion.config")
    if block.get("config_fingerprint") != fingerprint_of(config):
        raise OutputError("meralion.config_fingerprint does not match meralion.config")
    declared = block.get("formatter")
    if not isinstance(declared, dict) or declared.get("name") != configured:
        raise OutputError("meralion.formatter.name must equal the formatter in meralion.config")
    if declared.get("version") != formatter_version(configured):
        raise OutputError(f"meralion.formatter.version must be {formatter_version(configured)!r} for formatter {configured!r}")
    if block["transcript_final"] != apply_formatter(configured, block["transcript_raw"]):
        raise OutputError(f"transcript_final must equal what formatter {configured!r} makes of transcript_raw (formatter 'none' leaves the text unchanged)")
    if declared.get("changed") != (block["transcript_raw"] != block["transcript_final"]):
        raise OutputError("meralion.formatter.changed disagrees with the transcripts")
    return document


def serialize(document):
    """UTF-8 bytes, no ASCII escaping (Chinese stays readable)."""
    validate_document(document)
    return (json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
