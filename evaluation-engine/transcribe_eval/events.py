"""The S3 upload event the handler receives, and the object-key convention the evaluation uses.

The handler derives the Transcribe job name from the key (`uploads/<userId>/...`), and a job name can only be used once. Each run therefore
uses its own `<userId>` of the form `eval-<run id>`, so reruns never collide with earlier jobs. The master audio stays at a fixed key per
dataset and render and is copied inside S3 to the run's key. The dataset id and render id are part of the keys so datasets and renders never mix.
The local files are all called `audio.mp3` (one per call folder), so the call id is put in the S3 file name instead.
"""

import re
import urllib.parse

MASTER_PREFIX = "eval-audio"
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
DATASET_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
RENDER_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
CALL_ID = re.compile(r"CALL_\d{3}")


def _check(pattern, value, what):
    if not pattern.fullmatch(value):
        raise ValueError(f"invalid {what}: {value!r}")


def master_key(dataset_id, render_id, call_id, extension="mp3"):
    """`eval-audio/<dataset id>/<render id>/CALL_xxx.mp3`: the kept master copy of one call's audio for one render."""
    _check(DATASET_ID, dataset_id, "dataset id"), _check(RENDER_ID, render_id, "render id"), _check(CALL_ID, call_id, "call id")
    return f"{MASTER_PREFIX}/{dataset_id}/{render_id}/{call_id}.{extension}"


def run_key(run_id, dataset_id, call_id, extension="mp3"):
    """`uploads/eval-<run id>/<dataset id>/CALL_xxx.mp3`: the key handed to the handler for this run."""
    _check(RUN_ID, run_id, "run id"), _check(DATASET_ID, dataset_id, "dataset id"), _check(CALL_ID, call_id, "call id")
    return f"uploads/eval-{run_id}/{dataset_id}/{call_id}.{extension}"


def s3_event(bucket, keys):
    """A minimal S3 'object created' event. Keys are URL-encoded the way S3 delivers them (the handler decodes them)."""
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": urllib.parse.quote_plus(key, safe="/")}}} for key in keys]}
