"""Run the approved 16-call baseline and vocabulary trial from evaluation-engine."""

import json
import os
from pathlib import Path
import sys
import time

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path.cwd()))

from transcribe_eval import collect
from transcribe_eval.config import ENGINE_ROOT, FYP_BE_ROOT, HANDLER_PATH, Settings
from transcribe_eval.export import _language
from transcribe_eval.handler_loader import load_handler
from transcribe_eval.reference import load_terms
from transcribe_eval.score import score_run


REGION = "ap-southeast-1"
BUCKET = "fyp-transcribe-eval-team"
DATASET = FYP_BE_ROOT / "evaluation-dataset" / "datasets" / "advisory_execution_multilingual_v1_development"
CALL_IDS = ('CALL_071', 'CALL_090', 'CALL_091', 'CALL_139', 'CALL_003', 'CALL_123', 'CALL_177', 'CALL_244', 'CALL_001', 'CALL_035', 'CALL_041', 'CALL_209', 'CALL_146', 'CALL_184', 'CALL_198', 'CALL_235')
LOCALES = ("en-US", "zh-CN")
BASE_SETTINGS = {
    "TRANSCRIBE_LANGUAGE_OPTIONS": ",".join(LOCALES),
    "TRANSCRIBE_LANGUAGE_ID_MODE": "multiple",
    "TRANSCRIBE_MAX_SPEAKER_LABELS": "2",
}


def register_vocabulary(s3, transcribe, owner, tag):
    names = {}
    for locale in LOCALES:
        name = f"clearcall-v2-{locale.lower()}-{tag}"
        table = FYP_BE_ROOT / "stt-experiments" / "vocabulary-v2" / f"vocab-{locale}.txt"
        key = f"eval-vocabulary/{tag}/{table.name}"
        s3.put_object(
            Bucket=BUCKET, ExpectedBucketOwner=owner, Key=key,
            Body=table.read_bytes(), ContentType="text/plain; charset=utf-8", IfNoneMatch="*",
        )
        transcribe.create_vocabulary(
            VocabularyName=name, LanguageCode=locale, VocabularyFileUri=f"s3://{BUCKET}/{key}",
        )
        names[locale] = name
        print(f"Requested vocabulary {name}", flush=True)
    pending = set(names.values())
    deadline = time.monotonic() + 240
    while pending and time.monotonic() < deadline:
        for name in sorted(pending):
            result = transcribe.get_vocabulary(VocabularyName=name)
            if result["VocabularyState"] == "FAILED":
                raise RuntimeError(f"Vocabulary {name} failed: {result.get('FailureReason')}")
            if result["VocabularyState"] == "READY":
                pending.remove(name)
                print(f"Vocabulary ready: {name}", flush=True)
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"Vocabulary not ready: {', '.join(sorted(pending))}")
    return names


def run(s3, tag, label, names):
    for name in tuple(os.environ):
        if name.startswith("TRANSCRIBE_"):
            del os.environ[name]
    flags = dict(BASE_SETTINGS)
    if names:
        flags["TRANSCRIBE_VOCABULARIES"] = ",".join(f"{locale}={name}" for locale, name in names.items())
    settings = Settings(REGION, BUCKET, HANDLER_PATH, DATASET, {}, flags)
    settings.apply_to_environ()
    handler = load_handler(settings.handler_path)
    run_id = f"stt-vocab-broader-{label}-{tag}"
    print(f"Starting {run_id} with {flags}", flush=True)
    result = collect.collect(
        settings, handler, s3, dataset_root=DATASET, render_id="auto", call_ids=CALL_IDS,
        run_id=run_id, poll_seconds=7, timeout_seconds=900, progress=lambda message: print(message, flush=True),
    )
    if result["status"] != "completed":
        for entry in result["calls"]:
            print(entry["call_id"], entry["status"], entry["failure_reason"], flush=True)
        raise RuntimeError(f"{run_id} did not complete all 16 calls")

    settings_verified = True
    for entry in result["calls"]:
        job = entry["job"]
        applied = {locale: row.get("VocabularyName") for locale, row in (job.get("LanguageIdSettings") or {}).items()}
        speaker = job.get("Settings") or {}
        matches = (
            applied == names and job.get("IdentifyMultipleLanguages") is True
            and set(job.get("LanguageOptions") or ()) == set(LOCALES)
            and speaker.get("ShowSpeakerLabels") is True and speaker.get("MaxSpeakerLabels") == 2
        )
        settings_verified = settings_verified and matches
        print(entry["call_id"], "AWS job settings matched:", matches, flush=True)

    reviewed, terms = load_terms(ENGINE_ROOT / "domain_terms.json")
    folder = collect.RUNS_ROOT / run_id
    summary = score_run(folder, terms=terms, terms_reviewed=reviewed)
    rows = [json.loads(row) for row in (folder / "scores" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]

    def metrics(row, wer, speaker):
        w, e, s = row[wer], row["entity_wer"], row[speaker]
        return {
            "wer": [w["errors"], w["reference_words"]],
            "entity_wer": [e["errors"], e["reference_words"]] if e and e.get("available", True) else None,
            "speaker": [s["correct_items"], s["eligible_items"]],
        }

    print(json.dumps({
        "run": run_id,
        "normalizer": summary["normalizer_version"],
        "settings_verified": settings_verified,
        "overall": metrics(summary, "normalised_wer", "speaker_attribution"),
        "by_call": {row["call_id"]: metrics(row, "wer", "speaker") for row in rows},
    }), flush=True)
    if not settings_verified:
        raise RuntimeError(f"Inspect the recorded AWS settings for {run_id} before comparing")
    return {entry["call_id"]: entry["audio_sha256"] for entry in result["calls"]}


def main():
    if Path.cwd() != ENGINE_ROOT:
        raise RuntimeError("Run from the evaluation-engine directory")
    plan = collect.plan(DATASET, "auto", CALL_IDS, BUCKET)
    if len(plan["calls"]) != 16 or plan["audio_minutes"] > 200:
        raise RuntimeError("Unexpected canary audio plan")
    if os.environ.get("AWS_PROFILE") != "fyp-stt-eval":
        raise RuntimeError("Set the authenticated evaluation profile")
    session = boto3.Session(profile_name="fyp-stt-eval", region_name=REGION)
    owner = session.client("sts").get_caller_identity()["Account"]
    s3 = session.client("s3")
    bucket = s3.head_bucket(Bucket=BUCKET, ExpectedBucketOwner=owner)
    if bucket["ResponseMetadata"]["HTTPHeaders"].get("x-amz-bucket-region") != REGION:
        raise RuntimeError("Evaluation bucket is outside the expected region")
    notifications = s3.get_bucket_notification_configuration(Bucket=BUCKET, ExpectedBucketOwner=owner)
    if any(key != "ResponseMetadata" for key in notifications):
        raise RuntimeError("Evaluation bucket has notifications")
    try:
        status = s3.get_bucket_policy_status(Bucket=BUCKET, ExpectedBucketOwner=owner)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchBucketPolicy":
            raise
    else:
        if status["PolicyStatus"]["IsPublic"]:
            raise RuntimeError("Evaluation bucket has a public policy")

    tag = collect.new_run_id().lower()
    print(f"Trial tag {tag}; 16 calls, {plan['audio_minutes']} audio minutes per run", flush=True)
    names = register_vocabulary(s3, session.client("transcribe"), owner, tag)
    baseline = run(s3, tag, "baseline", {})
    candidate = run(s3, tag, "candidate", names)
    if baseline != candidate:
        raise RuntimeError("The runs used different audio hashes")
    print("Matched audio hashes across all 16 call pairs.", flush=True)


if __name__ == "__main__":
    main()
