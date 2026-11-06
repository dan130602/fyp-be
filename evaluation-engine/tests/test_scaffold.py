"""Offline checks of the scaffold: settings, the S3 event and key convention, and the REAL production handler driven through them."""
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transcribe_eval import config
from transcribe_eval.config import ConfigError, load_settings
from transcribe_eval.events import master_key, run_key, s3_event
from transcribe_eval.handler_loader import load_handler


# ---- settings -------------------------------------------------------------------------------------------------------------------------------

def test_the_environment_wins_over_the_env_file_and_defaults_apply(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EVAL_BUCKET=from-file-bucket\nAWS_REGION=eu-west-1\nTRANSCRIBE_MAX_SPEAKER_LABELS=3\n", encoding="utf-8")
    settings = load_settings({"EVAL_BUCKET": "from-env-bucket"}, env_file)
    assert settings.bucket == "from-env-bucket"
    assert settings.region == "eu-west-1"  # not in the environment, so the file's value
    assert load_settings({"EVAL_BUCKET": "my-bucket"}, tmp_path / "missing.env").region == config.DEFAULT_REGION == "ap-southeast-1"


def test_the_handler_only_gets_the_optional_settings_that_are_set(tmp_path):
    bare = load_settings({"EVAL_BUCKET": "my-bucket"}, tmp_path / "none")
    assert bare.handler_environment() == {"TRANSCRIBE_OUTPUT_BUCKET": "my-bucket"}  # the handler's own defaults then apply
    mirrored = load_settings({"EVAL_BUCKET": "my-bucket", "TRANSCRIBE_LANGUAGE_CODE": "en-US", "TRANSCRIBE_MAX_SPEAKER_LABELS": " "}, tmp_path / "none")
    assert mirrored.handler_environment() == {"TRANSCRIBE_OUTPUT_BUCKET": "my-bucket", "TRANSCRIBE_LANGUAGE_CODE": "en-US"}


@pytest.mark.parametrize("environ", [{}, {"EVAL_BUCKET": "Bad_Bucket"}, {"EVAL_BUCKET": "x"}])
def test_a_missing_or_invalid_bucket_is_refused(environ, tmp_path):
    with pytest.raises(ConfigError):
        load_settings(environ, tmp_path / "none")


def test_credentials_are_carried_but_never_echoed_in_errors(tmp_path):
    settings = load_settings({"EVAL_BUCKET": "my-bucket", "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE", "AWS_SECRET_ACCESS_KEY": "s3cr3t"}, tmp_path / "none")
    environ = {}
    settings.apply_to_environ(environ)
    assert environ["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE" and environ["AWS_DEFAULT_REGION"] == "ap-southeast-1"
    with pytest.raises(ConfigError) as error:
        load_settings({"EVAL_BUCKET": "Bad_Bucket", "AWS_SECRET_ACCESS_KEY": "s3cr3t"}, tmp_path / "none")
    assert "s3cr3t" not in str(error.value)


# ---- keys and events ------------------------------------------------------------------------------------------------------------------------

def test_keys_follow_the_handlers_uploads_user_convention():
    assert master_key("legacy_pilot_v1", "clean_voice_A", "CALL_001") == "eval-audio/legacy_pilot_v1/clean_voice_A/CALL_001.mp3"
    assert run_key("20261016T0100Z-ab12", "legacy_pilot_v1", "CALL_005") == "uploads/eval-20261016T0100Z-ab12/legacy_pilot_v1/CALL_005.mp3"
    for bad in ("../x", "run id", ""):
        with pytest.raises(ValueError):
            run_key(bad, "legacy_pilot_v1", "CALL_001")
    with pytest.raises(ValueError):
        run_key("run1", "legacy_pilot_v1", "call_1")
    with pytest.raises(ValueError):
        master_key("legacy_pilot_v1", "1bad", "CALL_001")


# ---- the real production handler, with the AWS client faked ---------------------------------------------------------------------------------

@pytest.fixture
def handler(monkeypatch, tmp_path):
    settings = load_settings({"EVAL_BUCKET": "fyp-eval-bucket", "TRANSCRIBE_MAX_SPEAKER_LABELS": "2"}, tmp_path / "none")
    for name in ("TRANSCRIBE_OUTPUT_BUCKET", "TRANSCRIBE_LANGUAGE_CODE", "TRANSCRIBE_MAX_SPEAKER_LABELS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    client = MagicMock()
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: client)
    for name, value in settings.handler_environment().items():
        monkeypatch.setenv(name, value)
    module = load_handler(settings.handler_path)
    return module, client


def test_the_unmodified_production_handler_starts_one_job_per_call(handler):
    module, client = handler
    keys = [run_key("r1", "legacy_pilot_v1", "CALL_001"), run_key("r1", "legacy_pilot_v1", "CALL_028")]
    result = module.handler(s3_event("fyp-eval-bucket", keys), None)
    assert len(result["started_jobs"]) == 2
    first, second = (call.kwargs for call in client.start_transcription_job.call_args_list)
    assert first["Media"] == {"MediaFileUri": f"s3://fyp-eval-bucket/{keys[0]}"}
    assert first["OutputBucketName"] == "fyp-eval-bucket" and first["OutputKey"] == f"transcripts/{first['TranscriptionJobName']}.json"
    assert first["Settings"] == {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 2}
    assert first["MediaFormat"] == "mp3" and first["LanguageCode"] == "en-US"  # the handler's own default
    digest = hashlib.sha256(keys[0].encode()).hexdigest()[:16]
    assert first["TranscriptionJobName"] == f"transcribe-eval-r1-{digest}" and second["TranscriptionJobName"] != first["TranscriptionJobName"]


def test_each_run_gets_its_own_job_names_so_reruns_do_not_collide(handler):
    module, client = handler
    one = module.handler(s3_event("fyp-eval-bucket", [run_key("run-a", "legacy_pilot_v1", "CALL_001")]), None)["started_jobs"]
    two = module.handler(s3_event("fyp-eval-bucket", [run_key("run-b", "legacy_pilot_v1", "CALL_001")]), None)["started_jobs"]
    assert one != two


def test_the_production_lambda_is_the_only_source_of_the_handler(handler, tmp_path):
    module, _ = handler
    expected = config.FYP_BE_ROOT / "lambda" / "transcribe-trigger" / "app" / "handler.py"
    assert expected.is_file()
    assert Path(module.__file__).resolve() == expected.resolve()
    # no setting can redirect the engine to another handler
    settings = load_settings({"EVAL_BUCKET": "my-bucket", "HANDLER_PATH": str(tmp_path / "other.py")}, tmp_path / "none")
    assert settings.handler_path == config.HANDLER_PATH


def test_every_transcribe_setting_reaches_the_handler_but_never_the_output_bucket(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TRANSCRIBE_VOCABULARY_NAME=my-vocab\nTRANSCRIBE_OUTPUT_BUCKET=production-bucket\nOTHER=1\n", encoding="utf-8")
    settings = load_settings({"EVAL_BUCKET": "eval-bucket", "TRANSCRIBE_LANGUAGE_CODE": "en-US"}, env_file)
    assert settings.handler_environment() == {"TRANSCRIBE_OUTPUT_BUCKET": "eval-bucket", "TRANSCRIBE_LANGUAGE_CODE": "en-US", "TRANSCRIBE_VOCABULARY_NAME": "my-vocab"}
