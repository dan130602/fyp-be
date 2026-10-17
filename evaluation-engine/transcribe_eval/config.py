"""Settings for the transcription evaluation.

Read from this folder's own `.env` and the operating-system environment (the environment wins). The outer fyp-be `.env`, which
docker-compose reads, is never used, so evaluation credentials cannot leak into the application stack.

Production's Lambda settings (language, speaker count) are mirrored by optional variables. Production's Terraform currently uses en-US and 2
speakers, which are also the handler's own defaults, so leaving them empty is equivalent. Set them only if production's values ever change.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
FYP_BE_ROOT = ENGINE_ROOT.parent
ENV_FILE = ENGINE_ROOT / ".env"
HANDLER_PATH = FYP_BE_ROOT / "lambda" / "transcribe-trigger" / "app" / "handler.py"
DATASET_ROOT = FYP_BE_ROOT / "evaluation-dataset" / "datasets" / "advisory_execution_pilot_v1"
DEFAULT_REGION = "ap-southeast-1"

# Optional mirrors of the Lambda's environment; only set variables are passed on.
HANDLER_OPTIONAL = ("TRANSCRIBE_LANGUAGE_CODE", "TRANSCRIBE_MAX_SPEAKER_LABELS")
AWS_CREDENTIAL_NAMES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
BUCKET_NAME = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    region: str
    bucket: str
    handler_path: Path
    dataset_root: Path
    aws_credentials: dict  # only the variables that were set; never printed
    handler_optional: dict  # TRANSCRIBE_* values mirrored from production

    def handler_environment(self):
        """The environment the unmodified handler reads (the evaluation bucket is its output bucket)."""
        return {"TRANSCRIBE_OUTPUT_BUCKET": self.bucket, **self.handler_optional}

    def apply_to_environ(self, environ=None):
        """Make boto3 and the handler see these settings (call once, before the handler is imported)."""
        environ = os.environ if environ is None else environ
        environ["AWS_DEFAULT_REGION"] = self.region
        environ.update(self.aws_credentials)
        environ.update(self.handler_environment())


def _clean(value):
    return (value or "").strip()


def load_settings(environ=None, env_file=ENV_FILE):
    """Settings from the process environment over the `.env` file; raises ConfigError naming the problem (never a secret)."""
    environ = os.environ if environ is None else environ
    file_values = {}
    if Path(env_file).is_file():
        from dotenv import dotenv_values

        file_values = dotenv_values(env_file)

    def get(name):
        return _clean(environ.get(name)) or _clean(file_values.get(name))

    bucket = get("EVAL_BUCKET")
    if not bucket:
        raise ConfigError("EVAL_BUCKET is not set (put it in evaluation-engine/.env)")
    if not BUCKET_NAME.fullmatch(bucket):
        raise ConfigError(f"EVAL_BUCKET {bucket!r} is not a valid S3 bucket name")
    return Settings(
        region=get("AWS_REGION") or DEFAULT_REGION,
        bucket=bucket,
        handler_path=HANDLER_PATH,  # fixed: the production Lambda is the one source of truth, never overridable
        dataset_root=Path(get("DATASET_ROOT") or DATASET_ROOT),
        aws_credentials={name: get(name) for name in AWS_CREDENTIAL_NAMES if get(name)},
        handler_optional={name: get(name) for name in HANDLER_OPTIONAL if get(name)},
    )
