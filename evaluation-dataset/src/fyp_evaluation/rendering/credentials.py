"""AWS credentials for the Polly renderer.

Same convention as `llm/env.py`: `.env` is read from the evaluation-dataset project root (never the working directory) with
python-dotenv's `dotenv_values`, and a non-empty variable in the operating-system environment always wins, so nothing is overwritten and
`os.environ` is never mutated (the OpenAI and Gemini keys in the same file are not even read). The pair AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY is taken as a whole from ONE source (the environment if it has both, else `.env`), so a key and a secret from different
places are never mixed; AWS_SESSION_TOKEN is used when that same source has it and is never required. Values are returned only to the caller
that builds the Boto3 session; they are never printed, logged or stored, and errors name variables, not values.

The region is NOT taken from AWS_DEFAULT_REGION: the renderer's `--region` (default ap-southeast-1) is always explicit.
"""

import os
from pathlib import Path

from fyp_evaluation.llm.errors import CredentialsError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
OPTIONAL = ("AWS_SESSION_TOKEN",)


def _clean(mapping):
    return {name: (mapping.get(name) or "").strip() for name in REQUIRED + OPTIONAL}


def aws_credentials(project_root=None, environ=None):
    """{aws_access_key_id, aws_secret_access_key[, aws_session_token]} for `boto3.Session`; raises CredentialsError when incomplete."""
    sources = [("the operating-system environment", _clean(os.environ if environ is None else environ))]
    path = Path(PROJECT_ROOT if project_root is None else project_root) / ".env"
    if path.is_file():
        from dotenv import dotenv_values

        sources.append((str(path), _clean(dotenv_values(path))))
    for _, values in sources:
        if all(values[name] for name in REQUIRED):
            credentials = {"aws_access_key_id": values["AWS_ACCESS_KEY_ID"], "aws_secret_access_key": values["AWS_SECRET_ACCESS_KEY"]}
            if values["AWS_SESSION_TOKEN"]:
                credentials["aws_session_token"] = values["AWS_SESSION_TOKEN"]
            return credentials
    raise CredentialsError(f"{' and '.join(REQUIRED)} are not both set. Set them in the operating-system environment or in {path}. "
                           f"AWS_SESSION_TOKEN is optional. No request was made.")


def polly_client(region, project_root=None, environ=None):
    """A Polly client in `region` using only the credentials above (never the shared credentials file or a profile)."""
    import boto3

    return boto3.Session(region_name=region, **aws_credentials(project_root, environ)).client("polly")
