"""Project-local credentials.

`.env` is read from the evaluation-dataset project root (never the working directory). A non-empty variable in
the operating-system environment always wins; `.env` is only a fallback and is read with `dotenv_values`, so
`os.environ` is never mutated. The key is returned only to the caller that builds a transport; it is never
printed, logged or persisted.
"""

import os
from pathlib import Path

from .errors import CredentialsError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
API_KEY_VARIABLE = "OPENAI_API_KEY"


def get_api_key(project_root=PROJECT_ROOT):
    key = os.environ.get(API_KEY_VARIABLE, "").strip()
    path = Path(project_root) / ".env"
    if not key and path.is_file():
        from dotenv import dotenv_values

        key = (dotenv_values(path).get(API_KEY_VARIABLE) or "").strip()
    if not key:
        raise CredentialsError(
            f"{API_KEY_VARIABLE} is not set. Set it in the operating-system environment or in {path} "
            f"(copy .env.example). No request was made.")
    return key
