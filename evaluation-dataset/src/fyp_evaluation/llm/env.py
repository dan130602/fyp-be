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


def get_api_key(project_root=PROJECT_ROOT, variable=API_KEY_VARIABLE):
    """The key named `variable` (default OPENAI_API_KEY; G5 passes GEMINI_API_KEY)."""
    key = os.environ.get(variable, "").strip()
    path = Path(project_root) / ".env"
    if not key and path.is_file():
        from dotenv import dotenv_values

        key = (dotenv_values(path).get(variable) or "").strip()
    if not key:
        raise CredentialsError(
            f"{variable} is not set. Set it in the operating-system environment or in {path} "
            f"(copy .env.example). No request was made.")
    return key
