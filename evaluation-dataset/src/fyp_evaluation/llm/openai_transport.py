"""OpenAI transport: Responses API (sync) and Batch API, with provider exceptions mapped to `errors`.

Transport only: it sends bodies it is given and returns plain dicts. It knows nothing about prompts, schemas or
validation. The SDK is imported lazily so offline code and tests never need it, and the key is read only when
a transport is constructed (never at import time).
"""

from .env import PROJECT_ROOT, get_api_key
from .errors import QUOTA_CODES, AmbiguousError, FatalRequestError, RetryableError, TransportError

BATCH_ENDPOINT = "/v1/responses"
COMPLETION_WINDOW = "24h"
SYNC_MAX_RETRIES = 2
SYNC_TIMEOUT_SECONDS = 600.0


def is_quota_error(error):
    """True for a 429 that reports exhausted credits/quota rather than a transient rate limit."""
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    return QUOTA_CODES & {str(getattr(error, "code", None)), str(detail.get("code")), str(detail.get("type"))} != set()


def _json_safe(value):
    """SDK model -> plain JSON data (or None)."""
    if value is None:
        return None
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _message(error):
    return str(getattr(error, "message", None) or error)[:300]


class OpenAITransport:
    """Thin wrapper over the official SDK. Construct with `from_env()`; inject a fake in tests instead."""

    def __init__(self, client):
        self._client = client  # SDK retries on: idempotent reads and sync generation
        self._batch_client = client.with_options(max_retries=0)  # batch creation is never retried by the SDK

    @classmethod
    def from_env(cls, project_root=PROJECT_ROOT):
        api_key = get_api_key(project_root)  # raises CredentialsError before anything is constructed or sent
        import openai

        return cls(openai.OpenAI(api_key=api_key, max_retries=SYNC_MAX_RETRIES, timeout=SYNC_TIMEOUT_SECONDS))

    # ---- error mapping -------------------------------------------------------------------------------
    @staticmethod
    def _classify(error, *, creating_batch=False):
        import openai

        permanent = (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError,
                     openai.NotFoundError, openai.UnprocessableEntityError)
        if isinstance(error, TransportError):
            return error
        if isinstance(error, permanent):
            return FatalRequestError(f"{type(error).__name__}: {_message(error)}")
        if isinstance(error, openai.RateLimitError):
            if is_quota_error(error):  # a billing condition, not congestion: retrying cannot help and only burns the budget
                return FatalRequestError(f"InsufficientQuota: {_message(error)}")
            return RetryableError(f"RateLimitError: {_message(error)}")  # definitely not processed
        if creating_batch:
            return AmbiguousError(f"{type(error).__name__}: {_message(error)}")
        return RetryableError(f"{type(error).__name__}: {_message(error)}")

    # ---- sync ----------------------------------------------------------------------------------------
    def create_response(self, body):
        try:
            return self._client.responses.create(**body).model_dump(mode="json")
        except Exception as error:  # noqa: BLE001 - translated, never swallowed
            raise self._classify(error) from error

    # ---- batch ---------------------------------------------------------------------------------------
    def upload_batch_file(self, data, filename):
        try:
            return self._client.files.create(file=(filename, data), purpose="batch").id
        except Exception as error:  # noqa: BLE001
            raise self._classify(error) from error

    def create_batch(self, file_id, metadata):
        try:
            batch = self._batch_client.batches.create(
                input_file_id=file_id, endpoint=BATCH_ENDPOINT, completion_window=COMPLETION_WINDOW, metadata=metadata)
        except Exception as error:  # noqa: BLE001
            raise self._classify(error, creating_batch=True) from error
        return batch.id

    def retrieve_batch(self, batch_id):
        try:
            batch = self._client.batches.retrieve(batch_id)
        except Exception as error:  # noqa: BLE001
            raise self._classify(error) from error
        counts = getattr(batch, "request_counts", None)
        return {
            "id": batch.id, "status": batch.status, "endpoint": batch.endpoint,
            "metadata": dict(batch.metadata or {}), "input_file_id": batch.input_file_id,
            "output_file_id": batch.output_file_id, "error_file_id": batch.error_file_id,
            "request_counts": None if counts is None else {
                "total": counts.total, "completed": counts.completed, "failed": counts.failed},
            "errors": _json_safe(getattr(batch, "errors", None)),  # batch-level errors (e.g. rejected before any result file)
            "usage": _json_safe(getattr(batch, "usage", None)),
        }

    def download_file(self, file_id):
        try:
            return self._client.files.content(file_id).read()
        except Exception as error:  # noqa: BLE001
            raise self._classify(error) from error
