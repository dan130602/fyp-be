"""Gemini transport: one synchronous Interactions API request per call, with provider exceptions mapped to `errors`.

Transport only: it sends the body it is given and returns a plain dict. It never retries, and the SDK must not either:
by default the Interactions client retries 408/409/429/5xx, connection errors AND timeouts (3 retries), which could
silently resend a paid request. `HttpRetryOptions` cannot switch that off in google-genai 2.25: client construction
rewrites `attempts=0` to 1 (`_api_client.retry_args`) and the Interactions client reads `attempts` as the RETRY count, so
`attempts=0` and `attempts=1` both still retry once (proven offline against a mocked 503). The transport therefore sets the
Interactions client's own retry configuration to strategy "none" and verifies it; a mocked-httpx test asserts exactly one
HTTP request per send. The application is the single retry owner. The SDK is imported lazily and the key is read only
when a transport is built.

Error mapping:
  400/401/403/404/409/422 and other 4xx  -> FatalRequestError  (configuration, credentials, unsupported setting or model)
  429 / 503                              -> CapacityError      (definite capacity rejection; safe to resend the same bytes)
  timeout, connection error, other 5xx   -> AmbiguousError     (the request may have been processed and billed)
"""

from .env import PROJECT_ROOT, get_api_key
from .errors import AmbiguousError, CapacityError, FatalRequestError, TransportError

CAPACITY_STATUS = {429, 503}


def _message(error):
    return str(getattr(error, "message", None) or error)[:300]


def _class_names(error):
    return {cls.__name__ for cls in type(error).__mro__}


def classify(error):
    """Translate an SDK/httpx exception into the transport error taxonomy (never swallowed, never retried here)."""
    if isinstance(error, TransportError):
        return error
    names = _class_names(error)
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None) if isinstance(getattr(error, "code", None), int) else None
    label = f"{type(error).__name__}: {_message(error)}"
    if names & {"APITimeoutError", "TimeoutException", "APIConnectionError", "ConnectError", "NetworkError", "RemoteProtocolError",
                "NoResponseError"}:
        return AmbiguousError(label)
    if isinstance(status, int):
        if status in CAPACITY_STATUS:
            return CapacityError(f"HTTP {status} {label}")
        if 400 <= status < 500:
            return FatalRequestError(f"HTTP {status} {label}")
        return AmbiguousError(f"HTTP {status} {label}")
    return AmbiguousError(label)  # unknown failure after the request may have left: never assume it was not processed


def _json_safe(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


class GeminiTransport:
    """Thin wrapper over the official google-genai SDK. Construct with `from_env()`; inject a fake in tests instead."""

    def __init__(self, client):
        self._client = client

    @classmethod
    def build_client(cls, api_key, *, timeout_ms, httpx_client=None):
        from google import genai
        from google.genai import types
        from google.genai._gaos.utils import RetryConfig

        options = {"timeout": timeout_ms}
        if httpx_client is not None:
            options["httpx_client"] = httpx_client
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**options))
        configuration = client.interactions.sdk_configuration  # the cached resource every create() uses
        configuration.retry_config = RetryConfig("none", None, False)
        if getattr(configuration.retry_config, "strategy", None) != "none":
            raise FatalRequestError("could not disable the SDK's automatic retries; refusing to send paid requests")
        return client

    @classmethod
    def from_env(cls, project_root=PROJECT_ROOT, *, variable="GEMINI_API_KEY", timeout_ms=900_000):
        api_key = get_api_key(project_root, variable)  # raises CredentialsError before anything is constructed or sent
        return cls(cls.build_client(api_key, timeout_ms=timeout_ms))

    def create_response(self, body):
        try:
            interaction = self._client.interactions.create(**body)
        except Exception as error:  # noqa: BLE001 - translated, never swallowed
            raise classify(error) from error
        result = _json_safe(interaction)
        if not isinstance(result, dict):
            raise AmbiguousError(f"unexpected interaction object {type(interaction).__name__}")
        text = getattr(interaction, "output_text", None)
        if text is not None and "output_text" not in result:
            result["output_text"] = text
        return result
