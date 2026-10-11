"""G5 Gemini transport against the real google-genai SDK with a mocked httpx transport (no network, dummy key).

Proves: exactly one HTTP request per send for every failure class (the SDK's automatic retries are off), the error
classification, the exact request JSON the SDK puts on the wire, and that a completed interaction comes back as a dict.
"""

import json

import httpx
import pytest

from fyp_evaluation.llm.errors import AmbiguousError, CapacityError, CredentialsError, FatalRequestError
from fyp_evaluation.llm.gemini_transport import GeminiTransport, classify
from fyp_evaluation.plan_verifications.request import RequestBuilder
from g5_support import SCHEMAS, shared_inputs

pytest.importorskip("google.genai")


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def body():
    return RequestBuilder(shared_inputs(), SCHEMAS).build("CALL_021", "A001")


def transport_with(handler):
    seen = []

    def recording(request):
        seen.append(request)
        return handler(request)
    client = httpx.Client(transport=httpx.MockTransport(recording))
    return GeminiTransport(GeminiTransport.build_client("dummy-test-key", timeout_ms=5000, httpx_client=client)), seen


def status_response(code):
    return lambda request: httpx.Response(code, json={"error": {"code": code, "message": "mocked", "status": "MOCKED"}})


@pytest.mark.parametrize("code, error_class", [(503, CapacityError), (429, CapacityError), (400, FatalRequestError),
                                               (401, FatalRequestError), (404, FatalRequestError), (500, AmbiguousError),
                                               (504, AmbiguousError)])
def test_one_http_request_per_send_for_every_status(body, code, error_class):
    transport, seen = transport_with(status_response(code))
    with pytest.raises(error_class):
        transport.create_response(body)
    assert len(seen) == 1


@pytest.mark.parametrize("exception", [httpx.ReadTimeout("mocked timeout"), httpx.ConnectError("mocked connection failure")])
def test_timeouts_and_connection_errors_are_ambiguous_and_never_resent(body, exception):
    def fail(request):
        raise exception
    transport, seen = transport_with(fail)
    with pytest.raises(AmbiguousError):
        transport.create_response(body)
    assert len(seen) == 1


def test_wire_request_carries_the_approved_settings_and_a_completed_interaction_is_returned(body):
    report = {"checks": [], "issues": []}

    def ok(request):
        return httpx.Response(200, json={
            "id": "int_mock", "status": "completed", "model": "gemini-3.8-flash", "service_tier": "flex",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": json.dumps(report)}]}],
            "usage": {"total_input_tokens": 10, "total_output_tokens": 5, "total_thought_tokens": 7, "total_tokens": 22}})
    transport, seen = transport_with(ok)
    result = transport.create_response(body)
    assert len(seen) == 1
    sent = json.loads(seen[0].content)
    assert seen[0].url.path.endswith("/interactions")
    assert sent["model"] == "gemini-3.8-flash" and sent["service_tier"] == "flex" and sent["store"] is False
    assert sent["generation_config"] == {"thinking_level": "medium", "max_output_tokens": 32768}
    assert sent["response_format"]["type"] == "text" and sent["response_format"]["mime_type"] == "application/json"
    assert sent["response_format"]["schema"] == body["response_format"]["schema"]
    assert sent["system_instruction"] == body["system_instruction"] and sent["input"] == body["input"]
    for absent in ("tools", "previous_interaction_id", "background", "stream", "response_mime_type"):
        assert absent not in sent
    assert result["status"] == "completed" and result["service_tier"] == "flex" and json.loads(result["output_text"]) == report
    assert result["usage"]["total_thought_tokens"] == 7


def test_sdk_default_would_have_retried(body):
    """Control: the same mocked 503 against an SDK client WITHOUT our override is sent several times."""
    from google import genai
    from google.genai import types

    seen = []

    def recording(request):
        seen.append(request)
        return httpx.Response(503, json={"error": {"code": 503, "message": "mocked"}})
    client = genai.Client(api_key="dummy-test-key", http_options=types.HttpOptions(
        timeout=5000, httpx_client=httpx.Client(transport=httpx.MockTransport(recording)),
        retry_options=types.HttpRetryOptions(attempts=0, initial_delay=0.01, max_delay=0.01)))
    with pytest.raises(CapacityError):
        GeminiTransport(client).create_response(body)
    assert len(seen) > 1  # attempts=0 is rewritten to 1 and read as a retry count: why the transport overrides it


def test_classify_maps_unknown_failures_to_ambiguous():
    class Weird(Exception):
        pass

    class Status(Exception):
        def __init__(self, status_code):
            super().__init__("x")
            self.status_code = status_code
    assert isinstance(classify(Weird("?")), AmbiguousError)
    assert isinstance(classify(Status(503)), CapacityError)
    assert isinstance(classify(Status(422)), FatalRequestError)
    assert isinstance(classify(Status(502)), AmbiguousError)


def test_from_env_requires_the_gemini_key_before_building_anything(tmp_path):
    with pytest.raises(CredentialsError, match="GEMINI_API_KEY"):
        GeminiTransport.from_env(project_root=tmp_path)
