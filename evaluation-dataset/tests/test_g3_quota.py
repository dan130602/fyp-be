"""Exhausted credits are a billing condition, not a rate limit: fatal, never retried, never charged to the budget."""

import pytest

from fyp_evaluation.llm.errors import FatalRequestError, RetryableError
from g3_support import CALL_IDS, make_runner


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def rate_limit_error(code, kind="insufficient_quota"):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    body = {"error": {"message": "You have no credits remaining.", "type": kind, "param": None, "code": code}}
    return openai.RateLimitError("limit", response=httpx.Response(429, request=request), body=body)


@pytest.mark.parametrize("code", ["credit_balance_exhausted", "insufficient_quota"])
def test_a_429_for_exhausted_credits_is_fatal_for_sync_and_batch_creation(code):
    openai = pytest.importorskip("openai")
    from fyp_evaluation.llm.openai_transport import OpenAITransport

    class Responses:
        def create(self, **_):
            raise rate_limit_error(code)

    class Batches:
        def create(self, **_):
            raise rate_limit_error(code)

    class Client:
        responses, batches = Responses(), Batches()

        def with_options(self, **_):
            return self

    transport = OpenAITransport(Client())
    with pytest.raises(FatalRequestError, match="InsufficientQuota"):
        transport.create_response({"model": "x"})
    with pytest.raises(FatalRequestError, match="InsufficientQuota"):
        transport.create_batch("file_1", {})
    assert openai  # the SDK is installed in the dev environment


def test_an_ordinary_429_stays_retryable():
    pytest.importorskip("openai")
    from fyp_evaluation.llm.openai_transport import OpenAITransport

    class Responses:
        def create(self, **_):
            raise rate_limit_error("rate_limit_exceeded", kind="requests")

    class Client:
        responses = Responses()
        batches = None

        def with_options(self, **_):
            return self

    with pytest.raises(RetryableError):
        OpenAITransport(Client()).create_response({"model": "x"})


def test_a_batch_result_line_reporting_exhausted_credits_is_fatal_and_not_charged(tmp_path):
    runner, fake = make_runner(tmp_path)
    report = runner.submit(CALL_IDS[:2])
    fake.finish_batch(report["batch_id"], errors={"CALL_001-G3-A001": (429, "credit_balance_exhausted", "no credits")})
    result = runner.collect()
    assert result["fatal"] is True and any("exhausted credits" in a for a in result["anomalies"])
    attempt = runner.store.read_attempt("CALL_001", "A001")
    assert attempt["state"] == "TRANSPORT_FAILED" and attempt["fatal"] is True
    assert runner.call_state("CALL_001")["transport_failed"] == 0
