"""Gemini Batch (file input) behind the transport interface `StageRunner.submit` / `collect` already use for OpenAI Batch.

The stage runner keeps owning intent, attempts, custom IDs, request hashes, ambiguity handling, freshness and acceptance. This module
only translates, in both directions, between the runner's OpenAI-style batch lines and the Gemini batch file format:

  runner line   {"custom_id", "method", "url", "body": <Interactions request body>}
  Gemini line   {"key": custom_id, "request": <GenerateContent request>}                 (input file)
  Gemini result {"key", "response": <GenerateContentResponse>} | {"key", "error": {...}}   (output file)
  runner result {"custom_id", "response": {"status_code": 200, "body": <interaction-shaped body>}} | {"custom_id", "error": {...}}

The translated response body has the shape `gemini_runner.parse_interaction` reads (status, output_text, usage, model, service_tier),
so the Flex parsing, cost estimate and validation stay one code path. The tier reported is "batch". Nothing here retries or falls back.
"""

import io
import json

from .errors import AmbiguousError, CapacityError, FatalRequestError, TransportError
from .gemini_transport import GeminiTransport, _json_safe, classify

BATCH_TIER = "batch"
ENDPOINT = "/v1/responses"  # the endpoint the shared runner records; Gemini has none, so this is only the runner's label

_STATUS = {
    "JOB_STATE_SUCCEEDED": "completed", "JOB_STATE_PARTIALLY_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed", "JOB_STATE_CANCELLED": "cancelled", "JOB_STATE_CANCELLING": "in_progress",
    "JOB_STATE_EXPIRED": "expired",
}


def to_generate_request(body):
    """The GenerateContent request equivalent to one runner (Interactions-shaped) request body."""
    configuration = body.get("generation_config") or {}
    response_format = body.get("response_format") or {}
    generation = {"maxOutputTokens": configuration["max_output_tokens"],
                  "responseMimeType": response_format.get("mime_type", "application/json"),
                  "responseJsonSchema": response_format["schema"]}
    if configuration.get("thinking_level"):
        generation["thinkingConfig"] = {"thinkingLevel": str(configuration["thinking_level"]).upper()}
    request = {"contents": [{"role": "user", "parts": [{"text": body["input"]}]}], "generationConfig": generation}
    if body.get("system_instruction"):
        request["systemInstruction"] = {"parts": [{"text": body["system_instruction"]}]}
    return request


def _field(obj, camel, snake=None):
    if not isinstance(obj, dict):
        return None
    return obj.get(camel) if camel in obj else obj.get(snake or camel)


def to_interaction_body(response, model):
    """An interaction-shaped body for one GenerateContent response (a plain dict, camelCase or snake_case)."""
    candidates = _field(response, "candidates") or []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    parts = _field(_field(candidate, "content"), "parts") or []
    text = "".join(p.get("text") or "" for p in parts if isinstance(p, dict) and not p.get("thought"))
    finish = str(_field(candidate, "finishReason", "finish_reason") or "")
    status = "completed" if finish == "STOP" else "incomplete" if finish == "MAX_TOKENS" else "failed"
    usage = _field(response, "usageMetadata", "usage_metadata") or {}
    served = _field(response, "modelVersion", "model_version")
    body = {
        "id": _field(response, "responseId", "response_id"),
        # The batch result names a model version (possibly with a suffix); a different model must still be caught, so only a
        # name that begins with the requested model is reported as the requested model.
        "model": model if isinstance(served, str) and served.startswith(model) else (served or model),
        "model_version": served,
        "status": status,
        "service_tier": BATCH_TIER,
        "output_text": text,
        "usage": {
            "total_input_tokens": int(_field(usage, "promptTokenCount", "prompt_token_count") or 0),
            "total_cached_tokens": int(_field(usage, "cachedContentTokenCount", "cached_content_token_count") or 0),
            "total_output_tokens": int(_field(usage, "candidatesTokenCount", "candidates_token_count") or 0),
            "total_thought_tokens": int(_field(usage, "thoughtsTokenCount", "thoughts_token_count") or 0),
            "total_tokens": int(_field(usage, "totalTokenCount", "total_token_count") or 0),
        },
    }
    return body


def input_file_text(data):
    """Gemini input JSONL for the runner's batch JSONL bytes (lines are kept in order)."""
    out = []
    for raw in data.decode("utf-8").splitlines():
        if raw.strip():
            line = json.loads(raw)
            out.append(json.dumps({"key": line["custom_id"], "request": to_generate_request(line["body"])},
                                  sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(out) + "\n"


def result_file_text(data, model):
    """The runner's result lines for a Gemini output file (bytes in, text out). Unknown shapes become a retryable error line."""
    out = []
    for raw in data.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        key = line.get("key")
        if not isinstance(key, str) or not key:
            continue  # no key: the runner reports it as a missing result for the affected attempt
        response, error = line.get("response"), line.get("error")
        if isinstance(response, dict) and not error:
            converted = {"custom_id": key, "response": {"status_code": 200, "body": to_interaction_body(response, model)}}
        else:
            detail = error if isinstance(error, dict) else {}
            converted = {"custom_id": key, "error": {"code": str(detail.get("code") or "unknown"),
                                                      "message": str(detail.get("message") or "")[:300]}}
        out.append(json.dumps(converted, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(out) + "\n"


class GeminiBatchTransport(GeminiTransport):
    """Upload, create, poll and download one Gemini batch, speaking the runner's batch interface."""

    def __init__(self, client, model):
        super().__init__(client)
        self._model = model

    @classmethod
    def from_env(cls, project_root=None, *, variable="GEMINI_API_KEY", timeout_ms=900_000, model="gemini-3.8-flash"):
        from .env import PROJECT_ROOT, get_api_key

        return cls(cls.build_client(get_api_key(project_root or PROJECT_ROOT, variable), timeout_ms=timeout_ms), model)

    def upload_batch_file(self, data, filename):
        from google.genai import types

        try:
            text = input_file_text(data)
            uploaded = self._client.files.upload(
                file=io.BytesIO(text.encode("utf-8")), config=types.UploadFileConfig(mime_type="jsonl", display_name=filename))
        except Exception as error:  # noqa: BLE001 - translated, never swallowed
            raise self._not_created(error) from error
        return uploaded.name

    def create_batch(self, file_id, metadata):
        try:
            job = self._client.batches.create(model=self._model, src=file_id, config={"display_name": json.dumps(metadata, sort_keys=True)})
        except Exception as error:  # noqa: BLE001
            translated = classify(error)
            if isinstance(translated, (CapacityError, FatalRequestError)):
                raise translated from error
            raise AmbiguousError(str(translated)) from error  # a timeout or 5xx: the batch may exist and be billed
        return job.name

    def retrieve_batch(self, batch_id):
        try:
            job = self._client.batches.get(name=batch_id)
        except Exception as error:  # noqa: BLE001
            raise classify(error) from error
        state = getattr(job.state, "name", str(job.state))
        stats = getattr(job, "completion_stats", None)
        counts = None
        if stats is not None:
            ok, failed = int(getattr(stats, "successful_count", 0) or 0), int(getattr(stats, "failed_count", 0) or 0)
            counts = {"total": ok + failed + int(getattr(stats, "incomplete_count", 0) or 0), "completed": ok, "failed": failed}
        try:
            metadata = json.loads(job.display_name or "{}")
        except ValueError:
            metadata = {}
        dest = getattr(job, "dest", None)
        source = getattr(job, "src", None)
        return {"id": job.name, "status": _STATUS.get(state, "in_progress"), "endpoint": ENDPOINT, "metadata": metadata,
                "input_file_id": getattr(source, "file_name", None), "output_file_id": getattr(dest, "file_name", None),
                "error_file_id": None, "request_counts": counts, "errors": _json_safe(getattr(job, "error", None)),
                "usage": None, "provider_state": state}

    def download_file(self, file_id):
        try:
            data = self._client.files.download(file=file_id)
        except Exception as error:  # noqa: BLE001
            raise classify(error) from error
        if not isinstance(data, (bytes, bytearray)):
            raise TransportError("the batch output could not be downloaded as bytes")
        return result_file_text(bytes(data), self._model).encode("utf-8")

    @staticmethod
    def _not_created(error):
        """A failure before any batch exists is never billed, so even an unknown failure here is safe to retry."""
        translated = classify(error)
        return translated if isinstance(translated, (CapacityError, FatalRequestError)) else TransportError(str(translated))
