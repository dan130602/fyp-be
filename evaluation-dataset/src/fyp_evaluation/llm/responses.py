"""Interpret a provider Responses API body. Shared by the sync and batch paths (the only parser).

Outcomes:
  ("ok", parsed_json, None)        structured output parsed
  ("rejected", None, reason)       incomplete / refusal / unparseable: a recorded, semantic-budget attempt
  ("transport", None, reason)      the provider reports a failed response: a technical, retryable failure
"""

from fyp_evaluation.call_specs.compile import parse_json_text

_TRANSIENT_CODES = {"server_error", "rate_limit_exceeded", "timeout", "overloaded"}


def parse_provider_response(body):
    status = body.get("status")
    if status == "incomplete":
        reason = (body.get("incomplete_details") or {}).get("reason") or "unknown"
        return "rejected", None, f"incomplete response: {reason}"
    if status in ("failed", "cancelled", "queued", "in_progress"):
        code = (body.get("error") or {}).get("code") or status
        return "transport", None, f"provider response status {status} ({code})"
    if status != "completed":
        return "rejected", None, f"unexpected response status {status!r}"
    texts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "refusal":
                return "rejected", None, f"model refusal: {str(part.get('refusal'))[:200]}"
            if part.get("type") == "output_text":
                texts.append(part.get("text") or "")
    if not texts:
        return "rejected", None, "response contains no output text"
    try:
        parsed = parse_json_text("".join(texts))
    except ValueError as error:
        return "rejected", None, f"unparseable structured output: {error}"
    if not isinstance(parsed, dict):
        return "rejected", None, "structured output is not a JSON object"
    return "ok", parsed, None


def is_transient_error_code(code):
    return code in _TRANSIENT_CODES
