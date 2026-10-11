"""Deterministic in-memory stand-in for `GeminiTransport` (offline tests). Never touches the network."""

import json


def interaction_body(text, *, status="completed", model="gemini-3.8-flash", service_tier="flex", usage=None,
                     interaction_id="int_fake", steps=None):
    """A provider-shaped Interactions body carrying `text` as the structured output."""
    body = {
        "id": interaction_id, "status": status, "model": model, "service_tier": service_tier,
        "output_text": text,
        "steps": steps if steps is not None else [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": usage or {"total_input_tokens": 9000, "total_cached_tokens": 3000, "total_output_tokens": 2500,
                           "total_thought_tokens": 4000, "total_tokens": 15500},
    }
    if service_tier is None:
        del body["service_tier"]
    return body


def call_id_of(body):
    """The call a G5 request body is about (the request carries no metadata block, so read the brief)."""
    return json.loads(body["input"])["call"]["call_id"]


class FakeGeminiTransport:
    """Scriptable transport. `responder(body) -> interaction dict`. `errors` is a list of exceptions raised one per send
    (in order) before the responder is consulted. Every send is recorded in `calls`."""

    def __init__(self, responder=None, errors=None):
        self.responder = responder
        self.errors = list(errors or [])
        self.calls = []

    def create_response(self, body):
        self.calls.append(("create_response", call_id_of(body)))
        if self.errors:
            raise self.errors.pop(0)
        return self.responder(body)
