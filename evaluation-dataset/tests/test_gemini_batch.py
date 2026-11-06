"""Gemini Batch for G7 through the shared submit/collect lifecycle. A fake genai client; no network, no credentials."""
import json
from types import SimpleNamespace

import pytest

from fyp_evaluation.dialogue_verifications.run import DialogueVerificationBatchRunner, synthesize_response
from fyp_evaluation.llm import gemini_batch as gb
from fyp_evaluation.llm.errors import AmbiguousError, CapacityError, FatalRequestError
from g7_support import SCHEMAS, Clock, shared_inputs

CALLS = ["CALL_021", "CALL_041"]


def generate_response(text, *, finish="STOP", model="gemini-3.8-flash-001"):
    return {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": text}]}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 9000, "cachedContentTokenCount": 0, "candidatesTokenCount": 4000,
                              "thoughtsTokenCount": 6000, "totalTokenCount": 19000},
            "modelVersion": model, "responseId": "resp_1"}


class FakeGenai:
    """Just the surface `GeminiBatchTransport` uses."""

    def __init__(self, runner_ref):
        self.runner_ref, self.uploaded, self.created, self.state = runner_ref, {}, [], "JOB_STATE_RUNNING"
        self.create_error = None
        self.files = SimpleNamespace(upload=self._upload, download=self._download)
        self.batches = SimpleNamespace(create=self._create, get=self._get)
        self.outputs = {}

    def _upload(self, *, file, config):
        name = f"files/in{len(self.uploaded)}"
        self.uploaded[name] = file.read().decode("utf-8")
        return SimpleNamespace(name=name)

    def _create(self, *, model, src, config):
        if self.create_error:
            raise self.create_error
        self.created.append((model, src, config))
        return SimpleNamespace(name="batches/b1")

    def _get(self, *, name):
        return SimpleNamespace(
            name=name, state=SimpleNamespace(name=self.state), display_name=self.created[0][2]["display_name"],
            completion_stats=SimpleNamespace(successful_count=len(CALLS), failed_count=0, incomplete_count=0),
            src=SimpleNamespace(file_name=self.created[0][1]), dest=SimpleNamespace(file_name="files/out0"), error=None)

    def _download(self, *, file):
        return self.outputs[file]


def make(tmp_path):
    (tmp_path / "release").mkdir()
    client = FakeGenai(None)
    transport = gb.GeminiBatchTransport(client, "gemini-3.8-flash")
    runner = DialogueVerificationBatchRunner(tmp_path / "release", audit_root=tmp_path / "audit", schema_dir=SCHEMAS, transport=transport,
                                             clock=Clock(), inputs=shared_inputs())
    return runner, client


def test_request_translation_keeps_instruction_schema_and_thinking():
    body = {"system_instruction": "SYS", "input": "BRIEF", "generation_config": {"thinking_level": "high", "max_output_tokens": 32768},
            "response_format": {"type": "text", "mime_type": "application/json", "schema": {"type": "object"}}, "service_tier": "batch"}
    request = gb.to_generate_request(body)
    assert request["contents"] == [{"role": "user", "parts": [{"text": "BRIEF"}]}]
    assert request["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert request["generationConfig"] == {"maxOutputTokens": 32768, "responseMimeType": "application/json",
                                            "responseJsonSchema": {"type": "object"}, "thinkingConfig": {"thinkingLevel": "HIGH"}}


def test_response_translation_skips_thought_parts_and_maps_usage_and_status():
    body = gb.to_interaction_body(generate_response('{"a":1}'), "gemini-3.8-flash")
    assert body["output_text"] == '{"a":1}' and body["status"] == "completed" and body["service_tier"] == "batch"
    assert body["model"] == "gemini-3.8-flash" and body["model_version"] == "gemini-3.8-flash-001"
    assert body["usage"]["total_thought_tokens"] == 6000 and body["usage"]["total_output_tokens"] == 4000
    assert gb.to_interaction_body(generate_response("{", finish="MAX_TOKENS"), "gemini-3.8-flash")["status"] == "incomplete"
    other = gb.to_interaction_body(generate_response("{}", model="some-other-model"), "gemini-3.8-flash")
    assert other["model"] == "some-other-model"  # a different served model is surfaced so the parser can stop the run


def test_submit_collect_accepts_and_reconstructs_each_call(tmp_path):
    runner, client = make(tmp_path)
    result = runner.submit(CALLS)
    assert result["state"] == "SUBMITTED" and result["submitted"] == 2
    lines = [json.loads(line) for line in client.uploaded["files/in0"].splitlines()]
    assert [line["key"] for line in lines] == [runner.spec.custom_id(c, "A001") for c in CALLS]
    assert all(line["request"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"} for line in lines)
    assert runner.collect()["pending"] and not runner.collect()["collected"]  # still running: nothing ingested
    client.state = "JOB_STATE_SUCCEEDED"
    client.outputs["files/out0"] = "\n".join(
        json.dumps({"key": runner.spec.custom_id(c, "A001"),
                    "response": generate_response(json.dumps(synthesize_response(runner.builder.context(c))))}) for c in CALLS).encode()
    report = runner.collect()
    assert report["collected"] == ["SUB_0001"] and not report["anomalies"]
    assert [o[2] for o in report["outcomes"]] == ["ACCEPTED", "ACCEPTED"]
    attempt = runner.store.read_attempt(CALLS[0], "A001")
    assert attempt["transport"] == "batch" and attempt["params"]["service_tier"] == "batch" and attempt["service_tier"] == "batch"
    assert attempt["estimated_cost_usd"] == pytest.approx((9000 * 0.375 + (4000 + 6000) * 1.875) / 1_000_000)
    assert runner.validate() == [] and runner.check() == {"missing": [], "changed": [], "extra": []}


def test_one_failed_line_does_not_disturb_the_other_call(tmp_path):
    runner, client = make(tmp_path)
    runner.submit(CALLS)
    client.state = "JOB_STATE_SUCCEEDED"
    good = json.dumps({"key": runner.spec.custom_id(CALLS[0], "A001"),
                       "response": generate_response(json.dumps(synthesize_response(runner.builder.context(CALLS[0]))))})
    bad = json.dumps({"key": runner.spec.custom_id(CALLS[1], "A001"), "error": {"code": 503, "message": "overloaded"}})
    client.outputs["files/out0"] = (good + "\n" + bad).encode()
    report = runner.collect()
    states = {o[0]: o[2] for o in report["outcomes"]}
    assert states == {CALLS[0]: "ACCEPTED", CALLS[1]: "TRANSPORT_FAILED"}
    assert runner.store.read_attempt(CALLS[1], "A001")["fatal"] is False  # a retry is allowed under the existing budget


class Http(Exception):
    def __init__(self, status):
        self.status_code = status


@pytest.mark.parametrize("status,state", [(503, "NOT_SUBMITTED"), (500, "SUBMISSION_UNKNOWN")])
def test_creating_the_batch_distinguishes_capacity_from_unknown(tmp_path, status, state):
    runner, client = make(tmp_path)
    client.create_error = Http(status)
    assert runner.submit(CALLS)["state"] == state


def test_batch_runner_refuses_a_synchronous_request(tmp_path):
    runner, _ = make(tmp_path)
    with pytest.raises(Exception, match="only submits batches"):
        runner.run_sync("CALL_021")


def test_error_types_are_the_shared_taxonomy():
    assert issubclass(CapacityError, Exception) and issubclass(AmbiguousError, Exception) and issubclass(FatalRequestError, Exception)
