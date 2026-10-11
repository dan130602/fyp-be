"""Deterministic in-memory transport for offline tests and demonstrations. Never touches the network."""

import json


def response_body(text, *, status="completed", usage=None, response_id="resp_fake", incomplete_reason=None,
                  refusal=None, error=None):
    """A provider-shaped Responses body carrying `text` as structured output."""
    content = [{"type": "refusal", "refusal": refusal}] if refusal else [{"type": "output_text", "text": text}]
    body = {
        "id": response_id, "object": "response", "status": status, "model": "gpt-6.1-sol",
        "output": [{"type": "message", "role": "assistant", "content": content}],
        "usage": usage or {"input_tokens": 6000, "output_tokens": 2500,
                           "input_tokens_details": {"cached_tokens": 0},
                           "output_tokens_details": {"reasoning_tokens": 1000}},
        "service_tier": "default",
    }
    if incomplete_reason:
        body["incomplete_details"] = {"reason": incomplete_reason}
    if error:
        body["error"] = error
    return body


class FakeTransport:
    """Scriptable stand-in for `OpenAITransport`.

    `responder(request_body) -> response body dict` produces answers. Failure injection is by attribute
    (`sync_error`, `upload_error`, `create_error`, `retrieve_error`): an exception instance raised once, then
    cleared. Every call is recorded in `calls` so tests can assert that nothing was sent.
    """

    def __init__(self, responder=None):
        self.responder = responder
        self.calls = []
        self.files, self.batches = {}, {}
        self.sync_error = self.upload_error = self.create_error = self.retrieve_error = None
        self.create_succeeds_despite_error = False  # the batch exists although the caller saw an error
        self._counter = 0

    def _next(self, prefix):
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    @staticmethod
    def _raise_once(owner, name):
        error = getattr(owner, name)
        if error is not None:
            setattr(owner, name, None)
            raise error

    # ---- sync ----------------------------------------------------------------------------------------
    def create_response(self, body):
        self.calls.append(("create_response", body["metadata"]["call_id"]))
        self._raise_once(self, "sync_error")
        return self.responder(body)

    # ---- batch ---------------------------------------------------------------------------------------
    def upload_batch_file(self, data, filename):
        self.calls.append(("upload_batch_file", filename))
        self._raise_once(self, "upload_error")
        file_id = self._next("file")
        self.files[file_id] = data
        return file_id

    def create_batch(self, file_id, metadata):
        self.calls.append(("create_batch", file_id))
        error, self.create_error = self.create_error, None
        if error is not None:
            if self.create_succeeds_despite_error:
                self.last_hidden_batch = self._register_batch(file_id, metadata)
            raise error
        return self._register_batch(file_id, metadata)

    def _register_batch(self, file_id, metadata):
        batch_id = self._next("batch")
        lines = [json.loads(line) for line in self.files[file_id].decode("utf-8").splitlines() if line.strip()]
        self.batches[batch_id] = {
            "id": batch_id, "status": "in_progress", "endpoint": "/v1/responses", "metadata": dict(metadata),
            "input_file_id": file_id, "output_file_id": None, "error_file_id": None,
            "request_counts": {"total": len(lines), "completed": 0, "failed": 0}, "errors": None, "usage": None, "_lines": lines,
        }
        return batch_id

    def retrieve_batch(self, batch_id):
        self.calls.append(("retrieve_batch", batch_id))
        self._raise_once(self, "retrieve_error")
        batch = self.batches[batch_id]
        return {key: value for key, value in batch.items() if not key.startswith("_")}

    def download_file(self, file_id):
        self.calls.append(("download_file", file_id))
        return self.files[file_id]

    # ---- test controls -------------------------------------------------------------------------------
    def finish_batch(self, batch_id, *, status="completed", shuffle=False, drop=(), duplicate=(), errors=None,
                     unknown=(), responder=None, batch_errors=None, usage=None):
        """Create output/error files. `errors` = {custom_id: (status_code, code, message)}; `drop` = ids with no line."""
        batch = self.batches[batch_id]
        responder = responder or self.responder
        output, failed = [], []
        for line in batch["_lines"]:
            custom_id = line["custom_id"]
            if custom_id in drop:
                continue
            if errors and custom_id in errors:
                code_status, code, message = errors[custom_id]
                failed.append({"id": f"batch_req_{custom_id}", "custom_id": custom_id,
                               "response": {"status_code": code_status, "request_id": "req_fake", "body": {"error": {"code": code}}},
                               "error": {"code": code, "message": message}})
                continue
            output.append({"id": f"batch_req_{custom_id}", "custom_id": custom_id, "error": None,
                           "response": {"status_code": 200, "request_id": "req_fake", "body": responder(line["body"])}})
            if custom_id in duplicate:
                output.append(json.loads(json.dumps(output[-1])))
        for custom_id in unknown:
            output.append({"id": "batch_req_unknown", "custom_id": custom_id, "error": None,
                           "response": {"status_code": 200, "request_id": "req_fake",
                                        "body": responder(batch["_lines"][0]["body"])}})
        if shuffle:
            output.reverse()
        batch["status"] = status
        batch["errors"], batch["usage"] = batch_errors, usage
        batch["request_counts"] = {"total": len(batch["_lines"]), "completed": len(output), "failed": len(failed)}
        if output:
            batch["output_file_id"] = self._store("\n".join(json.dumps(o) for o in output) + "\n")
        if failed:
            batch["error_file_id"] = self._store("\n".join(json.dumps(o) for o in failed) + "\n")

    def _store(self, text):
        file_id = self._next("file")
        self.files[file_id] = text.encode("utf-8")
        return file_id
