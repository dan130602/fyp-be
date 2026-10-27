"""Offline reinstatement of an earlier accepted G6 dialogue that a later reopen superseded.

Nothing is sent to a provider and no history is rewritten. The source must appear in the committed `accepted_history`, its stored
request and raw response must still match their recorded hashes, and the raw response must parse to exactly the accepted parsed
response. A new, zero-cost local attempt copies that request and response, links to the source, and supersedes the currently
accepted attempt through the normal reopen path. Current upstream, G5 eligibility and every G6 validation rule still apply; the
original attempts, responses, costs and exposure records stay exactly as they were.
"""

from copy import deepcopy

from fyp_evaluation.call_specs.compile import serialize_json, text_hash
from fyp_evaluation.llm import live_guard, provenance
from fyp_evaluation.llm.canonical import response_hash
from fyp_evaluation.llm.stage_runner import RunError
from fyp_evaluation.llm.store import attempt_label, dataset_write_lock

# Links and per-attempt state that belong to the source attempt only and must not be carried into the reinstated attempt.
SOURCE_ONLY_KEYS = ("length_revalidation", "reinstatement", "reopen", "reopen_reason", "stale_upstream", "blocked", "ambiguous",
                    "last_error_detail")


def source_entry(runner, call_id, source_id):
    """(entry, source attempt, request, raw, parsed) for a superseded accepted attempt, or RunError naming what does not hold."""
    record, problem = runner._prov_state(call_id)
    if problem or record is None:
        raise RunError(f"provenance for {call_id} is not intact: {problem or 'missing'}")
    if record["accepted"] and record["accepted"]["attempt_id"] == source_id:
        raise RunError(f"{source_id} is already the accepted dialogue of {call_id}")
    entries = [entry for entry in record["accepted_history"] if entry["attempt_id"] == source_id]
    if not entries:
        raise RunError(f"{source_id} was never accepted and superseded for {call_id}; only an earlier accepted dialogue can be reinstated")
    entry = entries[-1]
    schema = runner.builder.response_schema
    if response_hash(entry["parsed_response"], schema) != entry["parsed_response_sha256"]:
        raise RunError("the recorded accepted response does not match its recorded hash")
    committed = next((a for a in record["attempts"] if a["attempt_id"] == source_id), None)
    if committed is None or committed["state"] != "SUPERSEDED":
        raise RunError(f"{source_id} is not recorded as a SUPERSEDED attempt")
    if source_id not in runner.store.attempt_ids(call_id):
        raise RunError("the source attempt is missing from the audit store")
    source = runner.store.read_attempt(call_id, source_id)  # the full record; committed provenance keeps only a summary of it
    if provenance.summary(source) != committed:
        raise RunError("the source attempt in the audit store does not match intact committed provenance")
    request = runner.store.read_request(call_id, source_id)
    if request is None or text_hash(request) != source["request_sha256"]:
        raise RunError("the source request is missing from the audit store or does not match its recorded hash")
    raw = runner.store.read_raw_response(call_id, source_id)
    if not raw:
        raise RunError("the source raw response is missing from the audit store")
    kind, parsed, _ = runner._parse_response(raw["response"])
    if kind != "ok" or response_hash(parsed, schema) != entry["parsed_response_sha256"]:
        raise RunError("the stored raw response does not parse to the recorded accepted response")
    if source["upstream_sha256"] != runner._current_upstream_hashes(call_id):
        raise RunError("the source upstream is stale; it cannot be rebound")
    if source["response_schema_sha256"] != runner.builder.response_schema_sha256:
        raise RunError("the source response schema differs from the current schema")
    if not any(text_hash(path.read_text(encoding="utf-8")) == source["prompt_template_sha256"]
               for path in runner.prompt_dir.glob("dialogue.v*.md")):
        raise RunError("the original generation prompt is unavailable")
    return entry, source, request, raw, parsed


def verify_replay(runner, attempt):
    """Recheck the linked source when an interrupted reinstatement is resumed; never call a provider."""
    link = attempt["reinstatement"]
    _, source, request, raw, _ = source_entry(runner, attempt["call_id"], link["source_attempt_id"])
    if (text_hash(serialize_json(raw)) != link["source_response_sha256"]
            or source["request_sha256"] != attempt["request_sha256"]
            or attempt["prompt_template_sha256"] != source["prompt_template_sha256"]
            or runner.store.read_request(attempt["call_id"], attempt["attempt_id"]) != request
            or runner.store.read_raw_response(attempt["call_id"], attempt["attempt_id"]) != raw):
        raise RunError("the reinstated source no longer matches its recorded hashes")


def reinstate_accepted(runner, call_id, source_id, reason):
    if not reason or not reason.strip():
        raise RunError("a reason is required for reinstatement")
    with runner.store.lock(), dataset_write_lock(runner.root, wait_seconds=runner.write_lock_wait):
        runner._ensure_fresh_inputs([call_id])
        state = runner.call_state(call_id)
        if (not state["accepted"] or state["inconsistent"] or state["in_flight"] or call_id in runner._blocked_calls()
                or runner._awaiting_acknowledgement(state["latest"])):
            raise RunError("reinstatement requires an accepted dialogue, intact provenance and no pending or unresolved attempt")
        if gate := runner._g5_gate(call_id):
            raise RunError(gate)
        entry, source, request, raw, parsed = source_entry(runner, call_id, source_id)
        errors = runner.validator.errors(parsed, runner.builder.context(call_id))
        if errors:
            raise RunError("the earlier dialogue fails current validation: " + "; ".join(errors))
        record, _ = runner._prov_state(call_id)
        current_id = record["accepted"]["attempt_id"]
        number = state["next_number"]
        attempt_id = attempt_label(number)
        code = live_guard.code_state()
        attempt = deepcopy(source)
        for key in SOURCE_ONLY_KEYS:
            attempt.pop(key, None)
        attempt.update({
            "attempt_id": attempt_id, "attempt_number": number, "custom_id": runner.spec.custom_id(call_id, attempt_id),
            "transport": "reinstatement", "state": "RESPONDED", "submission_id": None, "batch_id": None, "provider_response_id": None,
            "params": deepcopy(runner._attempt_metadata(call_id)["params"]), "usage": None, "estimated_cost_usd": 0.0,
            "service_tier": None, "possibly_billed": False, "fatal": False, "validation_errors": [], "rejection_reason": None,
            "created_at": runner.clock(), "finished_at": "",
            "exposure": {"run_kind": "offline", "code_commit": code["code_commit"], "code_dirty_paths": code["code_dirty_paths"]},
            "reinstatement": {"source_attempt_id": source_id, "source_parsed_response_sha256": entry["parsed_response_sha256"],
                              "source_artifact_sha256": entry["artifact_sha256"], "source_request_sha256": source["request_sha256"],
                              "source_response_sha256": text_hash(serialize_json(raw)), "reason": reason.strip()},
            # supersede the current accepted dialogue through the normal reopen path; that attempt stays in history
            "reopen": {"reason": reason.strip(), "previous_attempt_id": current_id},
        })
        # request.json and raw_response.json are copies of the source's (not sent again); the prompt hash stays the source's
        runner.store.write_request(call_id, attempt_id, request)
        runner.store.write_raw_response(call_id, attempt_id, raw)
        runner.store.write_attempt(attempt)
        return runner._judge_locked(attempt, parsed)
