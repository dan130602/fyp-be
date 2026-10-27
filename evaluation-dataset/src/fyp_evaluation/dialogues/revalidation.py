"""Offline reuse of a completed response rejected solely by the old length gate.

The original request, response, rejection and cost stay intact. A new, zero-cost local validation attempt links to them;
it is not a provider request. Every non-length rule and current upstream/G5 eligibility check still applies.
"""

from copy import deepcopy

from fyp_evaluation.call_specs.compile import serialize_json, text_hash
from fyp_evaluation.llm import live_guard, provenance
from fyp_evaluation.llm.stage_runner import RunError
from fyp_evaluation.llm.store import attempt_label, dataset_write_lock

from .config import length_is_advisory
from .request import DialogueContext


def source_response(runner, call_id, source_id):
    source = runner.store.read_attempt(call_id, source_id)
    record, problem = runner._prov_state(call_id)
    if problem or provenance.summary(source) not in record["attempts"]:
        raise RunError("the source attempt does not match intact committed provenance")
    if source["state"] != "REJECTED" or source.get("possibly_billed") or source.get("ambiguous") or source.get("fatal"):
        raise RunError("reuse requires an unambiguous, nonfatal REJECTED source")
    recorded_errors = source.get("validation_errors", [])
    if not recorded_errors or any(not error.startswith("length: ") for error in recorded_errors):
        raise RunError("the source was not rejected solely for length")
    request = runner.store.read_request(call_id, source_id)
    if request is None or text_hash(request) != source["request_sha256"]:
        raise RunError("the source request hash does not match")
    raw = runner.store.read_raw_response(call_id, source_id)
    if not raw or raw["response"].get("status") != "completed" or source.get("api_status") != "completed":
        raise RunError("the source response is missing or did not complete")
    if raw["response"].get("id") != source.get("provider_response_id"):
        raise RunError("the source provider response ID does not match")
    kind, parsed, _ = runner._parse_response(raw["response"])
    if kind != "ok":
        raise RunError("the source response cannot be parsed as a complete dialogue")
    if source["upstream_sha256"] != runner._current_upstream_hashes(call_id):
        raise RunError("the source upstream is stale; it cannot be rebound")
    if source["response_schema_sha256"] != runner.builder.response_schema_sha256:
        raise RunError("the source response schema differs from the current schema")
    if not any(text_hash(p.read_text(encoding="utf-8")) == source["prompt_template_sha256"]
               for p in runner.prompt_dir.glob("dialogue.v*.md")):
        raise RunError("the original generation prompt is unavailable")
    ctx = DialogueContext(runner.inputs, call_id, source["params"]["length_policy"],
                          language_order=source["params"].get("placement_rules") == runner._attempt_metadata(call_id)["params"]["placement_rules"])
    if runner.validator.errors(parsed, ctx) != recorded_errors:
        raise RunError("source validation no longer reproduces its length-only rejection")
    return source, request, raw, parsed


def verify_replay(runner, attempt):
    """Recheck the saved source when an interrupted local validation is resumed; never call a provider."""
    link = attempt["length_revalidation"]
    source, request, raw, _ = source_response(runner, attempt["call_id"], link["source_attempt_id"])
    if (text_hash(serialize_json(raw)) != link["source_response_sha256"]
            or source["request_sha256"] != attempt["request_sha256"]
            or attempt["prompt_template_sha256"] != source["prompt_template_sha256"]
            or runner.store.read_request(attempt["call_id"], attempt["attempt_id"]) != request
            or runner.store.read_raw_response(attempt["call_id"], attempt["attempt_id"]) != raw):
        raise RunError("the locally reused source no longer matches its recorded hashes")


def revalidate_length_only(runner, call_id, source_id, reason):
    if not reason or not reason.strip():
        raise RunError("a reason is required for length-only revalidation")
    with runner.store.lock(), dataset_write_lock(runner.root, wait_seconds=runner.write_lock_wait):
        runner._ensure_fresh_inputs([call_id])
        if not length_is_advisory(runner.builder.context(call_id).length_policy):
            raise RunError("length-only reuse requires the advisory policy")
        state = runner.call_state(call_id)
        if (state["accepted"] or state["inconsistent"] or state["in_flight"] or call_id in runner._blocked_calls()
                or not state["latest"] or state["latest"]["attempt_id"] != source_id):
            raise RunError("reuse requires the latest rejected attempt, no accepted output and no pending work")
        if gate := runner._g5_gate(call_id):
            raise RunError(gate)
        source, request, raw, parsed = source_response(runner, call_id, source_id)
        errors = runner.validator.errors(parsed, runner.builder.context(call_id))
        if errors:
            raise RunError("the response fails current non-length validation: " + "; ".join(errors))
        number = state["next_number"]
        attempt_id = attempt_label(number)
        code = live_guard.code_state()
        attempt = {**deepcopy(source), "attempt_id": attempt_id, "attempt_number": number,
                   "custom_id": runner.spec.custom_id(call_id, attempt_id), "transport": "revalidation", "state": "RESPONDED",
                   "submission_id": None, "batch_id": None, "provider_response_id": None,
                   "params": deepcopy(runner._attempt_metadata(call_id)["params"]),
                   "usage": None, "estimated_cost_usd": 0.0, "service_tier": None,
                   "validation_errors": [], "rejection_reason": None, "created_at": runner.clock(), "finished_at": "",
                   "exposure": {"run_kind": "offline", "code_commit": code["code_commit"],
                                "code_dirty_paths": code["code_dirty_paths"]},
                   "length_revalidation": {"source_attempt_id": source_id, "source_request_sha256": source["request_sha256"],
                                           "source_response_sha256": text_hash(serialize_json(raw)), "reason": reason.strip()}}
        # request.json is the original, unsent-again request. The source prompt hash remains the actual generation prompt.
        runner.store.write_request(call_id, attempt_id, request)
        runner.store.write_raw_response(call_id, attempt_id, raw)
        runner.store.write_attempt(attempt)
        return runner._judge_locked(attempt, parsed)
