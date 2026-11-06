"""The Gemini Flex synchronous lifecycle shared by G5 and G7: a narrow mixin over `StageRunner`, not a provider framework.

It holds exactly what both Gemini stages need and what the OpenAI Batch stages do not: a sync Interactions transport with no
SDK retries, application-owned capacity (429/503) retries, explicit handling of ambiguous outcomes, fatal tier/model echo
mismatches, recovery BEFORE eligibility in the single mutating entry point `run()`, and usage/cost reporting. Everything
stage-specific (inputs, request, validation, compilation, history, status) stays in the stage's own runner. The bodies here
were moved verbatim from the G5 runner, so G5 behaviour and request bytes are unchanged.

A stage sets `flex` (its `FlexSettings`) and implements `_runnable(call_id)` (does the call have the input to be verified).
"""

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.llm import provenance
from fyp_evaluation.llm.canonical import canonical_response
from fyp_evaluation.llm.errors import AmbiguousError, CapacityError, TransportError
from fyp_evaluation.llm.review_policy import CODEX_REVIEW_FORMAT, CODEX_REVIEWER, codex_review_policy
from fyp_evaluation.llm.stage_runner import RunError
from fyp_evaluation.llm.store import canonical_request_text
from fyp_evaluation.validation.stage0 import is_multilingual

_TOOL_STEP_WORDS = ("call", "tool", "function")

# Manual review: a reviewer reads the stage's own brief and writes the stage's own response; Python then
# validates, derives the verdict and accepts it exactly as for a provider response. The reviewer is recorded as itself, never as the
# provider the stage normally uses, and nothing is sent anywhere.
REVIEW_FORMAT = "claude-review/1"
REVIEW_FORMATS = {"claude-code-harness": REVIEW_FORMAT, CODEX_REVIEWER: CODEX_REVIEW_FORMAT}
REVIEWER_KINDS = tuple(REVIEW_FORMATS)
REVIEW_KEYS = {"format", "stage", "call_id", "reviewer", "request_sha256", "response"}


def load_review(path, stage, call_id):
    """The parsed review file, or a RunError naming what is wrong with it. Never reads or writes anything else."""
    try:
        raw = Path(path).read_bytes()
        review = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise RunError(f"cannot read the review file {path}: {error}") from error
    keys = REVIEW_KEYS | ({"review_policy_sha256"} if isinstance(review, dict) and review.get("format") == CODEX_REVIEW_FORMAT else set())
    if not isinstance(review, dict) or set(review) != keys:
        raise RunError(f"a review file needs exactly the keys {sorted(keys)}")
    if review["format"] not in REVIEW_FORMATS.values():
        raise RunError(f"unsupported review format {review['format']!r}; expected one of {tuple(REVIEW_FORMATS.values())}")
    if review["stage"] != stage:
        raise RunError(f"this review is for stage {review['stage']!r}, not {stage}")
    if review["call_id"] != call_id:
        raise RunError(f"this review is for {review['call_id']!r}, not {call_id}")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, dict) or set(reviewer) - {"kind", "model", "note"} or {"kind", "model"} - set(reviewer):
        raise RunError("reviewer needs kind and model (and optionally note)")
    if reviewer["kind"] not in REVIEWER_KINDS:
        raise RunError(f"reviewer.kind must be one of {REVIEWER_KINDS}")
    if REVIEW_FORMATS[reviewer["kind"]] != review["format"]:
        raise RunError("reviewer.kind does not match the review format")
    if not isinstance(reviewer["model"], str) or not reviewer["model"].strip() or "gemini" in reviewer["model"].lower():
        raise RunError("reviewer.model must name the reviewing model and must not claim to be the provider verifier")
    if not isinstance(review["request_sha256"], str) or len(review["request_sha256"]) != 64:
        raise RunError("request_sha256 must be the 64-hex hash of the brief the reviewer read (from --prepare-review)")
    if not isinstance(review["response"], dict):
        raise RunError("response must be the stage's JSON response object")
    review["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return review


@dataclass(frozen=True)
class FlexSettings:
    """Version-controlled provider settings of one Gemini stage."""

    model: str
    service_tier: str
    thinking_level: str
    max_output_tokens: int
    api_key_variable: str
    timeout_ms: int
    capacity_sends: int
    capacity_backoff_seconds: tuple
    capacity_jitter: float
    prices: dict  # {model: {tier: {"input", "cached_input", "output"}}} USD per million tokens
    price_table_version: str


def parse_interaction(body, model, tier):
    """Interpret one Interactions body: ("ok", parsed, None) | ("rejected"|"transport"|"fatal", None, reason)."""
    served = body.get("model")
    if served is not None and served != model:
        return "fatal", None, f"MODEL_MISMATCH: the provider served {served!r} but {model} was requested"
    reported = body.get("service_tier")
    if reported is not None and str(reported).lower() != tier:
        return "fatal", None, (f"TIER_MISMATCH: the provider reported service tier {reported!r} but {tier} was requested; "
                               "there is no fallback to another tier, so the run stops")
    status = body.get("status")
    if status in ("incomplete", "budget_exceeded"):
        return "rejected", None, f"truncated response (status {status})"
    if status == "requires_action":
        return "rejected", None, "the model asked for a tool action; no tools are sent"
    if status in ("failed", "cancelled"):
        errors = body.get("errors") or []
        detail = (errors[0] or {}).get("message") if errors and isinstance(errors[0], dict) else None
        return "transport", None, f"interaction status {status}" + (f" ({str(detail)[:200]})" if detail else "")
    if status != "completed":
        return "rejected", None, f"unexpected interaction status {status!r}"
    steps = body.get("steps") or []
    for step in steps:
        kind = str((step or {}).get("type", ""))
        if any(word in kind for word in _TOOL_STEP_WORDS):
            return "rejected", None, f"unexpected tool step {kind!r}"
    text = body.get("output_text")
    if not text:
        text = "".join(part.get("text") or "" for step in steps if (step or {}).get("type") == "model_output"
                       for part in step.get("content") or [] if part.get("type") == "text")
    if not text:
        return "rejected", None, "response contains no output text"
    try:
        parsed = parse_json_text(text)
    except ValueError as error:
        return "rejected", None, f"unparseable structured output: {error}"
    if not isinstance(parsed, dict):
        return "rejected", None, "structured output is not a JSON object"
    return "ok", parsed, None


def usage_summary(usage):
    usage = usage or {}
    return {"input_tokens": int(usage.get("total_input_tokens") or 0), "cached_tokens": int(usage.get("total_cached_tokens") or 0),
            "output_tokens": int(usage.get("total_output_tokens") or 0), "thought_tokens": int(usage.get("total_thought_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0)}


def estimate_cost_with(usage, prices):
    """Estimated USD (thinking tokens billed as output), or None when there is no price for the model/tier."""
    if prices is None:
        return None
    uncached = max(usage["input_tokens"] - usage["cached_tokens"], 0)
    cost = uncached * prices["input"] + usage["cached_tokens"] * prices["cached_input"]
    cost += (usage["output_tokens"] + usage.get("thought_tokens", 0)) * prices["output"]
    return round(cost / 1_000_000, 6)


def describe(attempt):
    """One status line for an attempt (CLI)."""
    line = f"{attempt['call_id']} {attempt['attempt_id']} {attempt['state']}"
    if attempt.get("rejection_reason"):
        line += f": {attempt['rejection_reason']}"
    if attempt.get("estimated_cost_usd") is not None:
        line += f" cost~${attempt['estimated_cost_usd']}"
    return line


class GeminiFlexMixin:
    """Put before `StageRunner` in the bases: `class Runner(GeminiFlexMixin, StageRunner)`."""

    flex = None  # the stage's FlexSettings
    sleep = staticmethod(time.sleep)  # replaced in tests
    jitter = random.Random()
    _manual = False  # True only inside prepare_review / ingest_review

    def _runnable(self, call_id):
        """True when the call has the input this stage verifies (restricts the freshness check and selection)."""
        raise NotImplementedError

    # ---- provider hooks ------------------------------------------------------------------------------
    def _default_transport_factory(self):
        from fyp_evaluation.llm.gemini_transport import GeminiTransport

        settings = self.flex
        return lambda: GeminiTransport.from_env(variable=settings.api_key_variable, timeout_ms=settings.timeout_ms)

    def _attempt_metadata(self):
        settings = self.flex
        return {"params": {"thinking_level": settings.thinking_level, "service_tier": settings.service_tier,
                           "max_output_tokens": settings.max_output_tokens},
                "price_table_version": settings.price_table_version}

    def _parse_response(self, body):
        return parse_interaction(body, self.flex.model, self.flex.service_tier)

    def _estimate(self, usage):
        return estimate_cost_with(usage, self.flex.prices.get(self.flex.model, {}).get(self.flex.service_tier))

    def _usage_and_cost(self, attempt, body):
        attempt["observed_model"] = body.get("model")
        usage = usage_summary(body.get("usage"))
        return usage, self._estimate(usage)

    def _send(self, attempt, transport, body):
        """Send identical request bytes; resend only after a definite capacity rejection, with bounded backoff."""
        settings = self.flex
        for number in range(1, settings.capacity_sends + 1):
            attempt["send_count"] = number  # provider sends, counted separately from this one logical attempt; saved BEFORE the send
            self._save(attempt)
            try:
                return transport.create_response(body)
            except CapacityError as error:
                attempt.setdefault("sends", []).append({"at": self.clock(), "outcome": f"capacity rejection: {str(error)[:200]}"})
                self._save(attempt)
                if number == settings.capacity_sends:
                    raise
                delay = settings.capacity_backoff_seconds[min(number, len(settings.capacity_backoff_seconds)) - 1]
                self.sleep(delay * (1 + self.jitter.uniform(-settings.capacity_jitter, settings.capacity_jitter)))
        raise AssertionError("unreachable")

    # ---- eligibility, ambiguity ------------------------------------------------------------------------
    @staticmethod
    def _awaiting_acknowledgement(latest):
        return bool(latest and latest["state"] == "TRANSPORT_FAILED" and latest.get("ambiguous")
                    and not latest.get("ambiguous_acknowledged_at"))

    def eligibility(self, call_id, *, retry_rejected=False, reopen=False, blocked=None):
        latest = self.call_state(call_id)["latest"]
        # an unacknowledged ambiguous PROVIDER attempt blocks another provider request; a review sends nothing, so it does not
        if not self._manual and self._awaiting_acknowledgement(latest):
            return False, (f"AWAITING_ACKNOWLEDGEMENT: the outcome of {latest['attempt_id']} is unknown and it may have been billed; "
                           "rerun with --acknowledge-possible-duplicate --call-id " + call_id)
        return super().eligibility(call_id, retry_rejected=retry_rejected, reopen=reopen, blocked=blocked)

    def _acknowledge(self, call_id):
        latest = self.call_state(call_id)["latest"]
        if not self._awaiting_acknowledgement(latest):
            raise RunError(f"{call_id}: there is no unacknowledged ambiguous attempt to acknowledge")
        attempt_id = latest["attempt_id"]
        attempt = self.store.read_attempt(call_id, attempt_id) if attempt_id in self.store.attempt_ids(call_id) else dict(latest)
        attempt["ambiguous_acknowledged_at"] = self.clock()
        if attempt_id in self.store.attempt_ids(call_id):
            self._save(attempt)
        record = provenance.read(self.spec, self.root, self.inputs.dataset_id, call_id)
        provenance.upsert_attempt(record, attempt)
        provenance.write(self.spec, self.root, record)
        return attempt_id

    def _observed_tier_missing(self, call_id):
        prov, problem = self._prov_state(call_id)
        accepted = prov["accepted"] if prov and not problem else None
        if not accepted:
            return False
        summary = next((a for a in prov["attempts"] if a["attempt_id"] == accepted["attempt_id"]), {})
        return summary.get("service_tier") is None

    def usage_report(self):
        """Observational usage and cost from the recorded attempts (committed summaries overlaid by the audit records). Read-only.

        Costs are estimates from the usage the provider reported and the dated price table, not a provider balance; an attempt that
        may have been billed but reported no usage (ambiguous or interrupted) is counted in `unpriced_possibly_billed` instead.
        Nothing here limits or stops a run.
        """
        totals = {"attempts": 0, "provider_sends": 0, "capacity_retries": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                  "thought_tokens": 0, "estimated_cost_usd": 0.0, "unpriced_possibly_billed": 0}
        calls = {}
        for call_id in self.inputs.call_ids:
            row = {"attempts": 0, "provider_sends": 0, "capacity_retries": 0, "estimated_cost_usd": 0.0}
            for attempt in self._attempts(call_id).values():
                usage = attempt.get("usage") or {}
                row["attempts"] += 1
                row["provider_sends"] += int(attempt.get("send_count") or 0)
                row["capacity_retries"] += len(attempt.get("sends") or [])
                row["estimated_cost_usd"] += attempt.get("estimated_cost_usd") or 0.0
                for key in ("input_tokens", "cached_tokens", "output_tokens", "thought_tokens"):
                    totals[key] += int(usage.get(key) or 0)
                if attempt.get("possibly_billed") and attempt.get("estimated_cost_usd") is None:
                    totals["unpriced_possibly_billed"] += 1
            row["estimated_cost_usd"] = round(row["estimated_cost_usd"], 6)
            for key in ("attempts", "provider_sends", "capacity_retries"):
                totals[key] += row[key]
            totals["estimated_cost_usd"] += row["estimated_cost_usd"]
            calls[call_id] = row
        totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 6)
        return {"calls": calls, "totals": totals}

    # ---- Claude-assisted review ------------------------------------------------------------------------
    def _exposure(self):
        if self._manual:  # not an exposure to a provider; recorded so the attempt is never mistaken for one
            from fyp_evaluation.llm import live_guard

            return {"run_kind": "manual", **live_guard.code_state(), "allowlist_batch": None}
        return super()._exposure()

    def _require_multilingual(self):
        if not is_multilingual(self.inputs.documents):
            raise RunError("Manual review is only available for multilingual (schema 2.0) releases")

    def prepare_review(self, call_id, *, retry_rejected=False, reopen=False, reviewer_kind="claude-code-harness"):
        """The brief a reviewer must read and the hashes the review will be bound to. Read-only: nothing is written or sent."""
        self._require_multilingual()
        if reviewer_kind not in REVIEWER_KINDS:
            raise RunError(f"reviewer.kind must be one of {REVIEWER_KINDS}")
        self._manual = True
        try:
            eligible, reason = self.eligibility(call_id, retry_rejected=retry_rejected, reopen=reopen)
            if not eligible:
                raise RunError(f"{call_id}: {reason}")
            body = self.builder.build(call_id, "A000")  # G5/G7 request bodies carry no attempt id
            packet = {"format": REVIEW_FORMATS[reviewer_kind], "stage": self.spec.stage, "call_id": call_id,
                    "request_sha256": text_hash(canonical_request_text(body)), "upstream_sha256": self._upstream_hashes(call_id),
                    "request": body}
            if reviewer_kind == CODEX_REVIEWER:
                packet["review_policy"] = codex_review_policy()
            return packet
        finally:
            self._manual = False

    def ingest_review(self, call_id, review_path, *, reopen_reason=None, retry_rejected=False):
        """Accept a review through the same recovery, freshness, eligibility, validation and acceptance as a provider response.

        The review must have been written against exactly the brief the stage would build NOW (its request hash), so a change to
        any input between preparation and acceptance is refused. An accepted call needs --reopen, exactly as for a provider run.
        """
        self._require_multilingual()
        review = load_review(review_path, self.spec.stage, call_id)
        reopen_ids = {call_id} if reopen_reason else set()
        with self.store.lock():
            self._manual = True
            try:
                self._recover()
                if self._runnable(call_id):
                    self._ensure_fresh_inputs([call_id])
                self._select([call_id], retry_rejected=retry_rejected, reopen_call_ids=reopen_ids)
                actual = text_hash(canonical_request_text(self.builder.build(call_id, "A000")))
                if actual != review["request_sha256"]:
                    raise RunError(f"{call_id}: the review was written against a different brief (request {review['request_sha256'][:12]}..., "
                                   f"now {actual[:12]}...); inputs or instructions changed, so it is refused")
                policy = codex_review_policy() if review["reviewer"]["kind"] == CODEX_REVIEWER else None
                if policy is not None and review["review_policy_sha256"] != policy["sha256"]:
                    raise RunError(f"{call_id}: review policy changed or its hash is missing/incorrect; prepare and review again")
                previous = self.call_state(call_id)
                reopen_of = None
                if reopen_reason:
                    reopen_of = next((a["attempt_id"] for a in reversed(previous["attempts"]) if a["state"] == "ACCEPTED"), None)
                attempt, _ = self._new_attempt(call_id, "manual", reopen_reason=reopen_reason, reopen_of=reopen_of)
                reviewer = review["reviewer"]
                attempt["model"] = f"{reviewer['kind']}:{reviewer['model']}"
                attempt["params"] = {"method": "manual_review", "reviewer": reviewer["kind"], "reviewer_model": reviewer["model"],
                                     "request_sha256": actual}
                if policy is not None:
                    attempt["params"].update(review_format=CODEX_REVIEW_FORMAT, review_policy_version=policy["version"],
                                             review_policy_sha256=policy["sha256"])
                self._save(attempt)
                # keys in the response schema's order, so the artifact compiled now is byte-identical to its later rebuild from the stored response
                ordered = canonical_response(review["response"], self.builder.response_schema)
                body = {"id": f"manual_review_{attempt['attempt_id']}", "status": "completed",
                        "output_text": json.dumps(ordered, ensure_ascii=False),
                        "manual_review": {"reviewer": reviewer, "request_sha256": actual, "review_file_sha256": review["_file_sha256"]}}
                if policy is not None:
                    body["manual_review"]["review_policy"] = policy
                self._store_response(attempt, body, {"transport": "manual", "reviewer": reviewer})
                return self._adjudicate(attempt)
            finally:
                self._manual = False

    # ---- recovery and the single mutating entry point ------------------------------------------------
    def _recover(self):
        """Under the stage lock: settle every interrupted synchronous attempt, then adjudicate stored responses.

        The stage lock is held across each provider request, so a PREPARED attempt seen here belongs to a run that died.
        A stored raw response is re-adjudicated (never resent). Without one, a provider request is ambiguous and possibly
        billed; a manual import sent nothing and closes unbilled. A stored fatal response raises here.
        """
        settled = []
        for call_id in self.inputs.call_ids:
            for attempt_id in self.store.attempt_ids(call_id):
                attempt = self.store.read_attempt(call_id, attempt_id)
                if attempt["state"] != "PREPARED":
                    continue
                raw = self.store.read_raw_response(call_id, attempt_id)
                if raw is not None:
                    self._store_response(attempt, raw["response"], raw.get("meta") or {"transport": "sync"})
                elif attempt["transport"] == "manual":
                    attempt["estimated_cost_usd"] = 0.0
                    self._close(attempt, "TRANSPORT_FAILED", possibly_billed=False,
                                reason="interrupted local review ingestion before its response was stored; no provider request was sent")
                    settled.append(attempt)
                else:
                    attempt["ambiguous"] = True
                    self._close(attempt, "TRANSPORT_FAILED", possibly_billed=True,
                                reason="interrupted during a synchronous request; the outcome is unknown and it may have been billed")
                    settled.append(attempt)
        return settled + super()._recover()

    def run(self, call_ids=None, *, limit=None, retry_rejected=False, reopen_reason=None, acknowledge_duplicate=False):
        """Recover interrupted state, then verify eligible calls one by one. Returns a report dict.

        Order (all under the stage lock): recovery -> optional acknowledgement -> freshness -> eligibility -> credentials -> sends.
        A stored or live fatal response stops the run before any further request. Recovery never needs credentials.
        """
        settings = self.flex
        explicit = call_ids is not None
        if explicit and (unknown := [c for c in call_ids if c not in self.inputs.call_ids]):
            raise RunError(f"unknown call id(s): {', '.join(unknown)}")
        if acknowledge_duplicate and not (explicit and len(call_ids) == 1):
            raise RunError("--acknowledge-possible-duplicate needs exactly one --call-id")
        if reopen_reason and not explicit:
            raise RunError("--reopen needs --call-id")
        report = {"recovered": [], "acknowledged": None, "attempts": [], "skipped": {}, "stopped": None}
        with self.store.lock():
            recovered = self._recover()
            report["recovered"] = [(a["call_id"], a["attempt_id"], a["state"]) for a in recovered]
            if acknowledge_duplicate:
                report["acknowledged"] = self._acknowledge(call_ids[0])
            targets = call_ids if explicit else list(self.inputs.call_ids)
            self._ensure_fresh_inputs([c for c in targets if self._runnable(c)])
            recovered_calls = {a["call_id"] for a in recovered}
            reopen_ids = set(call_ids) if reopen_reason else set()
            selectable = [c for c in call_ids if c not in recovered_calls] if explicit else None
            chosen, skipped = self._select(selectable, retry_rejected=retry_rejected, reopen_call_ids=reopen_ids)
            report["skipped"] = skipped
            if limit is not None:
                chosen = chosen[:limit]
            if not chosen:
                return report
            transport = self.transport()  # CredentialsError here: recovery is done, nothing was sent
            self._paid_guard(chosen, transport)  # before any attempt is written
            for call_id in chosen:
                previous = self.call_state(call_id)
                reopen_of = None
                if call_id in reopen_ids:
                    reopen_of = next((a["attempt_id"] for a in reversed(previous["attempts"]) if a["state"] == "ACCEPTED"), None)
                attempt, body = self._new_attempt(call_id, "sync", reopen_reason=reopen_reason if call_id in reopen_ids else None,
                                                  reopen_of=reopen_of)
                try:
                    response = self._send(attempt, transport, body)
                except CapacityError as error:
                    self._close(attempt, "TRANSPORT_FAILED", reason=f"capacity unavailable after {settings.capacity_sends} sends: {error}")
                    report["attempts"].append(attempt)
                    report["stopped"] = "Flex capacity is unavailable; the run stopped (rerun later)"
                    break
                except AmbiguousError as error:
                    attempt["ambiguous"] = True
                    self._close(attempt, "TRANSPORT_FAILED", reason=f"outcome unknown: {error}", possibly_billed=True)
                    report["attempts"].append(attempt)
                    report["stopped"] = (f"{call_id} {attempt['attempt_id']}: the outcome is unknown and it may have been billed; the run "
                                         "stopped. Rerun with --acknowledge-possible-duplicate --call-id " + call_id + " to try again")
                    break
                except TransportError as error:
                    self._fail_transport(attempt, error, possibly_billed=not error.fatal)
                    report["attempts"].append(attempt)
                    if error.fatal:
                        raise
                    report["stopped"] = f"{call_id}: {error}"
                    break
                self._store_response(attempt, response, {"transport": "sync"})
                report["attempts"].append(self._adjudicate(attempt))  # a fatal echo raises and stops the run here
        return report
