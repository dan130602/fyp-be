"""The Gemini Flex synchronous lifecycle shared by G5 and G7: a narrow mixin over `StageRunner`, not a provider framework.

It holds exactly what both Gemini stages need and what the OpenAI Batch stages do not: a sync Interactions transport with no
SDK retries, application-owned capacity (429/503) retries, explicit handling of ambiguous outcomes, fatal tier/model echo
mismatches, recovery BEFORE eligibility in the single mutating entry point `run()`, and usage/cost reporting. Everything
stage-specific (inputs, request, validation, compilation, history, status) stays in the stage's own runner. The bodies here
were moved verbatim from the G5 runner, so G5 behaviour and request bytes are unchanged.

A stage sets `flex` (its `FlexSettings`) and implements `_runnable(call_id)` (does the call have the input to be verified).
"""

import random
import time
from dataclasses import dataclass

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.llm import provenance
from fyp_evaluation.llm.errors import AmbiguousError, CapacityError, TransportError
from fyp_evaluation.llm.stage_runner import RunError

_TOOL_STEP_WORDS = ("call", "tool", "function")


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
        if self._awaiting_acknowledgement(latest):
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

    # ---- recovery and the single mutating entry point ------------------------------------------------
    def _recover(self):
        """Under the stage lock: settle every interrupted synchronous attempt, then adjudicate stored responses.

        The stage lock is held across each provider request, so a PREPARED attempt seen here belongs to a run that died.
        With a stored raw response it is re-adjudicated (never resent); without one its outcome is unknown, so it becomes an
        ambiguous TRANSPORT_FAILED that blocks the call until explicitly acknowledged. A stored fatal response raises here.
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
