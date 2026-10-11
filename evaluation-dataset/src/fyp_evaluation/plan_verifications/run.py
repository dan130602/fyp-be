"""G5 library API: `PlanVerificationRunner` is the G5 specialisation of the shared `StageRunner` lifecycle.

It reuses attempts, provenance, accepted history, the freshness check under `.stage_write.lock`, dependency status,
`--validate` and `--check`. G5-specific: a Gemini Flex transport (no Batch), application-owned capacity retries, explicit
handling of ambiguous outcomes, fatal tier/model echo mismatches, recovery BEFORE eligibility in its single mutating
entry point `run()`, and reports compiled against the exact G3/G4 versions they verified.
"""

import random
import time

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.check_plans.inputs import HistoricalContextError
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.llm import provenance
from fyp_evaluation.llm import stage_runner as shared
from fyp_evaluation.llm.errors import AmbiguousError, CapacityError, TransportError
from fyp_evaluation.llm.stage_runner import RunError, StageRunner, StageSpec

from .compile import build_record, compile_artifact
from .config import (
    API_KEY_VARIABLE, ARTIFACT_DIR, CAPACITY_BACKOFF_SECONDS, CAPACITY_JITTER, CAPACITY_SENDS, ESTIMATED_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS, MODEL, PRICE_TABLE_VERSION, PRICES, PROMPT_DIR, PROVENANCE_DIR, RECORD, SEMANTIC_BUDGET, SERVICE_TIER,
    STAGE, THINKING_LEVEL, TIMEOUT_MS, TRANSPORT_BUDGET,
)
from .inputs import fresh_upstream_hashes, inputs_with_versions, load_inputs
from .request import RequestBuilder, VerificationContext
from .validate_response import VerificationValidator

G5_SPEC = StageSpec(
    stage=STAGE, label="verification", noun="plan verification", artifact_dir=ARTIFACT_DIR, provenance_dir=PROVENANCE_DIR,
    record_path=RECORD, schema_subdir="stage5", artifact_schema="plan_verification", record_schema="plan_verification_record",
    provenance_schema="plan_verification_provenance", prompt_dir=PROMPT_DIR, model=MODEL, reasoning_effort=THINKING_LEVEL,
    max_output_tokens=MAX_OUTPUT_TOKENS, semantic_budget=SEMANTIC_BUDGET, transport_budget=TRANSPORT_BUDGET)

_TOOL_STEP_WORDS = ("call", "tool", "function")


def parse_interaction(body):
    """Interpret one Interactions body: ("ok", parsed) | ("rejected", reason) | ("transport", reason) | ("fatal", reason)."""
    model = body.get("model")
    if model is not None and model != MODEL:
        return "fatal", None, f"MODEL_MISMATCH: the provider served {model!r} but {MODEL} was requested"
    tier = body.get("service_tier")
    if tier is not None and str(tier).lower() != SERVICE_TIER:
        return "fatal", None, (f"TIER_MISMATCH: the provider reported service tier {tier!r} but {SERVICE_TIER} was requested; "
                               "there is no fallback to another tier, so the run stops")
    status = body.get("status")
    if status in ("incomplete", "budget_exceeded"):
        return "rejected", None, f"truncated response (status {status})"
    if status == "requires_action":
        return "rejected", None, "the model asked for a tool action; G5 sends no tools"
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


def estimate_cost(usage, model=MODEL, tier=SERVICE_TIER):
    """Estimated USD (thinking tokens billed as output), or None when the model/tier has no price."""
    prices = PRICES.get(model, {}).get(tier)
    if prices is None:
        return None
    uncached = max(usage["input_tokens"] - usage["cached_tokens"], 0)
    cost = uncached * prices["input"] + usage["cached_tokens"] * prices["cached_input"]
    cost += (usage["output_tokens"] + usage.get("thought_tokens", 0)) * prices["output"]
    return round(cost / 1_000_000, 6)


def synthesize_response(ctx):
    """A deterministic all-MATCH report that satisfies every structural rule for this call (offline feasibility/fixtures)."""
    checks = []
    for question_id in ctx.order:
        check, evidence = ctx.checks[question_id], []
        if check["evidence_mode"] == "PRESENCE":
            event_id = check.get("final_event_id") or check["evidence_event_ids"][0]
            evidence.append({"source": "G4", "ref": event_id, "quote": ctx.planned_text(event_id)[:60]})
        checks.append({"question_id": question_id, "evidence": evidence,
                       "rationale": "The planned content supports the expected outcome.", "assessment": "MATCH", "issue_ids": []})
    return {"checks": checks, "issues": []}


class PlanVerificationRunner(StageRunner):
    spec = G5_SPEC
    ARTIFACT_HASH_KEYS = {**StageRunner.ARTIFACT_HASH_KEYS, "check_plan": "check_plan_sha256"}
    GENERATION_KEYS = StageRunner.GENERATION_KEYS + ("params",)
    sleep = staticmethod(time.sleep)  # replaced in tests
    jitter = random.Random()

    # ---- stage hooks ---------------------------------------------------------------------------------
    def _load_inputs(self):
        return load_inputs(self.root, self.schema_dir)

    def _make_builder(self):
        return RequestBuilder(self.inputs, self.schema_dir, self.prompt_dir)

    def _make_validator(self):
        return VerificationValidator(self.builder.response_schema)

    def _feasibility(self):
        return {c: self.validator.errors(synthesize_response(self.builder.context(c)), self.builder.context(c))
                for c in self.inputs.call_ids if c in self.inputs.plans}

    def _compile_artifact(self, ctx, parsed, generation):
        return compile_artifact(self.inputs, ctx, parsed, generation, self.builder.response_schema)

    def _build_record(self, artifact_texts, provenance_by_call):
        return build_record(self.inputs, artifact_texts, provenance_by_call)

    def _upstream_hashes(self, call_id):
        inputs = self.inputs
        return {"call_spec": inputs.spec_hashes[call_id], "check_plan_skeleton": inputs.skeleton_hashes[call_id],
                "call_config": inputs.config_hashes[call_id], "check_plan": inputs.plan_hashes[call_id], **inputs.source_hashes()}

    def _current_upstream_hashes(self, call_id):
        return fresh_upstream_hashes(self.inputs.root, call_id)

    def _staleness(self, call_id):
        inputs = self.inputs
        return {"call_spec_sha256": inputs.spec_hashes[call_id], "check_plan_skeleton_sha256": inputs.skeleton_hashes[call_id],
                "call_config_sha256": inputs.config_hashes[call_id], "check_plan_sha256": inputs.plan_hashes.get(call_id)}

    def _reload_inputs(self):
        super()._reload_inputs()
        self.__dict__.pop("_historical_inputs", None)

    # ---- provider hooks ------------------------------------------------------------------------------
    def _default_transport_factory(self):
        from fyp_evaluation.llm.gemini_transport import GeminiTransport

        return lambda: GeminiTransport.from_env(variable=API_KEY_VARIABLE, timeout_ms=TIMEOUT_MS)

    def _attempt_metadata(self):
        return {"params": {"thinking_level": THINKING_LEVEL, "service_tier": SERVICE_TIER, "max_output_tokens": MAX_OUTPUT_TOKENS},
                "price_table_version": PRICE_TABLE_VERSION}

    def _parse_response(self, body):
        return parse_interaction(body)

    def _usage_and_cost(self, attempt, body):
        attempt["observed_model"] = body.get("model")
        usage = usage_summary(body.get("usage"))
        return usage, estimate_cost(usage)

    def _cost_estimate(self, usage):
        calls = usage["output_tokens"] // shared.ESTIMATED_OUTPUT_TOKENS
        planned = {"input_tokens": usage["input_tokens"], "cached_tokens": 0, "output_tokens": ESTIMATED_OUTPUT_TOKENS * calls,
                   "thought_tokens": 0}
        return {"assumed_output_tokens_per_call": ESTIMATED_OUTPUT_TOKENS, "flex_cost_usd": estimate_cost(planned),
                "price_table_version": PRICE_TABLE_VERSION}

    def _send(self, attempt, transport, body):
        """Send identical request bytes; resend only after a definite capacity rejection, with bounded backoff."""
        for number in range(1, CAPACITY_SENDS + 1):
            attempt["send_count"] = number  # provider sends, counted separately from this one logical attempt; saved BEFORE the send
            self._save(attempt)
            try:
                return transport.create_response(body)
            except CapacityError as error:
                attempt.setdefault("sends", []).append({"at": self.clock(), "outcome": f"capacity rejection: {str(error)[:200]}"})
                self._save(attempt)
                if number == CAPACITY_SENDS:
                    raise
                delay = CAPACITY_BACKOFF_SECONDS[min(number, len(CAPACITY_BACKOFF_SECONDS)) - 1]
                self.sleep(delay * (1 + self.jitter.uniform(-CAPACITY_JITTER, CAPACITY_JITTER)))
        raise AssertionError("unreachable")

    # ---- version-matched context (historical reports) ------------------------------------------------
    def _committed_versions(self, call_id, entry):
        """(G3 config hash, G4 plan hash) an accepted/historical report verified, from its COMMITTED attempt summary."""
        prov, problem = self._prov_state(call_id)
        if problem:
            raise HistoricalContextError(f"{call_id}: provenance is {problem}")
        summary = next((a for a in prov["attempts"] if a["attempt_id"] == entry["attempt_id"]), None)
        upstream = (summary or {}).get("upstream_sha256") or {}
        config_hash, plan_hash = upstream.get("call_config"), upstream.get("check_plan")
        if not (config_hash and plan_hash):
            raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: no committed attempt summary records the G3/G4 versions it verified")
        accepted = prov["accepted"]
        artifact_file = self.root / self.spec.artifact_path(call_id)
        if accepted and accepted["attempt_id"] == entry["attempt_id"] and artifact_file.is_file():
            artifact = parse_json_text(artifact_file.read_bytes().decode("utf-8"))
            if (artifact.get("call_config_sha256"), artifact.get("check_plan_sha256")) != (config_hash, plan_hash):
                raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: the report and its committed attempt summary disagree "
                                             "about the G3/G4 versions it verified")
        return config_hash, plan_hash

    def _inputs_for_versions(self, call_id, config_hash, plan_hash):
        """Inputs bound to the given G3/G4 versions: the current ones, or hash-verified reconstructions. Never a fallback."""
        inputs = self.inputs
        if inputs.config_hashes.get(call_id) == config_hash and inputs.plan_hashes.get(call_id) == plan_hash:
            return inputs
        cache = self.__dict__.setdefault("_historical_inputs", {})
        key = (call_id, config_hash, plan_hash)
        if key in cache:
            return cache[key]
        g4 = CheckPlanRunner(self.root, schema_dir=self.schema_dir, inputs=inputs.upstream)
        if inputs.plan_hashes.get(call_id) == plan_hash:
            plan_text = inputs.plan_texts[call_id]
        else:
            g4_prov, problem = g4._prov_state(call_id)
            if problem:
                raise HistoricalContextError(f"{call_id}: G4 provenance is {problem}")
            plan_text, problems = None, []
            for entry in [g4_prov["accepted"], *g4_prov["accepted_history"]]:
                if entry is None:
                    continue
                try:
                    text = g4._rebuild_text(call_id, entry)
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    problems.append(f"{entry.get('attempt_id')}: cannot be rebuilt ({error!r})"[:200])
                    continue
                if text_hash(text) != entry["artifact_sha256"]:
                    problems.append(f"{entry['attempt_id']}: rebuilt hash differs from its recorded artifact_sha256 (corrupted history)")
                    continue
                if text_hash(text) == plan_hash:
                    plan_text = text
                    break
            if plan_text is None:
                raise HistoricalContextError(
                    f"{call_id}: the G4 plan version {plan_hash[:12]}... this report verified cannot be reconstructed from G4's "
                    "accepted history" + (f" ({'; '.join(problems)})" if problems else ""))
        if parse_json_text(plan_text)["call_config_sha256"] != config_hash:
            raise HistoricalContextError(f"{call_id}: the reconstructed G4 plan was built from a different G3 config than the report records")
        if inputs.config_hashes.get(call_id) == config_hash:
            config_text = inputs.config_texts[call_id]
        else:
            config_text = g4._inputs_for_config(call_id, config_hash).config_texts[call_id]  # hash-verified; raises otherwise
        cache[key] = inputs_with_versions(inputs, call_id, config_text, plan_text)
        return cache[key]

    def _entry_context(self, call_id, entry):
        versioned = self._inputs_for_versions(call_id, *self._committed_versions(call_id, entry))

        def compile_for_entry(ctx, parsed, generation):
            return compile_artifact(versioned, ctx, parsed, generation, self.builder.response_schema)
        ctx = self.builder.context(call_id) if versioned is self.inputs else VerificationContext(versioned, call_id)
        return compile_for_entry, ctx

    # ---- eligibility, ambiguity and status ---------------------------------------------------------
    @staticmethod
    def _awaiting_acknowledgement(latest):
        return bool(latest and latest["state"] == "TRANSPORT_FAILED" and latest.get("ambiguous")
                    and not latest.get("ambiguous_acknowledged_at"))

    def eligibility(self, call_id, *, retry_rejected=False, reopen=False, blocked=None):
        if call_id not in self.inputs.plans:
            return False, "no accepted G4 check plan to verify"
        if call_id in self.inputs.g4_stale:
            return False, f"the G4 plan is stale ({', '.join(self.inputs.g4_stale[call_id])} changed); regenerate G4 first"
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

    def _verdict(self, call_id):
        path = self.root / self.spec.artifact_path(call_id)
        if not path.is_file():
            return None
        try:
            return parse_json_text(path.read_bytes().decode("utf-8")).get("verdict")
        except ValueError:
            return None

    def _schema_validators(self):
        """The artifact/record/provenance schema validators `validate()` uses, built once per runner."""
        if getattr(self, "_g6_schemas", None) is None:
            from jsonschema import Draft202012Validator

            from fyp_evaluation.validation.schemas import load_json

            self._g6_schemas = {role: Draft202012Validator(load_json(self.schema_dir / self.spec.schema_subdir / f"{name}.schema.json"))
                                for role, name in (("artifact", self.spec.artifact_schema), ("record", self.spec.record_schema),
                                                   ("provenance", self.spec.provenance_schema))}
        return self._g6_schemas

    def _verified_verdict(self, call_id):
        """(verdict, problem): the verdict of the accepted report, but only once its integrity is established, else (None, why).

        Integrity is the shared per-call validation `validate()` runs (provenance schema and canonical form, accepted
        provenance present and agreeing with the artifact, parsed-response and artifact hashes, rebuild against the exact G3/G4
        versions, response rules, artifact schema, staleness). The verdict is Python-derived: the artifact bytes are proven equal
        to the rebuild from the grounded response, and the same bytes that were hashed are the ones the verdict is read from.
        """
        try:
            errors = self._validate_call(call_id, self._schema_validators())
            if errors:
                return None, f"integrity check failed: {errors[0]}"[:300]
            accepted = self._prov_state(call_id)[0]["accepted"]
            text = (self.root / self.spec.artifact_path(call_id)).read_bytes().decode("utf-8").replace("\r\n", "\n")
            if accepted is None or text_hash(text) != accepted["artifact_sha256"]:
                return None, "integrity check failed: the report does not match its accepted provenance"
            return parse_json_text(text).get("verdict"), None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            return None, f"integrity check failed: {error!r}"[:300]

    def g6_eligibility(self, call_ids=None):
        """{call_id: {eligible, reason, verdict}}: a call is G6-eligible only if it has a CURRENT, INTACT accepted report with VALID.

        Fails closed. "Intact" means the per-call integrity validation passes (see `_verified_verdict`); an orphan or tampered
        report, a missing or inconsistent accepted provenance, or a stale upstream is never eligible. `verdict` is None when the
        report's integrity could not be established, so an untrusted verdict is never advertised.
        """
        result = {}
        for call_id, dependency in self.dependency_status(call_ids).items():
            verdict = None
            if dependency["status"] == "none":
                reason = "no accepted G5 report"
            elif dependency["status"] != "current":
                verdict = self._verdict(call_id)
                reason = f"G5 report is {dependency['status']}" + (f" ({', '.join(dependency['changed'])} changed)" if dependency["changed"] else "")
            else:
                verdict, problem = self._verified_verdict(call_id)
                if problem:
                    reason = problem
                elif verdict != "VALID":
                    reason = f"verdict {verdict}"
                else:
                    reason = "eligible"
            result[call_id] = {"eligible": reason == "eligible", "reason": reason, "verdict": verdict}
        return result

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

    def status(self):
        report = super().status()
        g6 = self.g6_eligibility()
        usage = self.usage_report()
        for call_id, row in report["calls"].items():
            state = self.call_state(call_id)
            if not state["in_flight"] and not state["inconsistent"] and self._awaiting_acknowledgement(state["latest"]):
                row["status"] = "AWAITING_ACKNOWLEDGEMENT"
            row["verdict"] = g6[call_id]["verdict"]
            row["g6_eligible"] = g6[call_id]["eligible"]
            row["provider_sends"] = usage["calls"][call_id]["provider_sends"]
            row["estimated_cost_usd"] = usage["calls"][call_id]["estimated_cost_usd"]
        counts = {}
        for row in report["calls"].values():
            key = row["status"].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
        report["counts"] = counts
        report["verdicts"] = {v: sum(1 for row in report["calls"].values() if row["verdict"] == v) for v in ("VALID", "INVALID", "UNCLEAR")}
        report["tier_not_echoed"] = sum(1 for c in report["calls"] if self._observed_tier_missing(c))
        completed = sum(1 for row in report["calls"].values() if row["status"].startswith("ACCEPTED"))
        report["usage"] = {**usage["totals"], "completed_calls": completed, "outstanding_calls": len(report["calls"]) - completed}
        return report

    def _observed_tier_missing(self, call_id):
        prov, problem = self._prov_state(call_id)
        accepted = prov["accepted"] if prov and not problem else None
        if not accepted:
            return False
        summary = next((a for a in prov["attempts"] if a["attempt_id"] == accepted["attempt_id"]), {})
        return summary.get("service_tier") is None

    # ---- recovery and the single mutating entry point ------------------------------------------------
    def dry_run(self, call_ids=None, **kwargs):
        return super().dry_run(call_ids or [c for c in self.inputs.call_ids if c in self.inputs.plans], **kwargs)

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
            self._ensure_fresh_inputs([c for c in targets if c in self.inputs.plans])
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
                    self._close(attempt, "TRANSPORT_FAILED", reason=f"capacity unavailable after {CAPACITY_SENDS} sends: {error}")
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


def describe(attempt):
    """One status line for an attempt (CLI)."""
    line = f"{attempt['call_id']} {attempt['attempt_id']} {attempt['state']}"
    if attempt.get("rejection_reason"):
        line += f": {attempt['rejection_reason']}"
    if attempt.get("estimated_cost_usd") is not None:
        line += f" cost~${attempt['estimated_cost_usd']}"
    return line


__all__ = ["PlanVerificationRunner", "RunError", "describe", "estimate_cost", "parse_interaction", "synthesize_response"]
