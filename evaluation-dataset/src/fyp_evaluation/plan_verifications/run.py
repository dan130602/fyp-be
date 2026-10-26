"""G5 library API: `PlanVerificationRunner` is the G5 specialisation of the shared `StageRunner` lifecycle.

It reuses attempts, provenance, accepted history, the freshness check under `.stage_write.lock`, dependency status,
`--validate` and `--check`. G5-specific: a Gemini Flex transport (no Batch), application-owned capacity retries, explicit
handling of ambiguous outcomes, fatal tier/model echo mismatches, recovery BEFORE eligibility in its single mutating
entry point `run()`, and reports compiled against the exact G3/G4 versions they verified.
"""

from fyp_evaluation import register_guide
from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.check_plans.inputs import HistoricalContextError
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.llm import stage_runner as shared
from fyp_evaluation.llm.gemini_runner import (
    FlexSettings, GeminiFlexMixin, describe, estimate_cost_with, parse_interaction as _parse_interaction, usage_summary,
)
from fyp_evaluation.llm.stage_runner import RunError, StageRunner, StageSpec

from .compile import build_record, compile_artifact
from .config import (
    API_KEY_VARIABLE, ARTIFACT_DIR, CAPACITY_BACKOFF_SECONDS, CAPACITY_JITTER, CAPACITY_SENDS, ESTIMATED_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS, MODEL, PRICE_TABLE_VERSION, PRICES, PROMPT_DIR, PROVENANCE_DIR, RECORD, SEMANTIC_BUDGET, SERVICE_TIER,
    STAGE, THINKING_LEVEL, TIMEOUT_MS, TRANSPORT_BUDGET, schema_names,
)
from .inputs import fresh_upstream_hashes, inputs_with_versions, load_inputs
from .request import RequestBuilder, VerificationContext
from .validate_response import VerificationValidator

G5_SPEC = StageSpec(
    stage=STAGE, label="verification", noun="plan verification", artifact_dir=ARTIFACT_DIR, provenance_dir=PROVENANCE_DIR,
    record_path=RECORD, schema_subdir="stage5", artifact_schema="plan_verification", record_schema="plan_verification_record",
    provenance_schema="plan_verification_provenance", prompt_dir=PROMPT_DIR, model=MODEL, reasoning_effort=THINKING_LEVEL,
    max_output_tokens=MAX_OUTPUT_TOKENS, semantic_budget=SEMANTIC_BUDGET, transport_budget=TRANSPORT_BUDGET)

G5_FLEX = FlexSettings(
    model=MODEL, service_tier=SERVICE_TIER, thinking_level=THINKING_LEVEL, max_output_tokens=MAX_OUTPUT_TOKENS,
    api_key_variable=API_KEY_VARIABLE, timeout_ms=TIMEOUT_MS, capacity_sends=CAPACITY_SENDS,
    capacity_backoff_seconds=CAPACITY_BACKOFF_SECONDS, capacity_jitter=CAPACITY_JITTER, prices=PRICES,
    price_table_version=PRICE_TABLE_VERSION)


def parse_interaction(body):
    """Interpret one Interactions body: ("ok", parsed, None) | ("rejected"|"transport"|"fatal", None, reason)."""
    return _parse_interaction(body, MODEL, SERVICE_TIER)


def estimate_cost(usage, model=MODEL, tier=SERVICE_TIER):
    """Estimated USD (thinking tokens billed as output), or None when the model/tier has no price."""
    return estimate_cost_with(usage, PRICES.get(model, {}).get(tier))


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


class PlanVerificationRunner(GeminiFlexMixin, StageRunner):
    spec = G5_SPEC
    flex = G5_FLEX
    ARTIFACT_HASH_KEYS = {**StageRunner.ARTIFACT_HASH_KEYS, "check_plan": "check_plan_sha256"}
    GENERATION_KEYS = StageRunner.GENERATION_KEYS + ("params",)
    supports_selection = True  # --call-id: only the selected calls need an accepted G3 config and G4 plan

    def __init__(self, root, *, selected_call_ids=None, **kwargs):
        super().__init__(root, **kwargs)
        self.selected_call_ids = list(selected_call_ids) if selected_call_ids else None

    def _runnable(self, call_id):
        return call_id in self.inputs.plans

    # ---- stage hooks ---------------------------------------------------------------------------------
    def _load_inputs(self):
        if self.selected_call_ids:
            return load_inputs(self.root, self.schema_dir, self.selected_call_ids)
        return load_inputs(self.root, self.schema_dir)

    def _schema_names(self):
        artifact, record = schema_names(self.inputs.documents)
        return artifact, record, self.spec.provenance_schema

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
                "call_config": inputs.config_hashes[call_id], "check_plan": inputs.plan_hashes[call_id], **inputs.source_hashes(),
                **register_guide.extra_hashes(inputs.specs[call_id])}

    def _current_upstream_hashes(self, call_id):
        return fresh_upstream_hashes(self.inputs.root, call_id)

    def _staleness(self, call_id):
        inputs = self.inputs
        return {"call_spec_sha256": inputs.spec_hashes[call_id], "check_plan_skeleton_sha256": inputs.skeleton_hashes[call_id],
                "call_config_sha256": inputs.config_hashes[call_id], "check_plan_sha256": inputs.plan_hashes.get(call_id)}

    def _reload_inputs(self):
        super()._reload_inputs()
        self.__dict__.pop("_historical_inputs", None)

    # ---- provider hooks (the Gemini Flex send/recovery loop lives in GeminiFlexMixin) --------------------
    def _cost_estimate(self, usage):
        calls = usage["output_tokens"] // shared.ESTIMATED_OUTPUT_TOKENS
        planned = {"input_tokens": usage["input_tokens"], "cached_tokens": 0, "output_tokens": ESTIMATED_OUTPUT_TOKENS * calls,
                   "thought_tokens": 0}
        return {"assumed_output_tokens_per_call": ESTIMATED_OUTPUT_TOKENS, "flex_cost_usd": estimate_cost(planned),
                "price_table_version": PRICE_TABLE_VERSION}

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
    def eligibility(self, call_id, *, retry_rejected=False, reopen=False, blocked=None):
        if call_id not in self.inputs.plans:
            return False, "no accepted G4 check plan to verify"
        if call_id in self.inputs.g4_stale:
            return False, f"the G4 plan is stale ({', '.join(self.inputs.g4_stale[call_id])} changed); regenerate G4 first"
        return super().eligibility(call_id, retry_rejected=retry_rejected, reopen=reopen, blocked=blocked)  # acknowledgement, then shared

    def _verdict(self, call_id):
        path = self.root / self.spec.artifact_path(call_id)
        if not path.is_file():
            return None
        try:
            return parse_json_text(path.read_bytes().decode("utf-8")).get("verdict")
        except ValueError:
            return None

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

    def dry_run(self, call_ids=None, **kwargs):
        return super().dry_run(call_ids or [c for c in self.inputs.call_ids if c in self.inputs.plans], **kwargs)


__all__ = ["PlanVerificationRunner", "RunError", "describe", "estimate_cost", "parse_interaction", "synthesize_response", "usage_summary"]
