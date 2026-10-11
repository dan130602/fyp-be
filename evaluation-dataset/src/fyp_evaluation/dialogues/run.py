"""G6 library API: `DialogueRunner` is the G6 specialisation of the shared `StageRunner` lifecycle (Batch only).

G6-specific behaviour:
- per-call G5 gate: only a current, intact, VALID G5 report makes a call eligible (one call is blocked, never the run);
- acceptance order under the dataset write lock: unreadable inputs -> resumable UPSTREAM_INVALID hold; changed fingerprints
  (a confirmed-deleted G5 report included) -> terminal STALE_UPSTREAM; changed G6 settings -> terminal STALE_UPSTREAM;
  failing G5 approval with unchanged fingerprints -> resumable UPSTREAM_INVALID hold; otherwise validate and compile;
- historical dialogues are rebuilt against the exact G3/G4 versions and the verified G5 approval they recorded, with the
  length policy recorded in their own params.
"""

from copy import deepcopy

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.check_plans.inputs import HistoricalContextError
from fyp_evaluation.llm import pricing
from fyp_evaluation.llm.canonical import response_hash
from fyp_evaluation.llm.stage_runner import RunError, StageRunner, StageSpec, describe_error
from fyp_evaluation.plan_verifications.run import PlanVerificationRunner
from fyp_evaluation.planning import outputs as g1

from .compile import build_record, compile_artifact
from .config import (
    ARTIFACT_DIR, ESTIMATED_OUTPUT_TOKENS, G5_UNREADABLE, LENGTH_POLICY, MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, PROMPT_TEMPLATE,
    PROVENANCE_DIR, REASONING_EFFORT, RECORD, RESPONSE_SCHEMA, SEMANTIC_BUDGET, STAGE, TRANSPORT_BUDGET,
)
from .feasibility import preflight_errors, synthesize_dialogue
from .inputs import fresh_upstream_hashes, g5_report_fingerprint, inputs_with_versions, load_inputs
from .request import DialogueContext, RequestBuilder, schema_path
from .validate_response import DialogueValidator

G6_SPEC = StageSpec(
    stage=STAGE, label="dialogue", noun="dialogue", artifact_dir=ARTIFACT_DIR, provenance_dir=PROVENANCE_DIR, record_path=RECORD,
    schema_subdir="stage6", artifact_schema="dialogue", record_schema="dialogue_record", provenance_schema="dialogue_provenance",
    prompt_dir=PROMPT_DIR, model=MODEL, reasoning_effort=REASONING_EFFORT, max_output_tokens=MAX_OUTPUT_TOKENS,
    semantic_budget=SEMANTIC_BUDGET, transport_budget=TRANSPORT_BUDGET)


class DialogueRunner(StageRunner):
    spec = G6_SPEC
    ARTIFACT_HASH_KEYS = {**StageRunner.ARTIFACT_HASH_KEYS, "check_plan": "check_plan_sha256",
                          "plan_verification": "plan_verification_sha256"}
    GENERATION_KEYS = StageRunner.GENERATION_KEYS + ("params",)

    # ---- stage hooks ---------------------------------------------------------------------------------
    def _load_inputs(self):
        return load_inputs(self.root, self.schema_dir)

    def _make_builder(self):
        return RequestBuilder(self.inputs, self.schema_dir, self.prompt_dir)

    def _make_validator(self):
        return DialogueValidator(self.builder.response_schema)

    def _feasibility(self):
        result = {}
        for call_id in self.inputs.call_ids:
            if call_id not in self.inputs.plans:
                continue
            ctx = self.builder.context(call_id)
            errors = preflight_errors(ctx)
            result[call_id] = errors or self.validator.errors(synthesize_dialogue(ctx), ctx)
        return result

    def _compile_artifact(self, ctx, parsed, generation):
        return compile_artifact(self.inputs, ctx, parsed, generation, self.builder.response_schema)

    def _build_record(self, artifact_texts, provenance_by_call):
        return build_record(self.inputs, artifact_texts, provenance_by_call)

    def _upstream_hashes(self, call_id):
        inputs = self.inputs
        return {"call_spec": inputs.spec_hashes[call_id], "check_plan_skeleton": inputs.skeleton_hashes[call_id],
                "call_config": inputs.config_hashes[call_id], "check_plan": inputs.plan_hashes.get(call_id),
                "plan_verification": inputs.report_hashes.get(call_id), **inputs.source_hashes()}

    def _current_upstream_hashes(self, call_id):
        return fresh_upstream_hashes(self.inputs.root, call_id)

    def _staleness(self, call_id):
        inputs = self.inputs
        return {"call_spec_sha256": inputs.spec_hashes[call_id], "check_plan_skeleton_sha256": inputs.skeleton_hashes[call_id],
                "call_config_sha256": inputs.config_hashes[call_id], "check_plan_sha256": inputs.plan_hashes.get(call_id),
                "plan_verification_sha256": inputs.report_hashes.get(call_id)}

    def _reload_inputs(self):
        super()._reload_inputs()
        self.__dict__.pop("_g5_runner", None)
        self.__dict__.pop("_verified_g5", None)

    # ---- provider hooks ------------------------------------------------------------------------------
    def _attempt_metadata(self):
        return {"params": {"reasoning_effort": REASONING_EFFORT, "max_output_tokens": MAX_OUTPUT_TOKENS,
                           "length_policy": deepcopy(LENGTH_POLICY)},
                "price_table_version": pricing.PRICE_TABLE_VERSION}

    def run_sync(self, call_id, *, retry_rejected=False, reopen_reason=None):  # noqa: ARG002
        raise RunError("G6 is Batch-only: use --submit and then --collect")

    def dry_run(self, call_ids=None, **kwargs):
        report = super().dry_run(call_ids, **kwargs)
        eligible = [row for row in report["calls"] if row["eligible"]]
        output = sum(ESTIMATED_OUTPUT_TOKENS[self.builder.context(row["call_id"]).profile] for row in eligible)
        usage = {"input_tokens": report["estimate"]["approx_input_tokens"], "cached_tokens": 0, "cache_write_tokens": 0,
                 "output_tokens": output, "reasoning_tokens": 0}
        report["estimate"] = {"eligible_calls": len(eligible), "approx_input_tokens": usage["input_tokens"],
                              "assumed_output_tokens_by_profile": dict(ESTIMATED_OUTPUT_TOKENS),
                              "batch_cost_usd": pricing.estimate_cost(MODEL, usage, "batch"),
                              "price_table_version": pricing.PRICE_TABLE_VERSION, "note": "rough planning estimate only"}
        return report

    # ---- the G5 gate -----------------------------------------------------------------------------------
    def _g5(self):
        if "_g5_runner" not in self.__dict__:
            self._g5_runner = PlanVerificationRunner(self.inputs.root, audit_root=self.audit_root, schema_dir=self.schema_dir,
                                                     inputs=self.inputs.upstream)
        return self._g5_runner

    def _g5_gate(self, call_id):
        """None when the call's G5 report is current, intact and VALID right now; otherwise the reason (fails closed)."""
        if g5_report_fingerprint(self.inputs.root, call_id) == G5_UNREADABLE:
            return "the G5 report is unreadable; its fingerprint cannot be established"
        try:
            result = self._g5().g6_eligibility([call_id])[call_id]
        except Exception as error:  # noqa: BLE001 - any failure to establish the approval keeps the call ineligible
            return f"the G5 approval cannot be checked: {describe_error(error)}"[:300]
        return None if result["eligible"] else f"not G6-eligible per G5: {result['reason']}"

    def eligibility(self, call_id, *, retry_rejected=False, reopen=False, blocked=None):
        if call_id not in self.inputs.plans:
            return False, "no accepted G4 check plan"
        gate = self._g5_gate(call_id)
        if gate:
            return False, gate
        return super().eligibility(call_id, retry_rejected=retry_rejected, reopen=reopen, blocked=blocked)

    def _settings_changes(self, attempt):
        """Keys whose current value differs from what the attempt was built with. The prompt and response schema are
        re-hashed from the configured files NOW (the builder's cached hashes cannot see a change made while this process
        is alive). Raises OSError when either file cannot be read."""
        current = {"params": self._attempt_metadata()["params"],
                   "prompt_template_sha256": g1.sha256((self.prompt_dir / PROMPT_TEMPLATE).read_bytes()),
                   "response_schema_sha256": g1.sha256(schema_path(RESPONSE_SCHEMA, self.schema_dir).read_bytes())}
        return sorted(key for key, value in current.items() if attempt.get(key) != value)

    def _judge_locked(self, attempt, parsed):
        """Classify, then accept: see the module docstring for the order. Runs under the dataset write lock."""
        call_id = attempt["call_id"]
        try:
            fresh = self._current_upstream_hashes(call_id)
        except (OSError, ValueError, KeyError) as error:
            return self._hold(attempt, "UPSTREAM_INVALID", f"cannot read the upstream inputs: {describe_error(error)}")
        if fresh.get("plan_verification") == G5_UNREADABLE:
            return self._hold(attempt, "UPSTREAM_INVALID", "the G5 report is unreadable; its fingerprint cannot be established")
        recorded = attempt.get("upstream_sha256") or {}
        if any(recorded.get(key) != fresh.get(key) for key in set(recorded) | set(fresh)):
            return super()._judge_locked(attempt, parsed)  # closes the attempt as STALE_UPSTREAM
        try:
            changed = self._settings_changes(attempt)
        except OSError as error:
            return self._hold(attempt, "UPSTREAM_INVALID", f"cannot read the prompt or response schema: {describe_error(error)}")
        if changed:
            attempt["stale_upstream"] = {"changed": changed}
            self._close(attempt, "STALE_UPSTREAM", reason="G6 generation settings changed since the request was built: " + ", ".join(changed))
            return attempt
        gate = self._g5_gate(call_id)
        if gate:
            return self._hold(attempt, "UPSTREAM_INVALID", gate)
        return super()._judge_locked(attempt, parsed)

    def dependency_status(self, call_ids=None):
        """The shared status, downgraded to upstream_invalid when the G5 approval is unreadable or no longer intact."""
        result = super().dependency_status(call_ids)
        for call_id, info in result.items():
            if info["status"] == "none":
                continue
            if g5_report_fingerprint(self.inputs.root, call_id) == G5_UNREADABLE:
                result[call_id] = {"status": "upstream_invalid", "changed": [], "error": "the G5 report is unreadable"}
            elif info["status"] == "current" and (gate := self._g5_gate(call_id)):
                result[call_id] = {"status": "upstream_invalid", "changed": [], "error": gate}
        return result

    # ---- version-matched context (historical dialogues) -------------------------------------------------
    def _committed_versions(self, call_id, entry):
        """(G3 config, G4 plan, G5 report) hashes an accepted/historical dialogue was built from, from its COMMITTED summary."""
        prov, problem = self._prov_state(call_id)
        if problem:
            raise HistoricalContextError(f"{call_id}: provenance is {problem}")
        summary = next((a for a in prov["attempts"] if a["attempt_id"] == entry["attempt_id"]), None)
        upstream = (summary or {}).get("upstream_sha256") or {}
        versions = (upstream.get("call_config"), upstream.get("check_plan"), upstream.get("plan_verification"))
        if not all(versions) or versions[2] == G5_UNREADABLE:
            raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: no committed attempt summary records the G3/G4/G5 "
                                         "versions it was built from")
        accepted = prov["accepted"]
        artifact_file = self.root / self.spec.artifact_path(call_id)
        if accepted and accepted["attempt_id"] == entry["attempt_id"] and artifact_file.is_file():
            artifact = parse_json_text(artifact_file.read_bytes().decode("utf-8"))
            on_disk = (artifact.get("call_config_sha256"), artifact.get("check_plan_sha256"), artifact.get("plan_verification_sha256"))
            if on_disk != versions:
                raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: the dialogue and its committed attempt summary "
                                             "disagree about the G3/G4/G5 versions it was built from")
        return versions

    def _verified_g5_entry(self, call_id, g5_hash, config_hash, plan_hash):
        """Prove the G5 approval a dialogue recorded is intact and VALID for exactly those G3/G4 versions (or raise).

        The per-entry body of G5's own `_validate_call`, without requiring the CURRENT G5 report to be fresh (a dialogue's
        history can legitimately predate a G4 reopen whose re-verification is pending).
        """
        cache = self.__dict__.setdefault("_verified_g5", set())
        key = (call_id, g5_hash, config_hash, plan_hash)
        if key in cache:
            return
        g5 = self._g5()
        prov, problem = g5._prov_state(call_id)
        if problem:
            raise HistoricalContextError(f"{call_id}: G5 provenance is {problem}")
        entry = next((e for e in [prov["accepted"], *prov["accepted_history"]] if e and e["artifact_sha256"] == g5_hash), None)
        if entry is None:
            raise HistoricalContextError(f"{call_id}: the G5 report {g5_hash[:12]}... this dialogue recorded is not in G5's accepted history")
        try:
            if response_hash(entry["parsed_response"], g5.builder.response_schema) != entry["parsed_response_sha256"]:
                raise HistoricalContextError(f"{call_id}: the recorded G5 report's parsed response no longer matches its hash")
            if tuple(g5._committed_versions(call_id, entry)) != (config_hash, plan_hash):
                raise HistoricalContextError(f"{call_id}: the recorded G5 report approved different G3/G4 versions than this dialogue used")
            text = g5._rebuild_text(call_id, entry)
            if text_hash(text) != g5_hash:
                raise HistoricalContextError(f"{call_id}: the recorded G5 report cannot be reproduced from its stored response")
            _, g5_ctx = g5._entry_context(call_id, entry)
            if g5.validator.errors(entry["parsed_response"], g5_ctx):
                raise HistoricalContextError(f"{call_id}: the recorded G5 report's response no longer validates")
            if parse_json_text(text).get("verdict") != "VALID":
                raise HistoricalContextError(f"{call_id}: the recorded G5 report is not VALID")
        except HistoricalContextError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise HistoricalContextError(f"{call_id}: the recorded G5 report cannot be verified ({error!r})"[:300]) from error
        cache.add(key)

    def _entry_context(self, call_id, entry):
        config_hash, plan_hash, g5_hash = self._committed_versions(call_id, entry)
        upstream = self._g5()._inputs_for_versions(call_id, config_hash, plan_hash)  # hash-verified; never a fallback
        self._verified_g5_entry(call_id, g5_hash, config_hash, plan_hash)
        inputs = self.inputs
        if upstream is not inputs.upstream or inputs.report_hashes.get(call_id) != g5_hash:
            inputs = inputs_with_versions(inputs, call_id, upstream, g5_hash)
        ctx = DialogueContext(inputs, call_id, entry["params"]["length_policy"])

        def compile_for_entry(entry_ctx, parsed, generation):
            return compile_artifact(inputs, entry_ctx, parsed, generation, self.builder.response_schema)
        return compile_for_entry, ctx


__all__ = ["DialogueRunner", "RunError"]
