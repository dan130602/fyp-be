"""G7 library API: `DialogueVerificationRunner` is the G7 specialisation of the shared `StageRunner` lifecycle.

The Gemini Flex send/recovery loop is the one G5 proved (`llm.gemini_runner.GeminiFlexMixin`). G7-specific behaviour:
- per-call G7A preflight: only an accepted, current, intact G6 dialogue whose G5 approval is still current, intact and VALID
  makes a call eligible (`DialogueRunner.g7_eligibility`); one call is blocked, never the run;
- acceptance order under the dataset write lock: unreadable inputs -> resumable UPSTREAM_INVALID hold; changed fingerprints (a
  confirmed-deleted dialogue or G5 report included) -> terminal STALE_UPSTREAM; changed prompt/schema/settings, re-hashed from
  the files NOW -> terminal STALE_UPSTREAM; a failing preflight with unchanged fingerprints -> resumable hold; otherwise
  validate and compile with the same freshly verified context;
- reports are rebuilt against the exact G3/G4/G5 versions and the exact verified G6 dialogue they verified, never current ones;
- `g8_eligibility()`: VALID verdict only, from a current, intact report. The quality review is reported prominently but is
  advisory: a SERIOUS finding never changes the verdict or the eligibility.
"""

from fyp_evaluation import register_guide
from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.check_plans.inputs import HistoricalContextError
from fyp_evaluation.dialogues.config import DIALOGUE_UNREADABLE, G5_UNREADABLE
from fyp_evaluation.dialogues.inputs import inputs_with_versions as g6_inputs_with_versions
from fyp_evaluation.dialogues.run import DialogueRunner
from fyp_evaluation.llm.canonical import response_hash
from fyp_evaluation.llm.gemini_runner import FlexSettings, GeminiFlexMixin, describe, estimate_cost_with
from fyp_evaluation.llm.stage_runner import RunError, StageRunner, StageSpec, describe_error
from fyp_evaluation.llm.store import attempt_label, canonical_request_text
from fyp_evaluation.planning import outputs as g1

from .compile import build_record, compile_artifact
from .config import (
    API_KEY_VARIABLE, ARTIFACT_DIR, CAPACITY_BACKOFF_SECONDS, CAPACITY_JITTER, CAPACITY_SENDS, ESTIMATED_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS, MODEL, PRICE_TABLE_VERSION, PRICES, PROMPT_DIR, PROVENANCE_DIR, RECORD, SEMANTIC_BUDGET, SERVICE_TIER, STAGE,
    THINKING_LEVEL, TIMEOUT_MS, TRANSPORT_BUDGET, prompt_template, response_schema_name, schema_names,
)
from .inputs import fresh_upstream_hashes, inputs_with_versions, load_inputs
from .request import DialogueVerificationContext, RequestBuilder, schema_path
from .validate_response import DialogueVerificationValidator

G7_SPEC = StageSpec(
    stage=STAGE, label="verification", noun="dialogue verification", artifact_dir=ARTIFACT_DIR, provenance_dir=PROVENANCE_DIR,
    record_path=RECORD, schema_subdir="stage7", artifact_schema="dialogue_verification", record_schema="dialogue_verification_record",
    provenance_schema="dialogue_verification_provenance", prompt_dir=PROMPT_DIR, model=MODEL, reasoning_effort=THINKING_LEVEL,
    max_output_tokens=MAX_OUTPUT_TOKENS, semantic_budget=SEMANTIC_BUDGET, transport_budget=TRANSPORT_BUDGET)

G7_FLEX = FlexSettings(
    model=MODEL, service_tier=SERVICE_TIER, thinking_level=THINKING_LEVEL, max_output_tokens=MAX_OUTPUT_TOKENS,
    api_key_variable=API_KEY_VARIABLE, timeout_ms=TIMEOUT_MS, capacity_sends=CAPACITY_SENDS,
    capacity_backoff_seconds=CAPACITY_BACKOFF_SECONDS, capacity_jitter=CAPACITY_JITTER, prices=PRICES,
    price_table_version=PRICE_TABLE_VERSION)


def synthesize_response(ctx):
    """A deterministic all-MATCH report that satisfies every structural rule for this call (offline feasibility/fixtures)."""
    checks = []
    for question_id in ctx.order:
        check, evidence = ctx.checks[question_id], []
        if check["evidence_mode"] == "PRESENCE":
            event_id = check.get("final_event_id") or check["evidence_event_ids"][0]
            evidence.append({"turn_id": ctx.turn_of_event[event_id], "quote": ctx.planned_text(event_id)[:60]})
        checks.append({"question_id": question_id, "evidence": evidence,
                       "rationale": "The dialogue supports the expected outcome.", "assessment": "MATCH", "issue_ids": []})
    edge_cases = []
    for assignment_id in ctx.assignment_ids:
        carriers = ctx.events_of_assignment[assignment_id]
        evidence = [{"turn_id": ctx.turn_of_event[carriers[0]], "quote": ctx.planned_text(carriers[0])[:60]}] if carriers else []
        edge_cases.append({"assignment_id": assignment_id, "evidence": evidence,
                           "rationale": "The dialogue realises the assigned edge case.", "assessment": "MATCH", "issue_ids": []})
    report = {"checks": checks, "edge_cases": edge_cases, "issues": [], "quality_findings": []}
    if ctx.multilingual:  # one all-MATCH row per language row, each citing real speech of the kind its mechanical rule needs
        report["language_checks"] = [{"row_id": row_id, "evidence": _language_evidence(ctx, row_id),
                                      "rationale": "The dialogue realises the language requirement.", "assessment": "MATCH", "issue_ids": []}
                                     for row_id in ctx.language_row_ids]
    return report


def _language_evidence(ctx, row_id):
    """Real quotes that satisfy a MATCH row's mechanical citation rule on ANY dialogue that really meets the requirement (fixture only)."""
    from fyp_evaluation.validation.stage0 import HAN, LATIN

    from .language_rows import MIXED_BOTH, MIXED_SPREAD, MIXED_SWITCH, SINGLISH_EMPLOYEE, VARIANT_PREFIX

    def quote(turn):
        return {"turn_id": turn["turn_id"], "quote": turn["text"][:300]}

    client = [t for t in ctx.turns if t["speaker"] == "CLIENT"]
    if row_id == SINGLISH_EMPLOYEE:
        return [quote(next(t for t in ctx.turns if t["speaker"] == "EMPLOYEE"))]
    if row_id == MIXED_BOTH:
        han = next((t for t in client if HAN.search(t["text"])), client[0])
        latin = next((t for t in client if LATIN.search(t["text"])), client[0])
        return [quote(han), quote(latin)]
    if row_id == MIXED_SWITCH:
        both = min((t for t in client if HAN.search(t["text"]) and LATIN.search(t["text"])), key=lambda t: len(t["text"]), default=client[0])
        return [quote(both)]
    if row_id == MIXED_SPREAD:
        return [quote(client[0]), quote(client[-1])]
    if row_id.startswith(VARIANT_PREFIX):
        carriers = ctx.events_of_assignment.get(row_id[len(VARIANT_PREFIX):], [])
        return [{"turn_id": ctx.turn_of_event[carriers[0]], "quote": ctx.planned_text(carriers[0])[:60]}] if carriers else []
    return [quote(client[0])]


class DialogueVerificationRunner(GeminiFlexMixin, StageRunner):
    spec = G7_SPEC
    flex = G7_FLEX
    ARTIFACT_HASH_KEYS = {**StageRunner.ARTIFACT_HASH_KEYS, "check_plan": "check_plan_sha256",
                          "plan_verification": "plan_verification_sha256", "dialogue": "dialogue_sha256"}
    GENERATION_KEYS = StageRunner.GENERATION_KEYS + ("params",)
    USES_REGISTER_GUIDE = True  # SINGLISH calls record the guide version they were verified under
    supports_selection = True  # --call-id: only the selected calls need an accepted G3 config, G4 plan, G5 report and G6 dialogue

    def __init__(self, root, *, selected_call_ids=None, **kwargs):
        super().__init__(root, **kwargs)
        self.selected_call_ids = list(selected_call_ids) if selected_call_ids else None

    def _schema_names(self):
        artifact, record = schema_names(self.inputs.documents)
        return artifact, record, self.spec.provenance_schema

    def _runnable(self, call_id):
        return call_id in self.inputs.plans

    # ---- stage hooks ---------------------------------------------------------------------------------
    def _load_inputs(self):
        if self.selected_call_ids:
            return load_inputs(self.root, self.schema_dir, self.selected_call_ids)
        return load_inputs(self.root, self.schema_dir)

    def _make_builder(self):
        return RequestBuilder(self.inputs, self.schema_dir, self.prompt_dir)

    def _make_validator(self):
        return DialogueVerificationValidator(self.builder.response_schema)

    def _feasibility(self):
        """A synthetic all-MATCH report must validate for every call whose dialogue is readable."""
        return {c: self.validator.errors(synthesize_response(self.builder.context(c)), self.builder.context(c))
                for c in self.inputs.call_ids if c in self.inputs.plans and c in self.inputs.dialogues}

    def _compile_artifact(self, ctx, parsed, generation):
        return compile_artifact(self.inputs, ctx, parsed, generation, self.builder.response_schema)

    def _build_record(self, artifact_texts, provenance_by_call):
        return build_record(self.inputs, artifact_texts, provenance_by_call)

    def _upstream_hashes(self, call_id):
        inputs = self.inputs
        return {"call_spec": inputs.spec_hashes[call_id], "check_plan_skeleton": inputs.skeleton_hashes[call_id],
                "call_config": inputs.config_hashes[call_id], "check_plan": inputs.plan_hashes[call_id],
                "plan_verification": inputs.report_hashes.get(call_id), "dialogue": inputs.dialogue_hashes.get(call_id),
                **inputs.source_hashes(), **register_guide.extra_hashes(inputs.specs[call_id])}

    def _current_upstream_hashes(self, call_id):
        return fresh_upstream_hashes(self.inputs.root, call_id)

    def _staleness(self, call_id):
        inputs = self.inputs
        return {"call_spec_sha256": inputs.spec_hashes[call_id], "check_plan_skeleton_sha256": inputs.skeleton_hashes[call_id],
                "call_config_sha256": inputs.config_hashes[call_id], "check_plan_sha256": inputs.plan_hashes.get(call_id),
                "plan_verification_sha256": inputs.report_hashes.get(call_id), "dialogue_sha256": inputs.dialogue_hashes.get(call_id)}

    def _reload_inputs(self):
        super()._reload_inputs()
        for name in ("_g6_runner", "_verified_g6"):
            self.__dict__.pop(name, None)

    # ---- the G7A preflight -----------------------------------------------------------------------------
    def _g6(self):
        if "_g6_runner" not in self.__dict__:
            self._g6_runner = DialogueRunner(self.inputs.root, audit_root=self.audit_root, schema_dir=self.schema_dir,
                                             inputs=self.inputs.upstream)
        return self._g6_runner

    def _g7_gate(self, call_id):
        """None when the call's G6 dialogue is accepted, current, intact and G5-approved right now; otherwise the reason."""
        try:
            row = self._g6().g7_eligibility([call_id])[call_id]
        except Exception as error:  # noqa: BLE001 - any failure to establish the dialogue keeps the call ineligible
            return f"the G6 dialogue cannot be checked: {describe_error(error)}"[:300]
        return None if row["eligible"] else f"not G7-eligible per G6: {row['reason']}"

    def eligibility(self, call_id, *, retry_rejected=False, reopen=False, blocked=None):
        if call_id not in self.inputs.plans:
            return False, "no accepted G4 check plan"
        gate = self._g7_gate(call_id)
        if gate:
            return False, gate
        if call_id not in self.inputs.dialogues:
            return False, "the G6 dialogue is not loaded; reload the inputs"
        return super().eligibility(call_id, retry_rejected=retry_rejected, reopen=reopen, blocked=blocked)

    def _settings_changes(self, attempt):
        """Keys whose current value differs from what the attempt was built with. The prompt and response schema are re-hashed
        from the configured files NOW (the builder's cached hashes cannot see a change made while this process is alive).
        Raises OSError when either file cannot be read."""
        current = {"params": self._attempt_metadata()["params"],
                   "prompt_template_sha256": g1.sha256((self.prompt_dir / prompt_template(self.inputs.documents)).read_bytes()),
                   "response_schema_sha256": g1.sha256(schema_path(response_schema_name(self.inputs.documents), self.schema_dir).read_bytes())}
        if attempt.get("transport") == "manual":  # a review carries its own parameters; only the prompt and schema can go stale
            current["params"] = attempt.get("params")
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
        if fresh.get("dialogue") == DIALOGUE_UNREADABLE:
            return self._hold(attempt, "UPSTREAM_INVALID", "the G6 dialogue is unreadable; its fingerprint cannot be established")
        recorded = attempt.get("upstream_sha256") or {}
        if any(recorded.get(key) != fresh.get(key) for key in set(recorded) | set(fresh)):
            return super()._judge_locked(attempt, parsed)  # closes the attempt as STALE_UPSTREAM
        try:
            changed = self._settings_changes(attempt)
        except OSError as error:
            return self._hold(attempt, "UPSTREAM_INVALID", f"cannot read the prompt or response schema: {describe_error(error)}")
        if changed:
            attempt["stale_upstream"] = {"changed": changed}
            self._close(attempt, "STALE_UPSTREAM", reason="G7 generation settings changed since the request was built: " + ", ".join(changed))
            return attempt
        gate = self._g7_gate(call_id)
        if gate:
            return self._hold(attempt, "UPSTREAM_INVALID", gate)
        return super()._judge_locked(attempt, parsed)

    def dependency_status(self, call_ids=None):
        """The shared status, downgraded to upstream_invalid when the dialogue or G5 report is unreadable or the G6 gate fails."""
        result = super().dependency_status(call_ids)
        for call_id, info in result.items():
            if info["status"] == "none":
                continue
            try:
                current = self._current_upstream_hashes(call_id)
            except (OSError, ValueError, KeyError):
                continue  # the shared result already says upstream_invalid
            if current.get("dialogue") == DIALOGUE_UNREADABLE:
                result[call_id] = {"status": "upstream_invalid", "changed": [], "error": "the G6 dialogue is unreadable"}
            elif current.get("plan_verification") == G5_UNREADABLE:
                result[call_id] = {"status": "upstream_invalid", "changed": [], "error": "the G5 report is unreadable"}
            elif info["status"] == "current" and (gate := self._g7_gate(call_id)):
                result[call_id] = {"status": "upstream_invalid", "changed": [], "error": gate}
        return result

    # ---- version-matched context (historical reports) ------------------------------------------------------
    def _committed_versions(self, call_id, entry):
        """(G3 config, G4 plan, G5 report, G6 dialogue) hashes a report verified, from its COMMITTED attempt summary."""
        prov, problem = self._prov_state(call_id)
        if problem:
            raise HistoricalContextError(f"{call_id}: provenance is {problem}")
        summary = next((a for a in prov["attempts"] if a["attempt_id"] == entry["attempt_id"]), None)
        upstream = (summary or {}).get("upstream_sha256") or {}
        versions = tuple(upstream.get(key) for key in ("call_config", "check_plan", "plan_verification", "dialogue"))
        if not all(versions) or G5_UNREADABLE in versions or DIALOGUE_UNREADABLE in versions:
            raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: no committed attempt summary records the G3/G4/G5/G6 "
                                         "versions it verified")
        accepted = prov["accepted"]
        artifact_file = self.root / self.spec.artifact_path(call_id)
        if accepted and accepted["attempt_id"] == entry["attempt_id"] and artifact_file.is_file():
            artifact = parse_json_text(artifact_file.read_bytes().decode("utf-8"))
            on_disk = tuple(artifact.get(key) for key in ("call_config_sha256", "check_plan_sha256", "plan_verification_sha256",
                                                          "dialogue_sha256"))
            if on_disk != versions:
                raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: the report and its committed attempt summary "
                                             "disagree about the G3/G4/G5/G6 versions it verified")
        return versions

    def _verified_g6_entry(self, call_id, dialogue_hash, config_hash, plan_hash, g5_hash):
        """(dialogue text, G6 inputs bound to that dialogue's versions) once the G6 dialogue a report verified is proven intact.

        The per-entry body of G6's own validation, without requiring the CURRENT G6 dialogue to be fresh (a report's history can
        legitimately predate a G6 reopen). The G6 entry is found by its recorded hash, its stored response must still hash and
        rebuild to exactly that dialogue under its own context and validate, and the G3/G4/G5 versions G6 built it from must be the
        ones this report recorded. Anything else raises HistoricalContextError; the current dialogue is never substituted.
        """
        cache = self.__dict__.setdefault("_verified_g6", {})
        key = (call_id, dialogue_hash, config_hash, plan_hash, g5_hash)
        if key in cache:
            return cache[key]
        g6 = self._g6()
        prov, problem = g6._prov_state(call_id)
        if problem:
            raise HistoricalContextError(f"{call_id}: G6 provenance is {problem}")
        entry = next((e for e in [prov["accepted"], *prov["accepted_history"]] if e and e["artifact_sha256"] == dialogue_hash), None)
        if entry is None:
            raise HistoricalContextError(f"{call_id}: the G6 dialogue {dialogue_hash[:12]}... this report verified is not in G6's accepted history")
        try:
            if response_hash(entry["parsed_response"], g6.builder.response_schema) != entry["parsed_response_sha256"]:
                raise HistoricalContextError(f"{call_id}: the recorded G6 dialogue's parsed response no longer matches its hash")
            if tuple(g6._committed_versions(call_id, entry)) != (config_hash, plan_hash, g5_hash):
                raise HistoricalContextError(f"{call_id}: the recorded G6 dialogue was built from different G3/G4/G5 versions "
                                             "than this report recorded")
            text = g6._rebuild_text(call_id, entry)
            if text_hash(text) != dialogue_hash:
                raise HistoricalContextError(f"{call_id}: the recorded G6 dialogue cannot be reproduced from its stored response")
            _, g6_ctx = g6._entry_context(call_id, entry)
            if g6.validator.errors(entry["parsed_response"], g6_ctx):
                raise HistoricalContextError(f"{call_id}: the recorded G6 dialogue's response no longer validates")
            upstream = g6._g5()._inputs_for_versions(call_id, config_hash, plan_hash)  # hash-verified; never a fallback
        except HistoricalContextError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise HistoricalContextError(f"{call_id}: the recorded G6 dialogue cannot be verified ({error!r})"[:300]) from error
        g6_inputs = g6.inputs
        if upstream is not g6_inputs.upstream or g6_inputs.report_hashes.get(call_id) != g5_hash:
            g6_inputs = g6_inputs_with_versions(g6_inputs, call_id, upstream, g5_hash)
        cache[key] = (text, g6_inputs)
        return cache[key]

    def _entry_context(self, call_id, entry):
        config_hash, plan_hash, g5_hash, dialogue_hash = self._committed_versions(call_id, entry)
        text, g6_inputs = self._verified_g6_entry(call_id, dialogue_hash, config_hash, plan_hash, g5_hash)
        inputs = self.inputs
        if g6_inputs is not inputs.upstream or inputs.dialogue_hashes.get(call_id) != dialogue_hash or call_id not in inputs.dialogues:
            inputs = inputs_with_versions(inputs, call_id, g6_inputs, dialogue_hash, text)
        ctx = self.builder.context(call_id) if inputs is self.inputs else DialogueVerificationContext(inputs, call_id)

        def compile_for_entry(entry_ctx, parsed, generation):
            return compile_artifact(inputs, entry_ctx, parsed, generation, self.builder.response_schema)
        return compile_for_entry, ctx

    # ---- G8 eligibility and status --------------------------------------------------------------------------
    def _verified_report(self, call_id):
        """(artifact, problem): the accepted report, but only once its integrity is established, else (None, why).

        Integrity is the shared per-call validation `validate()` runs (provenance schema and canonical form, accepted provenance
        present and agreeing with the artifact, parsed-response and artifact hashes, rebuild against the exact G3/G4/G5/G6
        versions, response rules, artifact schema, staleness). The verdict is Python-derived and read from the same bytes that
        were hashed.
        """
        try:
            errors = self._validate_call(call_id, self._schema_validators())
            if errors:
                return None, f"integrity check failed: {errors[0]}"[:300]
            accepted = self._prov_state(call_id)[0]["accepted"]
            text = (self.root / self.spec.artifact_path(call_id)).read_bytes().decode("utf-8").replace("\r\n", "\n")
            if accepted is None or text_hash(text) != accepted["artifact_sha256"]:
                return None, "integrity check failed: the report does not match its accepted provenance"
            return parse_json_text(text), None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            return None, f"integrity check failed: {error!r}"[:300]

    def _lenient_artifact(self, call_id):
        path = self.root / self.spec.artifact_path(call_id)
        if not path.is_file():
            return None
        try:
            return parse_json_text(path.read_bytes().decode("utf-8"))
        except (OSError, ValueError):
            return None

    def g8_eligibility(self, call_ids=None):
        """{call_id: {eligible, reason, verdict, quality_review}}: a call is G8-eligible only with a CURRENT, INTACT accepted report
        whose verdict is VALID. Fails closed; `verdict` and `quality_review` are None when integrity could not be established.

        `quality_review` is advisory and never gates eligibility, but it is always reported: VALID with REVIEW_REQUIRED means
        the dialogue implements the gold and a human should still read the serious naturalness findings.
        """
        result = {}
        for call_id, dependency in self.dependency_status(call_ids).items():
            verdict = review = None
            if dependency["status"] == "none":
                reason = "no accepted G7 report"
            elif dependency["status"] != "current":
                artifact = self._lenient_artifact(call_id) or {}
                verdict, review = artifact.get("verdict"), artifact.get("quality_review")
                reason = f"G7 report is {dependency['status']}" + (
                    f" ({', '.join(dependency['changed'])} changed)" if dependency["changed"] else
                    f" ({dependency['error']})" if dependency.get("error") else "")
            else:
                artifact, problem = self._verified_report(call_id)
                if problem:
                    reason = problem
                else:
                    verdict, review = artifact["verdict"], artifact["quality_review"]
                    reason = "eligible" if verdict == "VALID" else f"verdict {verdict}"
            result[call_id] = {"eligible": reason == "eligible", "reason": reason, "verdict": verdict, "quality_review": review}
        return result

    def status(self):
        report = super().status()
        g8 = self.g8_eligibility()
        usage = self.usage_report()
        for call_id, row in report["calls"].items():
            state = self.call_state(call_id)
            if not state["in_flight"] and not state["inconsistent"] and self._awaiting_acknowledgement(state["latest"]):
                row["status"] = "AWAITING_ACKNOWLEDGEMENT"
            row["verdict"] = g8[call_id]["verdict"]
            row["quality_review"] = g8[call_id]["quality_review"]
            row["g8_eligible"] = g8[call_id]["eligible"]
            row["g8_reason"] = g8[call_id]["reason"]
            row["provider_sends"] = usage["calls"][call_id]["provider_sends"]
            row["estimated_cost_usd"] = usage["calls"][call_id]["estimated_cost_usd"]
        counts = {}
        for row in report["calls"].values():
            key = row["status"].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
        report["counts"] = counts
        report["verdicts"] = {v: sum(1 for row in report["calls"].values() if row["verdict"] == v) for v in ("VALID", "INVALID", "UNCLEAR")}
        report["quality_reviews"] = {q: sum(1 for row in report["calls"].values() if row["quality_review"] == q)
                                     for q in ("CLEAR", "REVIEW_REQUIRED")}
        report["g8_eligible"] = sum(1 for row in report["calls"].values() if row["g8_eligible"])
        report["tier_not_echoed"] = sum(1 for c in report["calls"] if self._observed_tier_missing(c))
        completed = sum(1 for row in report["calls"].values() if row["status"].startswith("ACCEPTED"))
        report["usage"] = {**usage["totals"], "completed_calls": completed, "outstanding_calls": len(report["calls"]) - completed}
        return report

    # ---- dry run ---------------------------------------------------------------------------------------------
    def dry_run(self, call_ids=None, *, retry_rejected=False, reopen_call_ids=(), print_request=False):
        """Gate, feasibility pass and request preview. Offline, credential-free, writes nothing.

        A call without a readable dialogue is listed with its reason and no request (none can be built). Token figures are
        APPROXIMATE (characters / 4) and cost is a rough planning estimate, never a promise.
        """
        inputs = self.inputs
        feasibility = self._feasibility()
        targets = list(call_ids) if call_ids else list(inputs.call_ids)
        blocked = self._blocked_calls()
        rows, total_in, planned_out = [], 0, 0
        for call_id in targets:
            if call_id not in inputs.call_ids:
                raise RunError(f"unknown call id {call_id}")
            ok, reason = self.eligibility(call_id, retry_rejected=retry_rejected, reopen=call_id in reopen_call_ids, blocked=blocked)
            row = {"call_id": call_id, "eligible": ok, "reason": reason, "attempt_id": None, "custom_id": None, "request_sha256": None,
                   "characters": None, "utf8_bytes": None, "approx_input_tokens": None}
            if call_id in inputs.plans and call_id in inputs.dialogues:
                attempt_id = attempt_label(self.call_state(call_id)["next_number"])
                text = canonical_request_text(self.builder.build(call_id, attempt_id))
                row.update(attempt_id=attempt_id, custom_id=self.spec.custom_id(call_id, attempt_id), request_sha256=text_hash(text),
                           characters=len(text), utf8_bytes=len(text.encode("utf-8")), approx_input_tokens=len(text) // 4)
                if print_request:
                    row["canonical_request"] = text
                if ok:
                    total_in += row["approx_input_tokens"]
                    planned_out += ESTIMATED_OUTPUT_TOKENS[self.builder.context(call_id).spec["metadata"]["duration_profile"]]
            rows.append(row)
        eligible = [row for row in rows if row["eligible"]]
        planned = {"input_tokens": total_in, "cached_tokens": 0, "output_tokens": planned_out, "thought_tokens": 0}
        return {
            "calls": rows,
            "feasibility": {"checked": len(feasibility), "infeasible": {k: v for k, v in feasibility.items() if v}},
            "estimate": {"eligible_calls": len(eligible), "approx_input_tokens": total_in,
                         "assumed_output_tokens_by_profile": dict(ESTIMATED_OUTPUT_TOKENS),
                         "flex_cost_usd": estimate_cost_with(planned, PRICES.get(MODEL, {}).get(SERVICE_TIER)),
                         "price_table_version": PRICE_TABLE_VERSION,
                         "note": "approximate tokens (characters / 4) and a rough planning estimate only; not a promise of cost"},
        }


__all__ = ["DialogueVerificationRunner", "RunError", "describe", "synthesize_response"]
