"""Shared lifecycle of a model-assisted authoring stage (G3 call configs, G4 check plans).

`StageRunner` owns the workflow that does not depend on what the stage writes: selection and eligibility, attempts,
the sync and Batch flows, adjudication of responses, accepted artifacts with their provenance and history, recovery of
interrupted state, `--validate` and `--check`. A stage supplies a `StageSpec` (paths, schemas, settings) and a handful of
hooks (inputs, request builder, validator, compiler, record, upstream hashes). Transports are injected.

State lives in two places only: the gitignored audit store (live attempts/submissions) and the committed provenance
files. Nothing here reads or writes frozen upstream files.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.llm import pricing, provenance
from fyp_evaluation.llm.canonical import canonical_response, response_hash
from fyp_evaluation.llm.env import PROJECT_ROOT
from fyp_evaluation.llm.errors import QUOTA_CODES, AmbiguousError, FatalRequestError, TransportError
from fyp_evaluation.llm.fsutil import write_text_atomic
from fyp_evaluation.llm.responses import is_transient_error_code, parse_provider_response
from fyp_evaluation.llm.store import (
    AttemptStore, StageWriteLockBusy, attempt_label, canonical_request_text, dataset_write_lock, submission_label,
)
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

IN_FLIGHT = ("PREPARED", "SUBMITTED", "RESPONDED")
BLOCKING_SUBMISSION_STATES = ("INTENT_RECORDED", "FILE_UPLOADED", "SUBMITTED", "SUBMISSION_UNKNOWN")
TERMINAL_BATCH = ("completed", "expired", "cancelled", "failed")
ESTIMATED_OUTPUT_TOKENS = 4000  # rough planning figure for --dry-run only
DEFAULT_AUDIT_ROOT = PROJECT_ROOT / ".llm_audit"


@dataclass(frozen=True)
class StageSpec:
    """Everything that names one stage: identity, generated paths, schema files, request settings and budgets."""

    stage: str  # "G3" / "G4"
    label: str  # short wording in messages, e.g. "config"
    noun: str  # full artifact name, e.g. "call config"
    artifact_dir: str
    provenance_dir: str
    record_path: str
    schema_subdir: str  # under the schema directory, e.g. "stage3"
    artifact_schema: str
    record_schema: str
    provenance_schema: str
    prompt_dir: Path
    model: str
    reasoning_effort: str
    max_output_tokens: int
    semantic_budget: int = 3  # REJECTED attempts since the last accepted version
    transport_budget: int = 3  # non-fatal TRANSPORT_FAILED attempts since the last accepted version

    def artifact_path(self, call_id):
        return f"{self.artifact_dir}/{call_id}.json"

    def provenance_path(self, call_id):
        return f"{self.provenance_dir}/{call_id}.json"

    def custom_id(self, call_id, attempt_id):
        """Valid for both OpenAI and Anthropic batch APIs: ^[a-zA-Z0-9_-]{1,64}$."""
        return f"{call_id}-{self.stage}-{attempt_id}"


def describe_error(error):
    """A short, path-free description (the file NAME and reason for an OSError) so messages are not cut inside a long path."""
    filename = getattr(error, "filename", None)
    if isinstance(error, OSError) and filename:
        return f"{Path(str(filename)).name}: {error.strerror or type(error).__name__}"
    return str(error)


class RunError(ValueError):
    """A precondition failed (ineligible call, wrong submission state, ...); nothing was changed or sent."""


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StageRunner:
    """Lifecycle shared by every stage. Subclasses set `spec` and implement the hooks marked below."""

    spec = None  # StageSpec
    write_lock_wait = 10.0  # seconds to wait for the short shared dataset write lock before deferring an acceptance

    # artifact key of each upstream hash an artifact records; any other key lives under artifact["source"]
    ARTIFACT_HASH_KEYS = {"call_spec": "call_spec_sha256", "check_plan_skeleton": "check_plan_skeleton_sha256",
                          "call_config": "call_config_sha256"}

    def __init__(self, root, *, audit_root=DEFAULT_AUDIT_ROOT, schema_dir=DEFAULT_SCHEMA_DIR, prompt_dir=None,
                 transport=None, transport_factory=None, clock=utc_now, inputs=None):
        self.root, self.schema_dir = Path(root), Path(schema_dir)
        self.prompt_dir = Path(prompt_dir) if prompt_dir is not None else self.spec.prompt_dir
        self.audit_root, self.clock = Path(audit_root), clock
        self._transport, self._transport_factory = transport, transport_factory
        self._inputs, self._builder, self._validator, self._store = inputs, None, None, None

    # ---- stage hooks (implemented by each stage) -------------------------------------------------------
    def _load_inputs(self):
        """Gate on the validated upstream state and return the inputs object."""
        raise NotImplementedError

    def _make_builder(self):
        """Request builder: .build(call_id, attempt_id), .context(call_id), .prompt_template_sha256,
        .response_schema_sha256, .response_schema."""
        raise NotImplementedError

    def _make_validator(self):
        """Object with .errors(parsed_response, context) -> list of error strings."""
        raise NotImplementedError

    def _feasibility(self):
        """{call_id: [errors]} for the offline synthetic-response pass used by --dry-run."""
        raise NotImplementedError

    def _compile_artifact(self, ctx, parsed, generation):
        """The artifact dict for a validated parsed response."""
        raise NotImplementedError

    def _build_record(self, artifact_texts, provenance_by_call):
        """The deterministic record dict over the accepted artifacts."""
        raise NotImplementedError

    def _upstream_hashes(self, call_id):
        """Hashes of the inputs one attempt was built from (stored on the attempt)."""
        raise NotImplementedError

    def _staleness(self, call_id):
        """{artifact key: current upstream hash}: an accepted artifact whose recorded value differs is stale."""
        raise NotImplementedError

    def _current_upstream_hashes(self, call_id):
        """The same hashes as `_upstream_hashes`, but re-read from the upstream FILES on disk right now."""
        raise NotImplementedError

    def _entry_context(self, call_id, entry):
        """(compile function, context) matching the upstream version an accepted or historical entry was built against.

        The default is the current context; a stage whose upstream can change (G4 over G3) overrides it and must fail
        explicitly when the matching version cannot be reconstructed.
        """
        return self._compile_artifact, self.builder.context(call_id)

    # ---- lazy collaborators --------------------------------------------------------------------------
    @property
    def inputs(self):
        if self._inputs is None:
            self._inputs = self._load_inputs()
        return self._inputs

    @property
    def builder(self):
        if self._builder is None:
            self._builder = self._make_builder()
        return self._builder

    @property
    def validator(self):
        if self._validator is None:
            self._validator = self._make_validator()
        return self._validator

    @property
    def store(self):
        if self._store is None:
            self._store = AttemptStore(self.audit_root, self.inputs.dataset_id, self.spec.stage)
        return self._store

    def transport(self):
        """The injected transport, or a real one built on first use (raises CredentialsError before any request)."""
        if self._transport is None:
            if self._transport_factory is None:
                from fyp_evaluation.llm.openai_transport import OpenAITransport

                self._transport_factory = OpenAITransport.from_env
            self._transport = self._transport_factory()
        return self._transport

    # ---- state ---------------------------------------------------------------------------------------
    def _provenance_validator(self):
        if getattr(self, "_prov_validator", None) is None:
            from jsonschema import Draft202012Validator

            from fyp_evaluation.validation.schemas import load_json

            self._prov_validator = Draft202012Validator(load_json(
                self.schema_dir / self.spec.schema_subdir / f"{self.spec.provenance_schema}.schema.json"))
        return self._prov_validator

    def _prov_state(self, call_id):
        """(provenance, problem): the schema-valid provenance (empty if absent), or (None, reason) if unreadable/invalid."""
        if not (self.root / self.spec.provenance_path(call_id)).is_file():
            return provenance.empty(self.spec, self.inputs.dataset_id, call_id), None
        try:
            prov = provenance.read(self.spec, self.root, self.inputs.dataset_id, call_id)
        except (ValueError, OSError) as error:
            return None, f"unreadable ({str(error)[:120]})"
        from jsonschema.exceptions import best_match

        best = best_match(self._provenance_validator().iter_errors(prov))  # descends into anyOf to name the real defect
        if best is not None:
            return None, f"invalid ({best.json_path}: {best.message})"[:260]
        return prov, None

    def _attempts(self, call_id):
        """{attempt_id: record}: committed summaries overlaid by the live audit records."""
        prov, _ = self._prov_state(call_id)
        merged = {a["attempt_id"]: a for a in (prov["attempts"] if prov else [])}
        for attempt_id in self.store.attempt_ids(call_id):
            merged[attempt_id] = self.store.read_attempt(call_id, attempt_id)
        return merged

    @staticmethod
    def _number(attempt_id):
        return int(attempt_id[1:])

    def call_state(self, call_id):
        attempts = self._attempts(call_id)
        accepted_numbers = [self._number(a) for a, rec in attempts.items() if rec["state"] in ("ACCEPTED", "SUPERSEDED")]
        epoch = max(accepted_numbers, default=0)
        since = [rec for a, rec in attempts.items() if self._number(a) > epoch]
        ordered = sorted(attempts.values(), key=lambda rec: rec["attempt_id"])
        prov, problem = self._prov_state(call_id)
        accepted_in_provenance = prov["accepted"] if prov else None
        artifact_exists = (self.root / self.spec.artifact_path(call_id)).is_file()
        inconsistent = None
        if problem:
            inconsistent = (f"INCONSISTENT: provenance for {call_id} is {problem}; no paid attempt is allowed. "
                            "Investigate (see --validate) before doing anything else")
        elif accepted_in_provenance and not artifact_exists:
            inconsistent = (f"INCONSISTENT: provenance records accepted attempt {accepted_in_provenance['attempt_id']} but "
                            f"{self.spec.artifact_path(call_id)} is missing; no paid attempt is allowed. Investigate (see --validate) and restore "
                            "the artifact; a stored but unjudged response is completed automatically by --collect")
        elif artifact_exists and not accepted_in_provenance:
            inconsistent = (f"INCONSISTENT: {self.spec.artifact_path(call_id)} exists but provenance records no accepted attempt; no paid "
                            "attempt or --reopen is allowed. Investigate (see --validate); do not delete either file until you "
                            "know which is authoritative")
        return {
            "call_id": call_id,
            "accepted": artifact_exists,
            # accepted artifact and accepted provenance disagree (either direction), or provenance is unusable
            "inconsistent": inconsistent,
            "attempts": ordered,
            "in_flight": [rec for rec in ordered if rec["state"] in IN_FLIGHT],
            "latest": ordered[-1] if ordered else None,
            "rejected": sum(1 for rec in since if rec["state"] == "REJECTED"),
            "transport_failed": sum(1 for rec in since if rec["state"] == "TRANSPORT_FAILED" and not rec.get("fatal")),
            "next_number": max((self._number(a) for a in attempts), default=0) + 1,
        }

    def _blocked_calls(self):
        blocked = {}
        for submission_id in self.store.submission_ids():
            submission = self.store.read_submission(submission_id)
            if submission["state"] in BLOCKING_SUBMISSION_STATES:
                for call_id in submission["call_ids"]:
                    blocked[call_id] = f"{submission_id} is {submission['state']}"
        return blocked

    def eligibility(self, call_id, *, retry_rejected=False, reopen=False, blocked=None):
        """(eligible, reason) for starting a new paid attempt on one call."""
        state = self.call_state(call_id)
        blocked = self._blocked_calls() if blocked is None else blocked
        if state["in_flight"]:
            return False, f"attempt {state['in_flight'][-1]['attempt_id']} is {state['in_flight'][-1]['state']}"
        if call_id in blocked:
            return False, blocked[call_id]
        if state["inconsistent"]:
            return False, state["inconsistent"]
        if state["accepted"] and not reopen:
            return False, "already accepted (use --reopen --reason to supersede it deliberately)"
        if reopen and not state["accepted"]:
            return False, f"--reopen only applies to a call with an accepted {self.spec.label}"
        if state["rejected"] >= self.spec.semantic_budget:
            return False, f"AWAITING_HUMAN: {state['rejected']} rejected attempts (budget {self.spec.semantic_budget})"
        if state["transport_failed"] >= self.spec.transport_budget:
            return False, f"AWAITING_HUMAN: {state['transport_failed']} transport failures (budget {self.spec.transport_budget})"
        latest = state["latest"]
        if latest and latest["state"] == "REJECTED" and not retry_rejected and not reopen:
            return False, "latest attempt was REJECTED (use --retry-rejected after reviewing the reasons)"
        return True, "eligible"

    def _select(self, call_ids, *, retry_rejected, reopen_call_ids):
        blocked = self._blocked_calls()
        explicit = call_ids is not None
        chosen, skipped = [], {}
        for call_id in (call_ids if explicit else self.inputs.call_ids):
            if call_id not in self.inputs.call_ids:
                raise RunError(f"unknown call id {call_id}")
            ok, reason = self.eligibility(call_id, retry_rejected=retry_rejected, reopen=call_id in reopen_call_ids, blocked=blocked)
            if ok:
                chosen.append(call_id)
            else:
                skipped[call_id] = reason
        if explicit and skipped:
            raise RunError("; ".join(f"{call_id}: {reason}" for call_id, reason in skipped.items()))
        return chosen, skipped

    # ---- attempts ------------------------------------------------------------------------------------
    def _new_attempt(self, call_id, transport, *, submission_id=None, reopen_reason=None, reopen_of=None):
        number = self.call_state(call_id)["next_number"]
        attempt_id = attempt_label(number)
        body = self.builder.build(call_id, attempt_id)
        text = canonical_request_text(body)
        inputs = self.inputs
        attempt = {
            "schema_version": "1.0", "stage": self.spec.stage, "dataset_id": inputs.dataset_id, "call_id": call_id,
            "attempt_id": attempt_id, "attempt_number": number, "custom_id": self.spec.custom_id(call_id, attempt_id),
            "transport": transport, "state": "PREPARED", "submission_id": submission_id, "batch_id": None,
            "provider_response_id": None, "model": self.spec.model,
            "params": {"reasoning_effort": self.spec.reasoning_effort, "max_output_tokens": self.spec.max_output_tokens},
            "prompt_template_sha256": self.builder.prompt_template_sha256,
            "response_schema_sha256": self.builder.response_schema_sha256,
            "request_sha256": text_hash(text),
            "upstream_sha256": self._upstream_hashes(call_id),
            "api_status": None, "rejection_reason": None, "validation_errors": [], "fatal": False, "possibly_billed": False,
            "usage": None, "service_tier": None, "estimated_cost_usd": None, "price_table_version": pricing.PRICE_TABLE_VERSION,
            "reopen": {"reason": reopen_reason, "previous_attempt_id": reopen_of} if reopen_reason else None,
            "created_at": self.clock(), "finished_at": "",
        }
        self.store.write_request(call_id, attempt_id, text)
        self.store.write_attempt(attempt)
        return attempt, body

    def _save(self, attempt):
        self.store.write_attempt(attempt)

    def _close(self, attempt, state, *, reason=None, errors=None, fatal=False, possibly_billed=False):
        """Move an attempt to a terminal state and commit its summary to the provenance file."""
        attempt.pop("blocked", None)
        attempt.update(state=state, rejection_reason=reason, validation_errors=list(errors or []), fatal=fatal,
                       possibly_billed=possibly_billed, finished_at=self.clock())
        self._save(attempt)
        record = provenance.read(self.spec, self.root, self.inputs.dataset_id, attempt["call_id"])
        provenance.upsert_attempt(record, attempt)
        provenance.write(self.spec, self.root, record)

    def _fail_transport(self, attempt, error, *, possibly_billed=False):
        self._close(attempt, "TRANSPORT_FAILED", reason=str(error), fatal=bool(getattr(error, "fatal", False)),
                    possibly_billed=possibly_billed)

    def _store_response(self, attempt, body, meta):
        """Persist the raw provider envelope before judging it, so a crash never loses a paid response."""
        self.store.write_raw_response(attempt["call_id"], attempt["attempt_id"], {"meta": meta, "response": body})
        attempt.update(state="RESPONDED", provider_response_id=body.get("id"), api_status=body.get("status"),
                       service_tier=body.get("service_tier"))
        usage = pricing.usage_summary(body.get("usage"))
        attempt["usage"] = usage
        attempt["estimated_cost_usd"] = pricing.estimate_cost(attempt["model"], usage, "batch" if attempt["transport"] == "batch" else "standard")
        self._save(attempt)

    def _adjudicate(self, attempt):
        """Judge a RESPONDED attempt: parse, validate, then accept or reject. Idempotent for crash recovery."""
        raw = self.store.read_raw_response(attempt["call_id"], attempt["attempt_id"])
        kind, parsed, reason = parse_provider_response(raw["response"])
        if kind == "transport":
            self._close(attempt, "TRANSPORT_FAILED", reason=reason, possibly_billed=True)
            return attempt
        if kind == "rejected":
            self._close(attempt, "REJECTED", reason=reason)
            return attempt
        return self._judge_parsed(attempt, parsed)

    # ---- acceptance safety: upstream freshness under the short shared write lock --------------------------
    def _judge_parsed(self, attempt, parsed):
        """Validate and accept a parsed response only against the upstream it was requested with (fail-closed).

        The shared dataset write lock covers only 'verify upstream -> validate -> write'; it is never held across a
        provider call. If it stays busy the response is kept (RESPONDED) and the next collect finishes it.
        """
        try:
            with dataset_write_lock(self.root, wait_seconds=self.write_lock_wait):
                return self._judge_locked(attempt, parsed)
        except StageWriteLockBusy as error:
            return self._hold(attempt, "WRITE_LOCK_BUSY", str(error))

    def _hold(self, attempt, kind, reason):
        """Keep a paid response unjudged (RESPONDED) and say why; a later collect/sync retries it."""
        attempt["blocked"] = {"kind": kind, "reason": reason[:400]}
        attempt["rejection_reason"] = f"{kind}: {reason[:400]}"
        self._save(attempt)
        return attempt

    def _judge_locked(self, attempt, parsed):
        call_id = attempt["call_id"]
        try:
            fresh = self._current_upstream_hashes(call_id)
        except (OSError, ValueError, KeyError) as error:
            return self._hold(attempt, "UPSTREAM_INVALID", f"cannot read the upstream inputs: {describe_error(error)}")
        recorded = attempt.get("upstream_sha256") or {}
        changed = sorted(key for key in set(recorded) | set(fresh) if recorded.get(key) != fresh.get(key))
        if changed:  # the response answers a request built from different upstream; never rebind it
            attempt["stale_upstream"] = {"changed": changed}
            self._close(attempt, "STALE_UPSTREAM", reason="upstream changed since the request was built: " + ", ".join(changed))
            return attempt
        try:
            if self._upstream_hashes(call_id) != fresh:  # this runner's cached inputs lag behind the files on disk
                self._reload_inputs()
        except Exception as error:  # noqa: BLE001 - any failure to load valid upstream blocks acceptance, never silently
            return self._hold(attempt, "UPSTREAM_INVALID", f"the upstream inputs failed validation: {error}")
        ctx = self.builder.context(call_id)  # the freshly verified context, used for validation AND compilation
        errors = self.validator.errors(parsed, ctx)
        if errors:
            self._close(attempt, "REJECTED", reason="validation failed", errors=errors)
            return attempt
        return self._accept(attempt, ctx, parsed)

    def _reload_inputs(self):
        inputs = self._load_inputs()  # raises when the upstream is invalid; the previous cache is then left untouched
        self._inputs, self._builder, self._validator = inputs, None, None

    def _ensure_fresh_inputs(self, call_ids):
        """Before building paid requests: reload the inputs if any upstream file changed since they were cached."""
        try:
            changed = any(self._current_upstream_hashes(c) != self._upstream_hashes(c) for c in call_ids)
        except (OSError, ValueError, KeyError) as error:
            raise RunError(f"upstream inputs are unreadable; no request was sent: {describe_error(error)}") from error
        if changed:
            try:
                self._reload_inputs()
            except Exception as error:  # noqa: BLE001
                raise RunError(f"upstream inputs changed but are not valid; no request was sent: {error}") from error

    def _accept(self, attempt, ctx, parsed):
        call_id, root = attempt["call_id"], self.root
        schema = self.builder.response_schema
        record = provenance.read(self.spec, root, self.inputs.dataset_id, call_id)
        previous = record["accepted"]
        reopen = attempt.get("reopen")
        if previous and previous["attempt_id"] != attempt["attempt_id"]:
            if not reopen:
                self._close(attempt, "REJECTED", reason=f"call already has an accepted {self.spec.label}; --reopen was not requested")
                return attempt
            record["accepted_history"].append({**previous, "superseded_at": self.clock(), "reason": reopen["reason"]})
            for entry in record["attempts"]:
                if entry["attempt_id"] == previous["attempt_id"]:
                    entry["state"] = "SUPERSEDED"
            self._mark_audit_superseded(call_id, previous["attempt_id"])
        generation = {key: attempt[key] for key in ("attempt_id", "model", "prompt_template_sha256", "response_schema_sha256")}
        artifact_text = serialize_json(self._compile_artifact(ctx, parsed, generation))
        scenario = canonical_response(parsed, schema)
        record["accepted"] = {**generation, "parsed_response": scenario, "parsed_response_sha256": response_hash(parsed, schema),
                              "artifact_sha256": text_hash(artifact_text)}
        attempt.pop("blocked", None)
        attempt.update(state="ACCEPTED", rejection_reason=None, validation_errors=[], finished_at=self.clock())
        provenance.upsert_attempt(record, attempt)
        provenance.write(self.spec, root, record)  # provenance first: an interrupted run is repaired by re-adjudicating
        write_text_atomic(root / self.spec.artifact_path(call_id), artifact_text)
        self._write_record()
        self._save(attempt)
        return attempt

    def _mark_audit_superseded(self, call_id, attempt_id):
        if attempt_id in self.store.attempt_ids(call_id):
            old = self.store.read_attempt(call_id, attempt_id)
            old["state"] = "SUPERSEDED"
            self.store.write_attempt(old)

    def _artifact_texts(self):
        texts = {}
        for call_id in self.inputs.call_ids:
            path = self.root / self.spec.artifact_path(call_id)
            if path.is_file():
                texts[call_id] = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        return texts

    def _provenance_map(self):
        """{call_id: provenance} for every call whose provenance is readable and schema-valid."""
        result = {}
        for call_id in self.inputs.call_ids:
            prov, problem = self._prov_state(call_id)
            if prov is not None and problem is None:
                result[call_id] = prov
        return result

    def _write_record(self):
        provs = self._provenance_map()
        texts = {c: t for c, t in self._artifact_texts().items() if c in provs}
        if texts:
            record = self._build_record(texts, provs)
            write_text_atomic(self.root / self.spec.record_path, serialize_json(record))

    def _stale_submissions(self):
        return [s for s in self.store.submission_ids() if self.store.read_submission(s)["state"] in ("INTENT_RECORDED", "FILE_UPLOADED")]

    def _reconcile_submissions(self):
        """Under the run lock, repair submissions an interrupted run left half-updated. Sends nothing to the provider."""
        for submission_id in self.store.submission_ids():
            sub = self.store.read_submission(submission_id)
            if sub["state"] == "SUBMITTED":  # batch recorded; some per-attempt updates may not have happened
                for custom, entry in sub["entries"].items():
                    attempt = self.store.read_attempt(entry["call_id"], entry["attempt_id"])
                    if attempt["state"] == "PREPARED" and attempt["submission_id"] == submission_id and attempt["custom_id"] == custom:
                        attempt.update(state="SUBMITTED", batch_id=sub["batch_id"])
                        self._save(attempt)
            elif sub["state"] == "FILE_UPLOADED":  # died during/after batch creation: outcome unknown, never retried blindly
                self._set_submission_state(sub, "SUBMISSION_UNKNOWN",
                                           error="interrupted after the file upload; whether the batch was created is unknown")
            elif sub["state"] == "INTENT_RECORDED":  # died before the upload: nothing can exist at the provider
                self._set_submission_state(sub, "NOT_SUBMITTED", error="interrupted before the upload; nothing was sent")
                for entry in sub["entries"].values():
                    attempt = self.store.read_attempt(entry["call_id"], entry["attempt_id"])
                    if attempt["state"] == "PREPARED":
                        self._close(attempt, "TRANSPORT_FAILED", reason="interrupted before the upload; nothing was sent")

    def _recover(self):
        self._reconcile_submissions()
        return self._finalize_responded()

    def _finalize_responded(self):
        """Crash recovery: adjudicate every attempt whose response was stored but not yet judged."""
        done = []
        for call_id in self.inputs.call_ids:
            for attempt_id in self.store.attempt_ids(call_id):
                attempt = self.store.read_attempt(call_id, attempt_id)
                if attempt["state"] == "RESPONDED":
                    done.append(self._adjudicate(attempt))
        return done

    # ---- read-only commands --------------------------------------------------------------------------
    def dry_run(self, call_ids=None, *, retry_rejected=False, reopen_call_ids=(), print_request=False):
        """Gate, feasibility pass and request preview. Sends nothing and writes nothing."""
        inputs = self.inputs
        feasibility = self._feasibility()
        targets = list(call_ids) if call_ids else list(inputs.call_ids)
        blocked = self._blocked_calls()
        rows, total_in = [], 0
        for call_id in targets:
            if call_id not in inputs.call_ids:
                raise RunError(f"unknown call id {call_id}")
            ok, reason = self.eligibility(call_id, retry_rejected=retry_rejected, reopen=call_id in reopen_call_ids, blocked=blocked)
            attempt_id = attempt_label(self.call_state(call_id)["next_number"])
            body = self.builder.build(call_id, attempt_id)
            text = canonical_request_text(body)
            tokens = len(text) // 4
            row = {"call_id": call_id, "attempt_id": attempt_id, "custom_id": self.spec.custom_id(call_id, attempt_id),
                   "request_sha256": text_hash(text), "approx_input_tokens": tokens, "eligible": ok, "reason": reason}
            if print_request:
                row["canonical_request"] = text
            rows.append(row)
            if ok:
                total_in += tokens
        eligible = [row for row in rows if row["eligible"]]
        usage = {"input_tokens": total_in, "cached_tokens": 0, "cache_write_tokens": 0,
                 "output_tokens": ESTIMATED_OUTPUT_TOKENS * len(eligible), "reasoning_tokens": 0}
        return {
            "calls": rows,
            "feasibility": {"checked": len(feasibility), "infeasible": {k: v for k, v in feasibility.items() if v}},
            "estimate": {"eligible_calls": len(eligible), "approx_input_tokens": total_in,
                         "assumed_output_tokens_per_call": ESTIMATED_OUTPUT_TOKENS,
                         "batch_cost_usd": pricing.estimate_cost(self.spec.model, usage, "batch"),
                         "sync_cost_usd": pricing.estimate_cost(self.spec.model, usage, "standard"),
                         "price_table_version": pricing.PRICE_TABLE_VERSION, "note": "rough planning estimate only"},
        }

    @classmethod
    def _recorded_hash(cls, artifact, key):
        if key in cls.ARTIFACT_HASH_KEYS:
            return artifact.get(cls.ARTIFACT_HASH_KEYS[key])
        return (artifact.get("source") or {}).get(key)

    def dependency_status(self, call_ids=None):
        """{call_id: {status, changed, error}}: is each accepted artifact still built on the CURRENT upstream files?

        status: 'current' (all recorded upstream hashes equal the hashes of the files on disk now), 'stale' (some differ;
        `changed` names them), 'upstream_invalid' (an upstream file is missing or unreadable) or 'none' (not accepted).
        Hashes are re-read from disk every time: a policy that inputs are frozen is not proof that they are intact.
        """
        result = {}
        for call_id in (call_ids or self.inputs.call_ids):
            path = self.root / self.spec.artifact_path(call_id)
            if not path.is_file():
                result[call_id] = {"status": "none", "changed": [], "error": None}
                continue
            try:
                artifact = parse_json_text(path.read_bytes().decode("utf-8"))
                fresh = self._current_upstream_hashes(call_id)
            except (OSError, ValueError, KeyError) as error:
                result[call_id] = {"status": "upstream_invalid", "changed": [], "error": describe_error(error)[:300]}
                continue
            changed = sorted(key for key, value in fresh.items() if self._recorded_hash(artifact, key) != value)
            result[call_id] = {"status": "stale" if changed else "current", "changed": changed, "error": None}
        return result

    def status(self):
        rows = {}
        dependency = self.dependency_status()
        for call_id in self.inputs.call_ids:
            state = self.call_state(call_id)
            latest = state["latest"]
            held = next((a["blocked"] for a in reversed(state["in_flight"]) if a.get("blocked")), None)
            if state["in_flight"] and held:
                label = "BLOCKED_UPSTREAM_INVALID" if held["kind"] == "UPSTREAM_INVALID" else "DEFERRED_WRITE_LOCK"
            elif state["in_flight"]:
                label = f"IN_FLIGHT ({state['in_flight'][-1]['attempt_id']} {state['in_flight'][-1]['state']})"
            elif state["inconsistent"]:
                label = "INCONSISTENT"
            elif state["accepted"]:
                label = {"current": "ACCEPTED", "stale": "ACCEPTED_STALE"}.get(dependency[call_id]["status"], "ACCEPTED_UPSTREAM_INVALID")
            elif state["rejected"] >= self.spec.semantic_budget or state["transport_failed"] >= self.spec.transport_budget:
                label = "AWAITING_HUMAN"
            elif latest is None:
                label = "PENDING"
            else:
                label = latest["state"]
            rows[call_id] = {"status": label, "attempts": len(state["attempts"]), "rejected": state["rejected"],
                             "transport_failed": state["transport_failed"], "latest_attempt": latest["attempt_id"] if latest else None,
                             "dependency": dependency[call_id]["status"]}
        submissions = []
        for submission_id in self.store.submission_ids():
            sub = self.store.read_submission(submission_id)
            submissions.append({"submission_id": submission_id, "state": sub["state"], "calls": len(sub["call_ids"]),
                                "batch_id": sub["batch_id"], "last_status": sub.get("last_status"), "anomalies": len(sub["anomalies"])})
        counts = {}
        for row in rows.values():
            key = row["status"].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
        return {"calls": rows, "submissions": submissions, "counts": counts}

    # ---- sync ----------------------------------------------------------------------------------------
    def run_sync(self, call_id, *, retry_rejected=False, reopen_reason=None):
        """One paid live request for one call. Raises TransportError subclasses that are fatal; others are recorded."""
        reopen_ids = {call_id} if reopen_reason else set()
        if not self._stale_submissions():
            self._select([call_id], retry_rejected=retry_rejected, reopen_call_ids=reopen_ids)  # fail early, before credentials
        transport = self.transport()  # CredentialsError here: no attempt exists yet and nothing was sent
        with self.store.lock():
            self._recover()
            self._ensure_fresh_inputs([call_id])
            self._select([call_id], retry_rejected=retry_rejected, reopen_call_ids=reopen_ids)
            previous = self.call_state(call_id)
            reopen_of = None
            if reopen_reason:
                reopen_of = next((a["attempt_id"] for a in reversed(previous["attempts"]) if a["state"] == "ACCEPTED"), None)
            attempt, body = self._new_attempt(call_id, "sync", reopen_reason=reopen_reason, reopen_of=reopen_of)
            try:
                response = transport.create_response(body)
            except TransportError as error:
                self._fail_transport(attempt, error, possibly_billed=not error.fatal)
                if error.fatal:
                    raise
                return attempt
            self._store_response(attempt, response, {"transport": "sync"})
            return self._adjudicate(attempt)

    # ---- batch: submit -------------------------------------------------------------------------------
    def submit(self, call_ids=None, *, retry_rejected=False, reopen_reason=None):
        """Submit all eligible calls (or the given ones) as one Batch. Returns a report dict."""
        reopen_ids = set(call_ids or []) if reopen_reason else set()
        if not self._stale_submissions():
            chosen, skipped = self._select(call_ids, retry_rejected=retry_rejected, reopen_call_ids=reopen_ids)
            if not chosen:
                return {"submission_id": None, "submitted": 0, "skipped": skipped, "state": None}
        transport = self.transport()
        with self.store.lock():
            self._recover()
            self._ensure_fresh_inputs(self.inputs.call_ids)
            chosen, skipped = self._select(call_ids, retry_rejected=retry_rejected, reopen_call_ids=reopen_ids)
            if not chosen:
                return {"submission_id": None, "submitted": 0, "skipped": skipped, "state": None}
            submission_id = submission_label(self.store.next_submission_number())
            attempts, lines = [], []
            for call_id in chosen:
                reopen_of = None
                if reopen_reason:
                    reopen_of = next((a["attempt_id"] for a in reversed(self.call_state(call_id)["attempts"]) if a["state"] == "ACCEPTED"), None)
                attempt, body = self._new_attempt(call_id, "batch", submission_id=submission_id, reopen_reason=reopen_reason,
                                                  reopen_of=reopen_of)
                attempts.append(attempt)
                lines.append(json.dumps({"custom_id": attempt["custom_id"], "method": "POST", "url": "/v1/responses", "body": body},
                                        sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            jsonl = "\n".join(lines) + "\n"
            submission = {
                "schema_version": "1.0", "submission_id": submission_id, "stage": self.spec.stage, "dataset_id": self.inputs.dataset_id,
                "state": "INTENT_RECORDED", "call_ids": list(chosen),
                "entries": {a["custom_id"]: {"call_id": a["call_id"], "attempt_id": a["attempt_id"], "request_sha256": a["request_sha256"]}
                            for a in attempts},
                "input_sha256": text_hash(jsonl), "file_id": None, "batch_id": None, "last_status": None,
                "anomalies": [], "history": [{"state": "INTENT_RECORDED", "at": self.clock()}], "created_at": self.clock(),
            }
            self.store.write_input_jsonl(submission_id, jsonl)
            self.store.write_submission(submission)  # intent is durable BEFORE any network call
            return self._send_submission(transport, submission, attempts, jsonl, skipped)

    def _set_submission_state(self, submission, state, **fields):
        submission.update(fields)
        submission["state"] = state
        submission["history"].append({"state": state, "at": self.clock()})
        self.store.write_submission(submission)

    def _fail_attempts(self, attempts, error, *, fatal=None):
        for attempt in attempts:
            if attempt["state"] in IN_FLIGHT:
                self._close(attempt, "TRANSPORT_FAILED", reason=str(error),
                            fatal=bool(getattr(error, "fatal", False)) if fatal is None else fatal)

    def _send_submission(self, transport, submission, attempts, jsonl, skipped):
        submission_id = submission["submission_id"]
        report = {"submission_id": submission_id, "submitted": len(attempts), "skipped": skipped}
        try:
            file_id = transport.upload_batch_file(jsonl.encode("utf-8"), f"{submission_id}.input.jsonl")
        except Exception as error:  # noqa: BLE001 - nothing was created yet, so this is safe to retry later
            self._set_submission_state(submission, "NOT_SUBMITTED", error=str(error)[:300])
            self._fail_attempts(attempts, error)
            if getattr(error, "fatal", False):
                raise
            return {**report, "state": "NOT_SUBMITTED", "submitted": 0, "error": str(error)}
        self._set_submission_state(submission, "FILE_UPLOADED", file_id=file_id)
        metadata = {"submission_id": submission_id, "stage": self.spec.stage, "dataset_id": self.inputs.dataset_id}
        try:
            batch_id = transport.create_batch(file_id, metadata)
        except FatalRequestError as error:  # the provider rejected the request outright: definitely not created
            self._set_submission_state(submission, "NOT_SUBMITTED", error=str(error)[:300])
            self._fail_attempts(attempts, error)
            raise
        except AmbiguousError as error:
            return self._mark_unknown(submission, report, error)
        except TransportError as error:  # e.g. rate limited: definitely not created, safe to retry
            self._set_submission_state(submission, "NOT_SUBMITTED", error=str(error)[:300])
            self._fail_attempts(attempts, error)
            return {**report, "state": "NOT_SUBMITTED", "submitted": 0, "error": str(error)}
        except Exception as error:  # noqa: BLE001 - unknown outcome: protect against a duplicate paid batch
            self._mark_unknown(submission, report, error)
            raise
        self._set_submission_state(submission, "SUBMITTED", batch_id=batch_id)
        for attempt in attempts:
            attempt.update(state="SUBMITTED", batch_id=batch_id)
            self._save(attempt)
        return {**report, "state": "SUBMITTED", "batch_id": batch_id}

    def _mark_unknown(self, submission, report, error):
        self._set_submission_state(submission, "SUBMISSION_UNKNOWN", error=str(error)[:300])
        return {**report, "state": "SUBMISSION_UNKNOWN", "error": str(error),
                "advice": (f"Check the provider dashboard for a batch whose metadata submission_id is {submission['submission_id']}. "
                           f"If found: --attach-batch {submission['submission_id']} <batch_id>. If certainly absent: "
                           f"--mark-not-submitted {submission['submission_id']} --acknowledge-duplicate-risk.")}

    # ---- batch: manual recovery ----------------------------------------------------------------------
    def attach_batch(self, submission_id, batch_id):
        transport = self.transport()
        with self.store.lock():
            submission = self._require_unknown(submission_id)
            for other_id in self.store.submission_ids():
                if other_id != submission_id and self.store.read_submission(other_id)["batch_id"] == batch_id:
                    raise RunError(f"batch {batch_id} is already attached to {other_id}")
            batch = transport.retrieve_batch(batch_id)
            problems = []
            if batch.get("id") != batch_id:
                problems.append("returned batch id differs")
            if batch.get("endpoint") != "/v1/responses":
                problems.append(f"endpoint is {batch.get('endpoint')!r}, expected '/v1/responses'")
            metadata = batch.get("metadata") or {}
            for key, expected in (("submission_id", submission_id), ("stage", self.spec.stage), ("dataset_id", self.inputs.dataset_id)):
                if metadata.get(key) != expected:
                    problems.append(f"metadata {key} is {metadata.get(key)!r}, expected {expected!r}")
            if submission.get("file_id") and batch.get("input_file_id") != submission["file_id"]:
                problems.append("input_file_id does not match the file uploaded for this submission")
            counts = batch.get("request_counts")
            if counts and counts.get("total") != len(submission["entries"]):
                problems.append(f"request count {counts.get('total')} differs from {len(submission['entries'])}")
            if problems:
                raise RunError(f"refusing to attach {batch_id} to {submission_id}: " + "; ".join(problems))
            self._set_submission_state(submission, "SUBMITTED", batch_id=batch_id)
            for entry in submission["entries"].values():
                attempt = self.store.read_attempt(entry["call_id"], entry["attempt_id"])
                if attempt["state"] == "PREPARED":
                    attempt.update(state="SUBMITTED", batch_id=batch_id)
                    self._save(attempt)
            return {"submission_id": submission_id, "batch_id": batch_id, "state": "SUBMITTED"}

    def mark_not_submitted(self, submission_id, *, acknowledge_duplicate_risk=False):
        if not acknowledge_duplicate_risk:
            raise RunError("--mark-not-submitted needs --acknowledge-duplicate-risk: if the batch does exist, retrying will pay twice")
        with self.store.lock():
            submission = self._require_unknown(submission_id)
            self._set_submission_state(submission, "NOT_SUBMITTED", operator_note="marked not submitted by operator")
            for entry in submission["entries"].values():
                attempt = self.store.read_attempt(entry["call_id"], entry["attempt_id"])
                if attempt["state"] in IN_FLIGHT:
                    self._close(attempt, "TRANSPORT_FAILED", reason="marked not submitted by operator after SUBMISSION_UNKNOWN",
                                possibly_billed=True)
            return {"submission_id": submission_id, "state": "NOT_SUBMITTED"}

    def _require_unknown(self, submission_id):
        if submission_id not in self.store.submission_ids():
            raise RunError(f"unknown submission {submission_id}")
        submission = self.store.read_submission(submission_id)
        if submission["state"] != "SUBMISSION_UNKNOWN":
            raise RunError(f"{submission_id} is {submission['state']}; this operation applies only to SUBMISSION_UNKNOWN")
        return submission

    # ---- batch: collect ------------------------------------------------------------------------------
    def collect(self):
        """Poll open batches and ingest finished ones. Safe to repeat; each result line is ingested at most once."""
        report = {"collected": [], "pending": [], "skipped": [], "anomalies": [], "fatal": False}
        open_ids = [s for s in self.store.submission_ids() if self.store.read_submission(s)["state"] == "SUBMITTED"]
        responded = [a for call_id in self.inputs.call_ids for a in self.store.attempt_ids(call_id)
                     if self.store.read_attempt(call_id, a)["state"] == "RESPONDED"]
        if not open_ids and not responded and not self._stale_submissions():
            report["skipped"] = [s for s in self.store.submission_ids()]
            return report
        transport = self.transport() if open_ids else None
        with self.store.lock():
            for attempt in self._recover():
                report.setdefault("finalized", []).append((attempt["call_id"], attempt["attempt_id"], attempt["state"]))
                if attempt["state"] == "RESPONDED":
                    report["anomalies"].append(f"{attempt['call_id']} {attempt['attempt_id']}: kept unjudged - {attempt['rejection_reason']}")
            for submission_id in open_ids:
                submission = self.store.read_submission(submission_id)
                batch = transport.retrieve_batch(submission["batch_id"])
                if batch.get("id") != submission["batch_id"]:
                    report["anomalies"].append(f"{submission_id}: retrieved batch id differs from the recorded one")
                    continue
                submission["last_status"] = batch["status"]
                self.store.write_submission(submission)
                if batch["status"] not in TERMINAL_BATCH:
                    report["pending"].append(f"{submission_id} ({batch['status']})")
                    continue
                anomalies, fatal = self._ingest_batch(transport, submission, batch)
                report["collected"].append(submission_id)
                report.setdefault("outcomes", []).extend(
                    (e["call_id"], e["attempt_id"], self.store.read_attempt(e["call_id"], e["attempt_id"])["state"])
                    for e in submission["entries"].values())
                report["anomalies"] += [f"{submission_id}: {a}" for a in anomalies]
                report["fatal"] = report["fatal"] or fatal
        return report

    @staticmethod
    def _parse_lines(data, source):
        """Result lines that are JSON objects with a non-empty string custom_id; everything else is an anomaly."""
        lines, anomalies = [], []
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return [], [f"{source} file is not UTF-8"]
        for number, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
            except ValueError:
                anomalies.append(f"{source} line {number} is not valid JSON")
                continue
            if not isinstance(line, dict):
                anomalies.append(f"{source} line {number} is not a JSON object (ignored)")
            elif not isinstance(line.get("custom_id"), str) or not line["custom_id"]:
                anomalies.append(f"{source} line {number} has no string custom_id (ignored)")
            else:
                lines.append(line)
        return lines, anomalies

    def _ingest_batch(self, transport, submission, batch):
        anomalies, fatal = [], False
        submission_id, status = submission["submission_id"], batch["status"]
        entries = submission["entries"]
        attempts = {cid: self.store.read_attempt(entry["call_id"], entry["attempt_id"]) for cid, entry in entries.items()}
        if status == "failed":
            for attempt in attempts.values():
                if attempt["state"] == "SUBMITTED":
                    self._close(attempt, "TRANSPORT_FAILED", reason="batch failed at the provider (batch-level validation)", fatal=True)
            anomalies.append("batch status 'failed': a configuration or input defect; no results were produced")
            self._finish_submission(submission, batch, anomalies)
            return anomalies, True
        by_id = {}
        for file_key, source in (("output_file_id", "output"), ("error_file_id", "error")):
            if batch.get(file_key):
                data = transport.download_file(batch[file_key])
                self.store.write_result_file(submission_id, source, data)  # exact bytes, retained before parsing
                lines, problems = self._parse_lines(data, source)
                anomalies += problems
                for line in lines:
                    by_id.setdefault(line.get("custom_id"), []).append((source, line))
        for cid in sorted(c for c in by_id if c not in entries):
            anomalies.append(f"unknown custom_id {cid!r} in results (ignored)")
        for cid, found in sorted(by_id.items(), key=lambda item: str(item[0])):
            if cid in entries and len(found) > 1:
                anomalies.append(f"duplicate result lines for {cid} (none used)")
        for cid, attempt in attempts.items():
            if attempt["state"] != "SUBMITTED":
                continue  # already ingested by an earlier collect
            found = by_id.get(cid, [])
            if len(found) > 1:
                self._close(attempt, "TRANSPORT_FAILED", reason="duplicate result lines for this custom_id; none used")
            elif not found:
                self._close(attempt, "TRANSPORT_FAILED", reason=f"no result line in terminal batch (status {status})")
            else:
                fatal = self._ingest_line(submission, attempt, found[0][1], anomalies) or fatal
        pending = {cid: self.store.read_attempt(entry["call_id"], entry["attempt_id"]) for cid, entry in entries.items()}
        unresolved = sorted(cid for cid, a in pending.items() if a["state"] in IN_FLIGHT)
        if unresolved:  # keep the submission open: the next collect resumes exactly these attempts
            submission["last_status"] = status
            self.store.write_submission(submission)
            anomalies.append(f"{len(unresolved)} attempt(s) still unresolved ({', '.join(unresolved[:5])}); submission stays SUBMITTED")
            anomalies += [f"{cid}: kept unjudged - {pending[cid]['rejection_reason']}" for cid in unresolved if pending[cid].get("blocked")]
            return anomalies, fatal
        self._finish_submission(submission, batch, anomalies)
        return anomalies, fatal

    def _finish_submission(self, submission, batch, anomalies):
        submission["anomalies"] = submission["anomalies"] + anomalies
        snapshot = {key: batch.get(key) for key in ("id", "status", "endpoint", "input_file_id", "output_file_id",
                                                    "error_file_id", "request_counts", "metadata", "errors", "usage")}
        self._set_submission_state(submission, "COLLECTED", last_status=batch["status"], batch=snapshot)

    @staticmethod
    def _as_dict(value):
        return value if isinstance(value, dict) else {}

    def _ingest_line(self, submission, attempt, line, anomalies):
        """Ingest one result line for a SUBMITTED attempt; returns True when a fatal configuration error was found."""
        response, error = line.get("response"), line.get("error")
        if response is not None and not isinstance(response, dict) or (error and not isinstance(error, dict)):
            anomalies.append(f"{attempt['custom_id']}: malformed result line (result not used)")
            self._close(attempt, "TRANSPORT_FAILED", reason="malformed result line")
            return False
        status_code = self._as_dict(response).get("status_code")
        if not error and response is not None and status_code == 200 and not isinstance(response.get("body"), dict):
            anomalies.append(f"{attempt['custom_id']}: result line has no response body object (result not used)")
            self._close(attempt, "TRANSPORT_FAILED", reason="malformed result body")
            return False
        if error or response is None or status_code != 200:
            body_error = self._as_dict(self._as_dict(self._as_dict(response).get("body")).get("error"))
            code = self._as_dict(error).get("code") or body_error.get("code")
            code = code if isinstance(code, str) else None
            status_code = status_code if isinstance(status_code, int) else None
            if code in QUOTA_CODES:
                self._close(attempt, "TRANSPORT_FAILED", reason=f"batch request failed: {code} (no credits/quota)", fatal=True)
                anomalies.append(f"{attempt['custom_id']}: provider reports exhausted credits/quota ({code})")
                return True
            if code == "batch_expired" or is_transient_error_code(code) or status_code in (None, 408, 409, 429) or (status_code or 0) >= 500:
                self._close(attempt, "TRANSPORT_FAILED", reason=f"batch request failed: {code or status_code}")
                return False
            self._close(attempt, "TRANSPORT_FAILED", reason=f"batch request rejected ({status_code} {code})", fatal=True)
            anomalies.append(f"{attempt['custom_id']}: permanent request error {status_code} {code}")
            return True
        if not self._request_hash_matches(submission, attempt):
            anomalies.append(f"{attempt['custom_id']}: local request hash mismatch (result not used)")
            self._close(attempt, "REJECTED", reason="local request hash mismatch; the persisted request no longer matches what was sent")
            return False
        self._store_response(attempt, response["body"], {"transport": "batch", "custom_id": attempt["custom_id"],
                                                         "batch_id": submission["batch_id"], "request_id": response.get("request_id")})
        self._adjudicate(attempt)
        return False

    def _request_hash_matches(self, submission, attempt):
        """The provider does not echo our hash: recompute it from the locally persisted request and the submitted line."""
        call_id, attempt_id, cid = attempt["call_id"], attempt["attempt_id"], attempt["custom_id"]
        local = self.store.read_request(call_id, attempt_id)
        if local is None:
            return False
        expected = submission["entries"][cid]["request_sha256"]
        if not (text_hash(local) == attempt["request_sha256"] == expected):
            return False
        for raw in self.store.read_input_jsonl(submission["submission_id"]).splitlines():
            line = json.loads(raw)
            if line["custom_id"] == cid:
                return text_hash(canonical_request_text(line["body"])) == expected
        return False

    # ---- validate / check ----------------------------------------------------------------------------
    def _rebuild_text(self, call_id, entry):
        """Artifact text compiled from an accepted (or historical) provenance entry and the current frozen inputs."""
        generation = {key: entry[key] for key in ("attempt_id", "model", "prompt_template_sha256", "response_schema_sha256")}
        compile_for_entry, ctx = self._entry_context(call_id, entry)
        return serialize_json(compile_for_entry(ctx, entry["parsed_response"], generation))

    def regenerate(self):
        """{relative path: text} rebuilt from provenance and frozen inputs only (never from provider envelopes).

        Calls whose provenance cannot be used are skipped and listed in `self.rebuild_problems`; the others still rebuild.
        """
        texts, artifact_texts, self.rebuild_problems = {}, {}, {}
        provs = self._provenance_map()
        for call_id in self.inputs.call_ids:
            if call_id not in provs:
                self.rebuild_problems[call_id] = self._prov_state(call_id)[1] or "provenance unusable"
                continue
            accepted = provs[call_id]["accepted"]
            if accepted is None:
                continue
            try:
                text = self._rebuild_text(call_id, accepted)
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                self.rebuild_problems[call_id] = f"cannot rebuild the accepted artifact: {error!r}"[:200]
                continue
            texts[self.spec.artifact_path(call_id)] = text
            artifact_texts[call_id] = text
        if artifact_texts:
            texts[self.spec.record_path] = serialize_json(self._build_record(artifact_texts, provs))
        return texts

    def _disk_texts(self):
        disk = {}
        for relative in [*(self.spec.artifact_path(c) for c in self.inputs.call_ids), self.spec.record_path]:
            path = self.root / relative
            if path.is_file() and not path.is_symlink():
                disk[relative] = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        directory = self.root / self.spec.artifact_dir
        if directory.is_dir():
            for entry in directory.iterdir():
                relative = f"{self.spec.artifact_dir}/{entry.name}"
                if relative not in disk:
                    disk[relative] = None  # extra, unreadable or symlinked
        record_tmp = self.root / (self.spec.record_path + ".tmp")
        if record_tmp.exists():
            disk[self.spec.record_path + ".tmp"] = None
        return disk

    def check(self):
        """Rebuild artifacts + record from provenance and compare with disk byte for byte."""
        expected, disk = self.regenerate(), self._disk_texts()
        unusable = {self.spec.artifact_path(call_id) for call_id in self.rebuild_problems}
        result = {
            "missing": sorted(path for path in expected if path not in disk),
            "changed": sorted(path for path, text in expected.items() if path in disk and disk[path] != text),
            "extra": sorted(path for path in disk if path not in expected and path not in unusable),
        }
        if self.rebuild_problems:
            result["unreproducible"] = sorted(f"{call_id}: {why}" for call_id, why in self.rebuild_problems.items())
        return result

    def validate(self, *, require_complete=False):
        """Independent validation of what is on disk. Returns a list of error strings ([] when valid)."""
        from jsonschema import Draft202012Validator

        from fyp_evaluation.validation.schemas import load_json

        errors = []
        schemas = {role: Draft202012Validator(load_json(self.schema_dir / self.spec.schema_subdir / f"{name}.schema.json"))
                   for role, name in (("artifact", self.spec.artifact_schema), ("record", self.spec.record_schema),
                                      ("provenance", self.spec.provenance_schema))}
        inputs = self.inputs
        directory = self.root / self.spec.provenance_dir
        known = {f"{c}.json" for c in inputs.call_ids}
        if directory.is_dir():
            for entry in sorted(directory.iterdir()):
                if entry.name not in known:
                    errors.append(f"unexpected file {self.spec.provenance_dir}/{entry.name}")
        for call_id in inputs.call_ids:
            errors += self._validate_call(call_id, schemas)
        difference = self.check()
        errors += [f"missing {p}" for p in difference["missing"]]
        errors += [f"differs from the rebuild: {p}" for p in difference["changed"]]
        errors += [f"unexpected file {p}" + ("; orphaned temporary file from an interrupted run" if p.endswith(".tmp") else "")
                   for p in difference["extra"]]
        errors += [f"cannot reproduce {item}" for item in difference.get("unreproducible", [])]
        if (self.root / self.spec.record_path).is_file():
            record = parse_json_text((self.root / self.spec.record_path).read_text(encoding="utf-8"))
            errors += [f"{self.spec.record_path}: {e.json_path}: {e.message}" for e in schemas["record"].iter_errors(record)]
        if require_complete:
            errors += [f"{c}: no accepted {self.spec.noun}" for c in inputs.call_ids if not (self.root / self.spec.artifact_path(c)).is_file()]
        return errors

    def _validate_call(self, call_id, schemas):
        errors = []
        inputs, root = self.inputs, self.root
        prov_file = root / self.spec.provenance_path(call_id)
        artifact_file = root / self.spec.artifact_path(call_id)
        has_prov, has_artifact = prov_file.is_file(), artifact_file.is_file()
        if not has_prov and not has_artifact:
            return errors
        prov, problem = self._prov_state(call_id)
        if problem:  # unreadable or schema-invalid (including deleted required fields): report and move on to the next call
            return [f"{call_id}: provenance {problem}"]
        if has_prov:
            text = prov_file.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text != serialize_json(prov):
                errors.append(f"{call_id}: provenance is not in canonical form")
        accepted = prov["accepted"]
        if has_artifact != (accepted is not None):
            errors.append(f"{call_id}: artifact and provenance disagree about acceptance")
        if accepted is None:
            return errors
        schema = self.builder.response_schema
        for label, entry in [("accepted", accepted)] + [(f"history[{i}]", h) for i, h in enumerate(prov["accepted_history"])]:
            try:
                if response_hash(entry["parsed_response"], schema) != entry["parsed_response_sha256"]:
                    errors.append(f"{call_id}: {label} parsed_response_sha256 does not match its parsed response")
                # historical versions have no file on disk: rebuild them and check the recorded artifact hash
                elif text_hash(self._rebuild_text(call_id, entry)) != entry["artifact_sha256"]:
                    errors.append(f"{call_id}: {label} artifact_sha256 cannot be reproduced from its stored parsed response "
                                  "and generation metadata under the upstream version it was built against")
                _, entry_ctx = self._entry_context(call_id, entry)  # response rules are checked against that same version
                errors += [f"{call_id}: {label} response: {e}" for e in self.validator.errors(entry["parsed_response"], entry_ctx)]
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                errors.append(f"{call_id}: {label} cannot be rebuilt ({error!r})"[:700])
        if has_artifact:
            text = artifact_file.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text_hash(text) != accepted["artifact_sha256"]:
                errors.append(f"{call_id}: artifact hash differs from provenance (edited or stale)")
            try:
                artifact = parse_json_text(text)
            except ValueError as error:
                return errors + [f"{call_id}: artifact unreadable: {error}"]
            errors += [f"{call_id}: artifact {e.json_path}: {e.message}" for e in schemas["artifact"].iter_errors(artifact)]
            for key, current in self._staleness(call_id).items():
                if artifact.get(key) != current:
                    errors.append(f"{call_id}: stale {key}: upstream changed since this {self.spec.label} was accepted")
        return errors
