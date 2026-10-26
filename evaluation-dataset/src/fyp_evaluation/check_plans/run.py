"""G4 library API: `CheckPlanRunner` is the G4 specialisation of the shared `StageRunner` lifecycle.

It supplies the G4 paths, schemas, settings and hooks. Eligibility, attempts, the sync and Batch flows, recovery,
provenance, accepted history, `--validate` and `--check` are the shared, G3-proven implementation.
"""

from fyp_evaluation import register_guide
from fyp_evaluation.call_configs.inputs import fresh_upstream_hashes
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.llm.stage_runner import RunError, StageRunner, StageSpec  # noqa: F401  (re-exported)

from .compile import build_record, compile_artifact
from .context import PlanContext
from .config import (
    ARTIFACT_DIR, MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, PROVENANCE_DIR, REASONING_EFFORT, RECORD, SEMANTIC_BUDGET, STAGE,
    TRANSPORT_BUDGET, schema_names,
)
from .feasibility import feasibility_report, make_validator
from .inputs import HistoricalContextError, inputs_with_config, load_inputs
from .request import RequestBuilder

G4_SPEC = StageSpec(
    stage=STAGE, label="plan", noun="check plan", artifact_dir=ARTIFACT_DIR, provenance_dir=PROVENANCE_DIR, record_path=RECORD,
    schema_subdir="stage4", artifact_schema="check_plan", record_schema="check_plan_record",
    provenance_schema="check_plan_provenance", prompt_dir=PROMPT_DIR, model=MODEL, reasoning_effort=REASONING_EFFORT,
    max_output_tokens=MAX_OUTPUT_TOKENS, semantic_budget=SEMANTIC_BUDGET, transport_budget=TRANSPORT_BUDGET)


class CheckPlanRunner(StageRunner):
    spec = G4_SPEC
    supports_selection = True  # the shared CLI passes --call-id so only the selected calls need an accepted G3 config

    def __init__(self, root, *, selected_call_ids=None, **kwargs):
        super().__init__(root, **kwargs)
        self.selected_call_ids = list(selected_call_ids) if selected_call_ids else None

    def _load_inputs(self):
        if self.selected_call_ids:
            return load_inputs(self.root, self.schema_dir, self.selected_call_ids)
        return load_inputs(self.root, self.schema_dir)

    def _schema_names(self):
        artifact, record = schema_names(self.inputs.documents)
        return artifact, record, self.spec.provenance_schema

    def eligibility(self, call_id, **kwargs):
        if call_id not in self.inputs.configs:  # selected-call mode: no accepted G3 config was required for this call
            return False, "no accepted G3 call config for this call"
        return super().eligibility(call_id, **kwargs)

    def _make_builder(self):
        return RequestBuilder(self.inputs, self.schema_dir, self.prompt_dir)

    def _make_validator(self):
        return make_validator(self.builder)

    def _feasibility(self):
        return feasibility_report(self.builder, self.validator, self.inputs)

    def _compile_artifact(self, ctx, parsed, generation):
        return compile_artifact(self.inputs, ctx, parsed, generation, self.builder.response_schema)

    def _build_record(self, artifact_texts, provenance_by_call):
        return build_record(self.inputs, artifact_texts, provenance_by_call)

    def _upstream_hashes(self, call_id):
        inputs = self.inputs
        return {"call_spec": inputs.spec_hashes[call_id], "check_plan_skeleton": inputs.skeleton_hashes[call_id],
                "call_config": inputs.config_hashes[call_id], **inputs.source_hashes(), **register_guide.extra_hashes(inputs.specs[call_id])}

    def _current_upstream_hashes(self, call_id):
        hashes = fresh_upstream_hashes(self.inputs.root, call_id)
        hashes["call_config"] = g1.sha256((self.inputs.root / "authoring/call_configs" / f"{call_id}.json").read_bytes())
        hashes.update(register_guide.extra_hashes(self.inputs.specs[call_id]))  # re-read now, SINGLISH calls only
        return hashes

    # ---- version-matched context for historical / stale plans -------------------------------------------
    def _committed_config_hash(self, call_id, entry):
        """The G3 config hash an accepted/historical G4 version was built from, read from its COMMITTED attempt summary."""
        prov, problem = self._prov_state(call_id)
        if problem:
            raise HistoricalContextError(f"{call_id}: provenance is {problem}")
        summary = next((a for a in prov["attempts"] if a["attempt_id"] == entry["attempt_id"]), None)
        recorded = ((summary or {}).get("upstream_sha256") or {}).get("call_config")
        if not recorded:
            raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: no committed attempt summary records the G3 config it was built from")
        accepted = prov["accepted"]
        artifact_file = self.root / self.spec.artifact_path(call_id)
        if accepted and accepted["attempt_id"] == entry["attempt_id"] and artifact_file.is_file():
            on_disk = parse_json_text(artifact_file.read_bytes().decode("utf-8")).get("call_config_sha256")
            if on_disk != recorded:
                raise HistoricalContextError(f"{call_id}/{entry['attempt_id']}: the artifact and its committed attempt summary disagree "
                                             "about the G3 config version")
        return recorded

    def _inputs_for_config(self, call_id, config_hash):
        """The G4 inputs in which this call's G3 config has the given hash: the current ones, or a rebuilt historical version."""
        if self.inputs.config_hashes[call_id] == config_hash:
            return self.inputs
        cache = self.__dict__.setdefault("_historical_inputs", {})
        if (call_id, config_hash) not in cache:
            g3 = ConfigRunner(self.root, schema_dir=self.schema_dir, inputs=self.inputs.upstream)
            g3_prov, problem = g3._prov_state(call_id)
            if problem:
                raise HistoricalContextError(f"{call_id}: G3 provenance is {problem}")
            problems = []
            for entry in [g3_prov["accepted"], *g3_prov["accepted_history"]]:
                if entry is None:
                    continue
                try:
                    text = g3._rebuild_text(call_id, entry)
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    problems.append(f"{entry.get('attempt_id')}: cannot be rebuilt ({error!r})"[:160])
                    continue
                if text_hash(text) != entry["artifact_sha256"]:
                    problems.append(f"{entry['attempt_id']}: rebuilt hash differs from its recorded artifact_sha256 (corrupted history)")
                    continue
                if text_hash(text) == config_hash:
                    cache[(call_id, config_hash)] = inputs_with_config(self.inputs, call_id, text)
                    break
            else:
                raise HistoricalContextError(
                    f"{call_id}: the G3 config version {config_hash[:12]}... this G4 plan was built from cannot be reconstructed from G3's "
                    "accepted history" + (f" ({'; '.join(problems)})" if problems else ""))
        return cache[(call_id, config_hash)]

    def _entry_context(self, call_id, entry):
        inputs = self._inputs_for_config(call_id, self._committed_config_hash(call_id, entry))

        def compile_for_entry(ctx, parsed, generation):
            return compile_artifact(inputs, ctx, parsed, generation, self.builder.response_schema)
        return compile_for_entry, (self.builder.context(call_id) if inputs is self.inputs else PlanContext(inputs, call_id))

    def _staleness(self, call_id):
        return {"call_spec_sha256": self.inputs.spec_hashes[call_id],
                "check_plan_skeleton_sha256": self.inputs.skeleton_hashes[call_id],
                "call_config_sha256": self.inputs.config_hashes[call_id]}
