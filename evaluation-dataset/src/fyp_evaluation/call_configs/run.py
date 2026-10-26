"""G3 library API: everything the CLI (and a future end-to-end runner) does, importable without a subprocess.

`ConfigRunner` is the G3 specialisation of the shared `StageRunner` lifecycle (fyp_evaluation.llm.stage_runner): it
supplies the G3 paths, schemas, settings and the handful of hooks (inputs, request builder, validator, compiler,
record, upstream hashes). Transports are injected (`OpenAITransport` for real runs, `FakeTransport` in tests); the request
builder, parser, validator and compiler are the same on the sync and Batch paths.
"""

from fyp_evaluation.llm.stage_runner import RunError, StageRunner, StageSpec, utc_now  # noqa: F401  (re-exported)

from .compile import build_record, compile_artifact
from .config import (
    CONFIG_DIR, MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, PROVENANCE_DIR, REASONING_EFFORT, RECORD, SEMANTIC_BUDGET, STAGE,
    TRANSPORT_BUDGET, schema_names,
)
from .feasibility import feasibility_report, make_validator
from .inputs import fresh_upstream_hashes, load_inputs
from .request import RequestBuilder

G3_SPEC = StageSpec(
    stage=STAGE, label="config", noun="call config", artifact_dir=CONFIG_DIR, provenance_dir=PROVENANCE_DIR, record_path=RECORD,
    schema_subdir="stage3", artifact_schema="call_config", record_schema="call_config_record",
    provenance_schema="call_config_provenance", prompt_dir=PROMPT_DIR, model=MODEL, reasoning_effort=REASONING_EFFORT,
    max_output_tokens=MAX_OUTPUT_TOKENS, semantic_budget=SEMANTIC_BUDGET, transport_budget=TRANSPORT_BUDGET)


class ConfigRunner(StageRunner):
    spec = G3_SPEC

    def _load_inputs(self):
        return load_inputs(self.root, self.schema_dir)

    def _schema_names(self):
        artifact, record = schema_names(self.inputs.documents)
        return artifact, record, self.spec.provenance_schema

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
                **inputs.source_hashes()}

    def _current_upstream_hashes(self, call_id):
        return fresh_upstream_hashes(self.inputs.root, call_id)

    def _staleness(self, call_id):
        return {"call_spec_sha256": self.inputs.spec_hashes[call_id],
                "check_plan_skeleton_sha256": self.inputs.skeleton_hashes[call_id]}
