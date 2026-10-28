"""G7 verification context, the per-call brief, the stable system prefix and the canonical Interactions request body.

The brief reuses G5's (frozen gold, applicable condition text, accepted G3 scenario and products, assigned edge cases with
catalog semantics, every planned G4 event with its relations, values and constraints) and adds the ENTIRE actual dialogue:
every integer turn ID, speaker and text, plus the turn each planned event landed in. It carries no hashes, provenance,
provider envelopes, character offsets, G5 verdicts or rationales. The system prefix additionally carries the release-local
normalization profiles (aliases, permitted spoken forms and their assignment-dependent restrictions). The request body
contains no attempt ID, so a dry-run hash equals the hash of the request that would be sent.
"""

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.dialogues.advisory import advisory_recommendation
from fyp_evaluation.plan_verifications.request import VerificationContext, compact_json
from fyp_evaluation.plan_verifications.request import build_brief as build_plan_brief
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .config import MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, SERVICE_TIER, THINKING_LEVEL, prompt_template, response_schema_name, schema_names
from .language_rows import rows_for


def schema_path(name, schema_dir=DEFAULT_SCHEMA_DIR):
    return schema_dir / "stage7" / f"{name}.schema.json"


class DialogueVerificationContext(VerificationContext):
    """Everything one call's validation and compilation need, bound to ONE version of its G3 config, G4 plan and G6 dialogue."""

    def __init__(self, inputs, call_id):
        super().__init__(inputs, call_id)
        self.dialogue = inputs.dialogues[call_id]
        self.turns = self.dialogue["turns"]
        self.alignment = self.dialogue["event_alignment"]
        self.spans_by_turn = {}
        for span in self.alignment:
            self.spans_by_turn.setdefault(span["turn_id"], []).append(span)
        self.turn_of_event = {span["event_id"]: span["turn_id"] for span in self.alignment}
        self.edge_cases = {edge["assignment_id"]: edge for edge in self.spec["edge_cases"]}
        self.events_of_assignment = {a: [e["event_id"] for e in self.events if a in e["assignment_ids"]] for a in self.assignment_ids}
        self.language_rows = rows_for(self.spec) if self.multilingual else []
        self.language_row_ids = [row["row_id"] for row in self.language_rows]


def build_brief(inputs, ctx):
    """The per-call input (a plain dict; serialised canonically by the caller)."""
    brief = build_plan_brief(inputs, ctx)
    for entry in brief["events"]:
        entry["turn_id"] = ctx.turn_of_event[entry["event_id"]]
    brief["dialogue"] = [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"]} for t in ctx.turns]
    if ctx.multilingual:
        brief["language_rows"] = ctx.language_rows
    expectation = advisory_expectation(ctx)
    if expectation is not None:  # only the calls whose plan has no planned recommendation; every other brief is unchanged
        brief["advisory_expectation"] = expectation
    return brief


def advisory_expectation(ctx):
    """The G6 `advisory_recommendation` data of this call (the same derivation the author was given) plus the turn of every named event."""
    data = advisory_recommendation(ctx.spec, ctx.config, ctx.events, None, ctx.inputs.documents)
    if data is None:
        return None
    turns = ctx.turn_of_event
    return {**data, "place_after_turn_ids": sorted({turns[e] for e in data["place_after_event_ids"]}),
            "place_before_turn_ids": sorted({turns[e] for e in data["place_before_event_ids"]}),
            "keep_apart_turn_ids": sorted({turns[e] for e in data["keep_apart_from_event_ids"]})}


def system_prefix(template, documents):
    """The stable instruction shared by every call (checklist, semantics and normalization inserted verbatim as canonical JSON)."""
    semantics = documents["check_semantics"]
    return (template.replace("{{CHECKLIST}}", compact_json(documents["checklist"]))
            .replace("{{SEMANTICS}}", compact_json({key: semantics[key] for key in ("checks", "ordering_rules", "cross_check_rules")}))
            .replace("{{NORMALIZATION}}", compact_json(documents["normalization"]["profiles"])))


class RequestBuilder:
    """Builds the identical request body for any call; computes the template/schema hashes once."""

    def __init__(self, inputs, schema_dir=DEFAULT_SCHEMA_DIR, prompt_dir=PROMPT_DIR, *, prompt_template_name=None, tier=SERVICE_TIER):
        self.tier = tier  # "flex" (synchronous) or "batch" (Gemini Batch); the only difference between the two request bodies
        self.inputs = inputs
        template_name = prompt_template_name or prompt_template(inputs.documents)
        if template_name not in {"dialogue_verification.v1.md", "dialogue_verification.v2.md", "dialogue_verification.v3.md"}:
            raise ValueError("Unknown G7 prompt template")
        template_path = prompt_dir / template_name
        self.prompt_template_sha256 = g1.sha256(template_path.read_bytes())
        response_path = schema_path(response_schema_name(inputs.documents), schema_dir)
        self.response_schema_sha256 = g1.sha256(response_path.read_bytes())
        self.response_schema = parse_json_text(response_path.read_text(encoding="utf-8"))
        self.artifact_schema = parse_json_text(schema_path(schema_names(inputs.documents)[0], schema_dir).read_text(encoding="utf-8"))
        self.system_instruction = system_prefix(template_path.read_text(encoding="utf-8").replace("\r\n", "\n"), inputs.documents)
        self._contexts = {}

    def context(self, call_id):
        if call_id not in self._contexts:
            self._contexts[call_id] = DialogueVerificationContext(self.inputs, call_id)
        return self._contexts[call_id]

    def brief_text(self, call_id):
        return compact_json(build_brief(self.inputs, self.context(call_id)))

    def build(self, call_id, attempt_id):  # noqa: ARG002 - the attempt ID is deliberately not part of the request
        return {
            "model": MODEL,
            "system_instruction": self.system_instruction,
            "input": self.brief_text(call_id),
            "generation_config": {"thinking_level": THINKING_LEVEL, "max_output_tokens": MAX_OUTPUT_TOKENS},
            "response_format": {"type": "text", "mime_type": "application/json", "schema": self.response_schema},
            "service_tier": self.tier,
            "store": False,
        }
