"""Prompt rendering, scenario brief and the canonical Responses API request body.

One `RequestBuilder` serves both transports: sync sends the body, batch wraps the same body in a JSONL line.
Nothing here touches the network or the attempt store.
"""

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.language_brief import language_brief
from fyp_evaluation.llm.store import canonical_request_text
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .config import (
    MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, REASONING_EFFORT, STAGE, prompt_template, response_format_name, response_schema_name,
)
from .slots import CallContext


def response_schema_path(schema_dir=DEFAULT_SCHEMA_DIR, documents=None):
    """The response-schema file; `documents` selects the schema-2.0 file for a multilingual release."""
    stem = response_schema_name(documents) if documents is not None else "call_config_response"
    return schema_dir / "stage3" / f"{stem}.schema.json"


def request_hash(body):
    return g1.sha256(canonical_request_text(body).encode("utf-8"))


def _checklist_text(documents):
    semantics = {entry["question_id"]: entry for entry in documents["check_semantics"]["checks"]}
    blocks = []
    for item in documents["checklist"]:
        entry = semantics[item["id"]]
        lines = [f"- {item['id']} ({item['type']}): {item['question']}"]
        if item["type"] == "COMPLIANCE":
            lines += [f"    PASS when: {entry['pass_condition']}", f"    FAIL when: {entry['fail_condition']}"]
            lines += [f"    Note: {note}" for note in entry.get("notes", [])]
        else:
            lines.append(f"    Resolution: {entry['resolution_rule']}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _rules_text(documents):
    semantics = documents["check_semantics"]
    lines = [f"- {rule['id']}: {rule['reason']}" for rule in semantics["ordering_rules"]]
    lines += [f"- {rule['id']}: {rule['reason']}" for rule in semantics["cross_check_rules"]]
    return "\n".join(lines)


IDENTITY_QUESTIONS = {"K01", "K02"}
WITHHELD = {"withheld": "client identity values (NRIC/FIN, date of birth) are never part of the scenario brief"}


def _brief_parameters(edge):
    """Edge parameters shown to the model; those that carry client identity gold are withheld."""
    return WITHHELD if IDENTITY_QUESTIONS & set(edge["target_question_ids"]) else edge["parameters"]


def build_brief(inputs, ctx, *, include_language_pins=False):
    """The per-call user message (a plain dict; serialised canonically by the caller)."""
    catalog = {entry["id"]: entry for entry in inputs.documents["edge_case_catalog"]["edge_cases"]}
    meta = ctx.spec["metadata"]
    if ctx.multilingual:  # schema 2.0: the language block replaces the pilot's single `language` value
        header = {"call_id": ctx.call_id, "language": language_brief(ctx.spec), "required_name_languages": ctx.name_languages,
                  "metadata": {"duration_profile": meta["duration_profile"], "target_duration_min_seconds": meta["target_duration_min_seconds"],
                               "target_duration_max_seconds": meta["target_duration_max_seconds"]}}
    else:
        header = {"call_id": ctx.call_id, "metadata": {
            "language": meta["language"], "duration_profile": meta["duration_profile"],
            "target_duration_min_seconds": meta["target_duration_min_seconds"],
            "target_duration_max_seconds": meta["target_duration_max_seconds"],
        }}
    brief = {
        **header,
        "fixed_verdicts": {q: ctx.verdict(q) for q in ctx.gold if q.startswith("C")},
        "fixed_transaction": {q: ctx.value(q) for q in ("T01", "T02", "T03", "T04", "T05", "T06")},
        "edge_assignments": [
            {"assignment_id": e["assignment_id"], "edge_case_id": e["edge_case_id"], "family": e["family"],
             "description": catalog[e["edge_case_id"]]["description"], "target_question_ids": e["target_question_ids"],
             "parameters": _brief_parameters(e), **({"language_variant": e["language_variant"]} if ctx.multilingual else {})}
            for e in ctx.edges if e["assignment_id"] in ctx.edge_context_ids],
        "products": ctx.products,
        "allowed_currencies": ctx.allowed_currencies,
        "required_slots": ctx.required_slots(),
    }
    if include_language_pins and ctx.multilingual:
        for edge in brief["edge_assignments"]:
            case = catalog[edge["edge_case_id"]]
            variant = next((v for v in case.get("language_variants", [])
                            if v["variant_id"] == edge["language_variant"]), None)
            if variant is not None:
                edge["language_variant_details"] = {
                    key: variant[key] for key in ("event_languages", "purpose") if key in variant}
    return brief


class RequestBuilder:
    """Builds the identical request body for any call; computes the template/schema hashes once."""

    def __init__(self, inputs, schema_dir=DEFAULT_SCHEMA_DIR, prompt_dir=PROMPT_DIR, *, prompt_template_name=None):
        self.inputs = inputs
        documents = inputs.documents
        template_name = prompt_template_name or prompt_template(documents)
        if template_name not in {"call_config.v1.md", "call_config.v2.md", "call_config.v3.md"}:
            raise ValueError("Unknown G3 prompt template")
        self.include_language_pins = template_name == "call_config.v3.md"
        template_path = prompt_dir / template_name
        self.prompt_template_sha256 = g1.sha256(template_path.read_bytes())
        self.response_format_name = response_format_name(documents)
        schema_path = response_schema_path(schema_dir, documents)
        self.response_schema_sha256 = g1.sha256(schema_path.read_bytes())
        self.response_schema = parse_json_text(schema_path.read_text(encoding="utf-8"))
        template = template_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.developer_message = (template.replace("{{CHECKLIST}}", _checklist_text(inputs.documents))
                                  .replace("{{RULES}}", _rules_text(inputs.documents)))
        self._contexts = {}

    def context(self, call_id):
        if call_id not in self._contexts:
            self._contexts[call_id] = CallContext(self.inputs, call_id)
        return self._contexts[call_id]

    def build(self, call_id, attempt_id):
        """The canonical Responses API body for one attempt. Identical for sync and batch."""
        ctx = self.context(call_id)
        return {
            "model": MODEL,
            "input": [
                {"role": "developer", "content": self.developer_message},
                {"role": "user", "content": serialize_json(build_brief(self.inputs, ctx, include_language_pins=self.include_language_pins))},
            ],
            "text": {"format": {"type": "json_schema", "name": self.response_format_name, "strict": True, "schema": self.response_schema}},
            "reasoning": {"effort": REASONING_EFFORT},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "metadata": {"dataset_id": self.inputs.dataset_id, "call_id": call_id, "stage": STAGE, "attempt_id": attempt_id},
        }
