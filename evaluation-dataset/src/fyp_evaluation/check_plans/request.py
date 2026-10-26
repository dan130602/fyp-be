"""G4 prompt rendering, per-call brief and the canonical Responses API request body (identical for sync and Batch)."""

import json

from fyp_evaluation.call_configs.request import _checklist_text, _rules_text
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.llm.store import canonical_request_text
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from fyp_evaluation import register_guide
from fyp_evaluation.language_brief import language_brief

from .analysis import question_surface, scenario_surface
from .config import (
    MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, REASONING_EFFORT, RESPONSE_SCHEMA, STAGE, prompt_template, response_format_name, schema_names,
)
from .context import PlanContext, forms_of


def schema_path(name, schema_dir=DEFAULT_SCHEMA_DIR):
    return schema_dir / "stage4" / f"{name}.schema.json"


def request_hash(body):
    return g1.sha256(canonical_request_text(body).encode("utf-8"))


def _surfaces(ctx, event, placeholder):
    """{form: rendered surface} a question placeholder can take in this event (previews for the model).

    Schema 2.0 reads the same `ctx.choices` the validator and the renderer use, and labels each form with its language."""
    if ctx.multilingual:
        choices = ctx.choices(event, placeholder)
        return {choice.form: {"language": choice.language, "surface": choice.surface} for choice in choices} if choices else None
    binding = ctx.binding(event, placeholder)
    forms = forms_of(binding[1]) if binding else None
    if not forms:
        return None
    return {form: question_surface(ctx, placeholder, binding[2], form, binding[1]) for form in forms}


def _event_brief(ctx, event):
    rules = ctx.slot_rules(event)
    entry = {
        "event_id": event["event_id"], "kind": event["kind"], "status": event["status"],
        "allowed_speakers": event["allowed_speakers"],
        "speaker_rule": ("choose one speaker" if len(event["allowed_speakers"]) > 1 else "speaker is fixed: use null"),
        "question_ids": event["question_ids"],
    }
    for key in ("role", "distractor_role", "currency", "incomplete", "strength", "corrects_event_id", "invalidates_event_id",
                "refers_to_event_id"):
        if key in event:
            entry[key] = event[key]
    if ctx.multilingual:
        if "required_language" in event:
            entry["required_language"] = event["required_language"]
        entry["spoken_languages"] = sorted(ctx.allowed_languages(event))
    if event["kind"] == "COMPLIANCE_ACT":
        question = event["question_ids"][0]
        ref = event.get("condition_ref", "")
        text = ctx.semantics[question].get(ref.split(".")[-1]) if ref else None
        entry["condition"] = {"check": question, "ref": ref, "meaning": text}
        if ctx.checks[question].get("wording_constraint"):
            entry["wording_constraint"] = "paraphrase: do not copy the checklist wording"
    if rules.readback:
        readback = event["readback"]
        wrong = readback["wrong_question_id"]
        selectable = {q: _surfaces(ctx, event, q) for q in ctx.readback_candidates(event)}
        entry["readback"] = {
            "must_include": readback["must_include_question_ids"], "min_field_count": readback["min_field_count"],
            "wrong_question_id": wrong, "wrong_value_preview": _surfaces(ctx, event, wrong),
            "selectable_fields": selectable,
            "note": ("list the fields you read back in readback_question_ids and write one [[Q]] per field; for every chosen field "
                     "that appears in form_choices, add one value_forms entry choosing one of its listed forms"),
        }
        choices = {q: list(surfaces) for q, surfaces in selectable.items() if surfaces and len(surfaces) > 1}
        if choices:
            entry["form_choices"] = choices
        return entry
    placeholders = {"required": list(rules.required), "optional": list(rules.optional)}
    entry["placeholders"] = placeholders
    previews, choices = {}, {}
    for name in rules.required + rules.optional:
        if name in ctx.checks or ctx.multilingual:
            surfaces = _surfaces(ctx, event, name)
            if surfaces:
                previews[name] = surfaces
                if len(surfaces) > 1:
                    choices[name] = list(surfaces)
        else:
            previews[name] = scenario_surface(ctx, event, name)[0]
    if previews:
        entry["rendered_surfaces"] = previews
    if choices:
        entry["form_choices"] = choices
    return entry


def _product_brief(ctx, product):
    entry = {"product_ref": product["product_ref"], "name": product["name"], "roles": product["roles"]}
    if ctx.multilingual:
        names = (ctx.inputs.documents["normalization"]["profiles"]["synthetic_product_name"].get("language_names") or {}).get(product["name"], {})
        if names.get("zh", {}).get("status") == "APPROVED":
            entry["approved_chinese_name"] = names["zh"]["surface"]
    return entry


def build_brief(inputs, ctx):
    scenario = ctx.scenario
    catalog = {e["id"]: e for e in inputs.documents["edge_case_catalog"]["edge_cases"]}
    motivations = {item["assignment_id"]: item["motivation"] for item in scenario["edge_case_context"]}
    checks = []
    for check in ctx.skeleton["checks"]:
        row = {"question_id": check["question_id"], "kind": check["kind"], "evidence_mode": check["evidence_mode"],
               "evidence_event_ids": check["evidence_event_ids"]}
        if check["kind"] == "COMPLIANCE":
            row["verdict"] = check["expected"]["expected_verdict"]
            if check["evidence_mode"] == "ABSENCE":
                ref = check["forbidden_condition_ref"]
                row["must_not_be_expressed_anywhere"] = ctx.semantics[check["question_id"]][ref.split(".")[-1]]
        else:
            row["final_event_id"] = check["final_event_id"]
        checks.append(row)
    return {
        "call_id": ctx.call_id,
        **({"language": language_brief(ctx.spec)} if ctx.multilingual else {}),
        **({"register_guide": block} if ctx.multilingual and (block := register_guide.brief_block(ctx.spec)) else {}),
        "scenario": {key: scenario[key] for key in ("client", "employee", "setting", "final_product", "other_products",
                                                    "holding_context", "red_flags")},
        "products": [_product_brief(ctx, p) for p in inputs.configs[ctx.call_id]["products"]],
        "edge_assignments": [{"assignment_id": e["assignment_id"], "edge_case_id": e["edge_case_id"],
                              "description": catalog[e["edge_case_id"]]["description"], "motivation": motivations.get(e["assignment_id"])}
                             for e in ctx.spec["edge_cases"]],
        "checks": checks,
        "constraints": [{key: value for key, value in c.items()} for c in ctx.skeleton["constraints"]],
        "events": [_event_brief(ctx, event) for event in ctx.events],
    }


class RequestBuilder:
    """Builds the identical request body for any call; computes the template/schema hashes once."""

    def __init__(self, inputs, schema_dir=DEFAULT_SCHEMA_DIR, prompt_dir=PROMPT_DIR):
        self.inputs = inputs
        template_path = prompt_dir / prompt_template(inputs.documents)
        self.response_format_name = response_format_name(inputs.documents)
        self.prompt_template_sha256 = g1.sha256(template_path.read_bytes())
        response_path = schema_path(RESPONSE_SCHEMA, schema_dir)
        self.response_schema_sha256 = g1.sha256(response_path.read_bytes())
        self.response_schema = parse_json_text(response_path.read_text(encoding="utf-8"))
        self.artifact_schema = parse_json_text(schema_path(schema_names(inputs.documents)[0], schema_dir).read_text(encoding="utf-8"))
        template = template_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.developer_message = (template.replace("{{CHECKLIST}}", _checklist_text(inputs.documents))
                                  .replace("{{RULES}}", _rules_text(inputs.documents)))
        self._contexts = {}

    def context(self, call_id):
        if call_id not in self._contexts:
            self._contexts[call_id] = PlanContext(self.inputs, call_id)
        return self._contexts[call_id]

    def build(self, call_id, attempt_id):
        ctx = self.context(call_id)
        return {
            "model": MODEL,
            "input": [
                {"role": "developer", "content": self.developer_message},
                {"role": "user", "content": serialize_json(build_brief(self.inputs, ctx))},
            ],
            "text": {"format": {"type": "json_schema", "name": self.response_format_name, "strict": True, "schema": self.response_schema}},
            "reasoning": {"effort": REASONING_EFFORT},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "metadata": {"dataset_id": self.inputs.dataset_id, "call_id": call_id, "stage": STAGE, "attempt_id": attempt_id},
        }
