"""G5 verification context, the per-call brief, the stable system prefix and the canonical Interactions request body.

The brief is built by Python from the frozen gold (call spec), the accepted G3 config and the accepted G4 plan. It carries
every planned event (distractors and abandoned false starts included) and every frozen or derived constraint, but no
hashes, provenance, templates, span offsets or spoken-form rules (G4 already proved those structurally). The request
body contains no attempt ID, so a dry-run hash equals the hash of the request that would be sent.
"""

import json

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .config import MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, PROMPT_TEMPLATE, RESPONSE_SCHEMA, SERVICE_TIER, THINKING_LEVEL

EVENT_RELATIONS = ("corrects_event_id", "refers_to_event_id", "invalidates_event_id")
EVENT_ATTRIBUTES = ("distractor_role", "currency", "incomplete", "strength")


def compact_json(value):
    """Deterministic compact JSON for model input (key order as constructed, no whitespace, UTF-8 kept)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def schema_path(name, schema_dir=DEFAULT_SCHEMA_DIR):
    return schema_dir / "stage5" / f"{name}.schema.json"


class VerificationContext:
    """Everything one call's validation and compilation need, bound to ONE version of its G3 config and G4 plan."""

    def __init__(self, inputs, call_id):
        self.inputs, self.call_id = inputs, call_id
        self.spec = inputs.specs[call_id]
        self.config = inputs.configs[call_id]
        self.plan = inputs.plans[call_id]
        self.events = self.plan["planned_events"]
        self.event_index = {event["event_id"]: index for index, event in enumerate(self.events)}
        self.events_by_id = {event["event_id"]: event for event in self.events}
        self.checks = {check["question_id"]: check for check in self.plan["checks"]}
        self.order = [item["id"] for item in inputs.documents["checklist"]]
        self.gold = {entry["question_id"]: entry["expected"] for entry in self.spec["gold"]}
        self.assignment_ids = [edge["assignment_id"] for edge in self.spec["edge_cases"]]
        self.g3 = {"products": self.config["products"], "scenario": self.config["scenario"]}
        self.semantics = {entry["question_id"]: entry for entry in inputs.documents["check_semantics"]["checks"]}

    def planned_text(self, event_id):
        return self.events_by_id[event_id]["authoring"]["planned_text"]


def _expected(expected):
    if expected["kind"] == "compliance":
        return expected["expected_verdict"]
    return {"value_type": expected["value_type"], "value": expected["expected_value"]}


def _check_brief(ctx, question_id):
    check = ctx.checks[question_id]
    row = {"question_id": question_id, "kind": check["kind"], "expected": _expected(ctx.gold[question_id]),
           "evidence_mode": check["evidence_mode"], "presence_pattern": check.get("presence_pattern"),
           "wording_constraint": check.get("wording_constraint")}
    if check["kind"] == "COMPLIANCE":
        ref = check.get("required_condition_ref") or check.get("forbidden_condition_ref")
        row["target"] = {"must": "ESTABLISH" if check["evidence_mode"] == "PRESENCE" else "NOT_ESTABLISH",
                         "condition_ref": ref, "condition_text": ctx.semantics[question_id][ref.split(".")[-1]]}
    else:
        row["target"] = {"must": "ESTABLISH" if check["evidence_mode"] == "PRESENCE" else "NOT_ESTABLISH",
                         "resolution_rule_ref": check.get("resolution_ref")}
    row["absence_scope"] = (check.get("absence") or {}).get("scope")
    row["evidence_event_ids"] = check["evidence_event_ids"]
    row["final_event_id"] = check.get("final_event_id")
    row["assignment_ids"] = check["assignment_ids"]
    return row


def _event_brief(event):
    authoring = event["authoring"]
    entry = {"event_id": event["event_id"], "kind": event["kind"], "status": event["status"], "role": event.get("role"),
             "condition_ref": event.get("condition_ref"), "question_ids": event["question_ids"],
             "assignment_ids": event["assignment_ids"], "speaker": authoring["speaker"], "planned_text": authoring["planned_text"],
             "values": {q: v["value"] for q, v in event["values"].items()} if "values" in event else None,
             "value_mentions": [{"question_id": m["question_id"], "value": m["value"], "surface": m["surface"], "source": m["source"]}
                                for m in authoring["value_mentions"]]}
    for key in EVENT_RELATIONS + EVENT_ATTRIBUTES:
        entry[key] = event.get(key)
    readback = event.get("readback")
    entry["readback"] = None if readback is None else {
        "wrong_question_id": readback["wrong_question_id"], "wrong_value": readback["wrong_value"],
        "must_include_question_ids": readback["must_include_question_ids"]}
    return entry


def build_brief(inputs, ctx):
    """The per-call input (a plain dict; serialised canonically by the caller)."""
    catalog = {entry["id"]: entry for entry in inputs.documents["edge_case_catalog"]["edge_cases"]}
    meta = ctx.spec["metadata"]
    return {
        "call": {"call_id": ctx.call_id, "call_intents": meta["call_intents"], "duration_profile": meta["duration_profile"],
                 "language": meta["language"]},
        "checks": [_check_brief(ctx, q) for q in ctx.order],
        "g3": ctx.g3,
        "edge_cases": [{"assignment_id": e["assignment_id"], "edge_case_id": e["edge_case_id"],
                        "target_question_ids": e["target_question_ids"], "parameters": e["parameters"],
                        "catalog": {key: catalog[e["edge_case_id"]][key] for key in ("description", "gold_constraints", "plan_template")}}
                       for e in ctx.spec["edge_cases"]],
        "events": [_event_brief(event) for event in ctx.events],
        "constraints": ([{**c, "derived": False} for c in ctx.plan["constraints"]]
                        + [{**c, "derived": True} for c in ctx.plan["derived_constraints"]]),
    }


def system_prefix(template, documents):
    """The stable instruction shared by every call (checklist and semantics inserted verbatim as canonical JSON)."""
    semantics = documents["check_semantics"]
    return (template.replace("{{CHECKLIST}}", compact_json(documents["checklist"]))
            .replace("{{SEMANTICS}}", compact_json({key: semantics[key] for key in ("checks", "ordering_rules", "cross_check_rules")})))


class RequestBuilder:
    """Builds the identical request body for any call; computes the template/schema hashes once."""

    def __init__(self, inputs, schema_dir=DEFAULT_SCHEMA_DIR, prompt_dir=PROMPT_DIR):
        self.inputs = inputs
        template_path = prompt_dir / PROMPT_TEMPLATE
        self.prompt_template_sha256 = g1.sha256(template_path.read_bytes())
        response_path = schema_path(RESPONSE_SCHEMA, schema_dir)
        self.response_schema_sha256 = g1.sha256(response_path.read_bytes())
        self.response_schema = parse_json_text(response_path.read_text(encoding="utf-8"))
        self.artifact_schema = parse_json_text(schema_path("plan_verification", schema_dir).read_text(encoding="utf-8"))
        self.system_instruction = system_prefix(template_path.read_text(encoding="utf-8").replace("\r\n", "\n"), inputs.documents)
        self._contexts = {}

    def context(self, call_id):
        if call_id not in self._contexts:
            self._contexts[call_id] = VerificationContext(self.inputs, call_id)
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
            "service_tier": SERVICE_TIER,
            "store": False,
        }
