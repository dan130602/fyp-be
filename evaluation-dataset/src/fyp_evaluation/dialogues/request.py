"""G6 dialogue context, the per-call brief and the canonical Responses API request body (the same body is sent by Batch and by `--sync`).

The brief is deliberately narrow: harmless scenario background, the planned events as markers with their exact text, the
placement constraints and the absence conditions. Ready-to-speak compliance material from G3 (concrete risks, product
description, recommendation basis, fees, limit prices, red flags, objectives, loss attitude) is never shown: every planned
use of it already exists as a G4 event, so supporting speech cannot leak what the model never sees. No gold values,
verdicts, question IDs, G5 rationales, hashes or offsets are included. Two Python-derived facts tell the model about the
advisory component: `call.call_intents` (the call_spec array, unchanged; for this pilot its order is the intended segment order)
and `planned_recommendation_event_ids` (the events the C07 plan structure already marks as the recommendation; empty when none).
When none, `advisory_recommendation` (see advisory.py) names the frozen product, the direction, and the derived placement window.
"""

from fyp_evaluation import register_guide
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.language_brief import language_brief
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR, is_multilingual

from .advisory import advisory_recommendation
from .config import (
    EVENT_RELATIONS, IDENTITY_QUESTIONS, MAX_OUTPUT_TOKENS, MODEL, PROMPT_DIR, REASONING_EFFORT, STAGE, length_policy_for,
    prompt_template, response_format_name, response_schema_name, schema_names, unit_of,
)


def schema_path(name, schema_dir=DEFAULT_SCHEMA_DIR):
    return schema_dir / "stage6" / f"{name}.schema.json"


def marker(event_id):
    return f"[[{event_id}]]"


class DialogueContext:
    """Everything one call's validation and compilation need, bound to ONE version of its G3/G4 inputs and length policy."""

    def __init__(self, inputs, call_id, length_policy=None):
        self.inputs, self.call_id = inputs, call_id
        self.spec = inputs.specs[call_id]
        self.multilingual = is_multilingual(inputs.documents)
        self.language_requirements = self.spec.get("language_requirements")
        if length_policy is None:
            length_policy = length_policy_for(self.spec, inputs.documents)
        self.config = inputs.configs[call_id]
        self.plan = inputs.plans[call_id]
        self.metadata = self.spec["metadata"]
        self.profile = self.metadata["duration_profile"]
        self.length_policy = length_policy
        self.unit = unit_of(length_policy)
        self.bounds = length_policy["profiles"][self.profile]
        self.events = self.plan["planned_events"]
        self.event_index = {event["event_id"]: index for index, event in enumerate(self.events)}
        self.speakers = {event["event_id"]: event["authoring"]["speaker"] for event in self.events}
        self.texts = {event["event_id"]: event["authoring"]["planned_text"] for event in self.events}
        self.constraints = self.plan["constraints"] + self.plan["derived_constraints"]
        # the planned events that already carry the employee's recommendation (the C07 plan structure; no keywords, no model).
        # Empty when the plan has none (C07 FAIL is an absence check): the brief then carries `advisory_recommendation`, the frozen product and
        # action, the derived placement window and the evidence to keep apart (see advisory.py), and asks for ONE clear rationale-free
        # recommendation. None when the plan already carries the recommendation (or ADVISORY is not an intent).
        self.recommendation_event_ids = [event["event_id"] for event in self.events if "C07" in event["question_ids"]]
        self.advisory = advisory_recommendation(self.spec, self.config, self.events, self.constraints, inputs.documents)
        mentions = sorted({(m["question_id"], m["surface"]) for event in self.events for m in event["authoring"]["value_mentions"]
                           if m.get("question_id")})  # scenario placeholders (fee, limit price) carry no question
        self.identity_surfaces = [item for item in mentions if item[0] in IDENTITY_QUESTIONS]
        self.value_surfaces = [item for item in mentions if item[0] not in IDENTITY_QUESTIONS]
        semantics = {entry["question_id"]: entry for entry in inputs.documents["check_semantics"]["checks"]}
        self.absence_conditions = [
            {"question_id": check["question_id"],
             "condition": semantics[check["question_id"]][check["forbidden_condition_ref"].split(".")[-1]]}
            for check in self.plan["checks"] if check["kind"] == "COMPLIANCE" and check["evidence_mode"] == "ABSENCE"]


def _event_brief(event):
    entry = {"event_id": event["event_id"], "marker": marker(event["event_id"]), "speaker": event["authoring"]["speaker"],
             "kind": event["kind"], "status": event["status"], "text": event["authoring"]["planned_text"]}
    for key in EVENT_RELATIONS:
        if key in event:
            entry[key] = event[key]
    return entry


def _constraint_brief(constraint):
    keys = ("before", "after") if constraint["type"] == "ORDER" else ("events",)
    entry = {"type": constraint["type"], **{key: constraint[key] for key in keys}}
    if constraint["type"] == "MIN_INTERVENING_TURNS":
        entry["turns"] = constraint["turns"]
    return entry


MAX_GUIDANCE_ITEMS = 8
MAX_GUIDANCE_CHARS = 700


def load_repair_feedback(path, root):
    """Validated optional repair feedback: {call_id: {source_g7_report_sha256, source_g7_attempt_id, round, guidance[]}}.

    The guidance of a call must be derived from that call's CURRENT accepted G7 report: the recorded hash must equal the hash of
    the report on disk now and the report must be INVALID. Nothing else about G6 acceptance changes: this only adds the guidance
    to the request brief and records its source on the attempt.
    """
    import re
    from pathlib import Path

    data = parse_json_text(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("repair feedback must be a non-empty JSON object keyed by call id")
    for call_id, entry in data.items():
        if not re.fullmatch(r"CALL_[0-9]{3,}", call_id) or set(entry) != {"source_g7_report_sha256", "source_g7_attempt_id", "round", "guidance"}:
            raise ValueError(f"{call_id}: needs exactly source_g7_report_sha256, source_g7_attempt_id, round and guidance")
        guidance = entry["guidance"]
        if (not isinstance(guidance, list) or not 1 <= len(guidance) <= MAX_GUIDANCE_ITEMS
                or any(not isinstance(item, str) or not item.strip() or len(item) > MAX_GUIDANCE_CHARS for item in guidance)):
            raise ValueError(f"{call_id}: guidance must be 1-{MAX_GUIDANCE_ITEMS} non-blank strings of at most {MAX_GUIDANCE_CHARS} characters")
        report_file = Path(root) / "authoring/dialogue_verifications" / f"{call_id}.json"
        try:
            raw = report_file.read_bytes()
        except OSError as error:
            raise ValueError(f"{call_id}: the source G7 report cannot be read ({error.__class__.__name__})") from error
        if g1.sha256(raw) != entry["source_g7_report_sha256"]:
            raise ValueError(f"{call_id}: source_g7_report_sha256 does not match the current G7 report")
        report = parse_json_text(raw.decode("utf-8"))
        if report.get("verdict") != "INVALID" or report["generation"]["attempt_id"] != entry["source_g7_attempt_id"]:
            raise ValueError(f"{call_id}: the source report must be the INVALID report of attempt {entry['source_g7_attempt_id']}")
    return data


def _people(ctx, scenario):
    people = {"client": {"name": scenario["client"]["name"], "background": scenario["client"]["background"]},
              "employee": {"name": scenario["employee"]["name"]}}
    if ctx.multilingual:  # the call is spoken in the speakers' own languages: give each speaker's spoken names
        for speaker in ("client", "employee"):
            people[speaker]["spoken_names"] = {language: name for language, name in scenario[speaker]["spoken_names"].items() if name}
    return people


def _call_block(ctx):
    bounds, meta = ctx.bounds, ctx.metadata
    call = {"call_id": ctx.call_id, "call_intents": list(meta["call_intents"]), "duration_profile": ctx.profile}
    if ctx.multilingual:
        call["language"] = language_brief(ctx.spec)
        call["length"] = {"unit": ctx.unit, "target": {"min": bounds["target_words"][0], "max": bounds["target_words"][1]},
                          "accepted": {"min": bounds["accepted_words"][0], "max": bounds["accepted_words"][1]}}
    else:
        call = {"call_id": ctx.call_id, "language": meta["language"], "call_intents": list(meta["call_intents"]),
                "duration_profile": ctx.profile,
                "target_words": {"min": bounds["target_words"][0], "max": bounds["target_words"][1]},
                "accepted_words": {"min": bounds["accepted_words"][0], "max": bounds["accepted_words"][1]}}
    return call


def build_brief(ctx):
    scenario, catalog = ctx.config["scenario"], {e["id"]: e for e in ctx.inputs.documents["edge_case_catalog"]["edge_cases"]}
    final = next(p for p in ctx.config["products"] if "FINAL" in p["roles"])
    brief = {
        "call": _call_block(ctx),
        "people": _people(ctx, scenario),
        "setting": {"call_reason": scenario["setting"]["call_reason"], "conversation_style": scenario["setting"]["conversation_style"]},
        "holding_context": scenario["holding_context"],
        "final_product": {"name": final["name"], "category": final["category"]},
        "discussion_topics": list(scenario["additional_discussion_topics"]),
        "events": [_event_brief(event) for event in ctx.events],
        "planned_recommendation_event_ids": list(ctx.recommendation_event_ids),
        "placement_constraints": [_constraint_brief(c) for c in ctx.constraints],
        "must_never_be_expressed_outside_events": [item["condition"] for item in ctx.absence_conditions],
        "edge_cases": [{"edge_case_id": e["edge_case_id"], "description": catalog[e["edge_case_id"]]["description"],
                        **({"language_variant": e["language_variant"]} if ctx.multilingual else {})}
                       for e in ctx.spec["edge_cases"]],
    }
    if ctx.advisory is not None:  # only the calls with no planned recommendation; every other brief is unchanged
        brief["advisory_recommendation"] = ctx.advisory
    guide = register_guide.brief_block(ctx.spec) if ctx.multilingual else None
    if guide:  # SINGLISH calls only
        brief["register_guide"] = guide
    return brief


class RequestBuilder:
    """Builds the identical request body for any call; computes the template/schema hashes once."""

    def __init__(self, inputs, schema_dir=DEFAULT_SCHEMA_DIR, prompt_dir=PROMPT_DIR, repair_feedback=None):
        self.inputs = inputs
        self.repair_feedback = repair_feedback or {}  # {call_id: {"guidance": [str, ...], ...}}; optional, see load_repair_feedback
        template_path = prompt_dir / prompt_template(inputs.documents)
        self.prompt_template_sha256 = g1.sha256(template_path.read_bytes())
        response_path = schema_path(response_schema_name(inputs.documents), schema_dir)
        self.response_schema_sha256 = g1.sha256(response_path.read_bytes())
        self.response_schema = parse_json_text(response_path.read_text(encoding="utf-8"))
        self.artifact_schema = parse_json_text(schema_path(schema_names(inputs.documents)[0], schema_dir).read_text(encoding="utf-8"))
        self.developer_message = template_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        self._contexts = {}

    def context(self, call_id):
        if call_id not in self._contexts:
            self._contexts[call_id] = DialogueContext(self.inputs, call_id)
        return self._contexts[call_id]

    def build(self, call_id, attempt_id):
        brief = build_brief(self.context(call_id))
        if call_id in self.repair_feedback:  # only the calls being repaired; every other request is byte-identical to before
            brief["repair_guidance"] = list(self.repair_feedback[call_id]["guidance"])
        return {
            "model": MODEL,
            "input": [
                {"role": "developer", "content": self.developer_message},
                {"role": "user", "content": serialize_json(brief)},
            ],
            "text": {"format": {"type": "json_schema", "name": response_format_name(self.inputs.documents), "strict": True, "schema": self.response_schema}},
            "reasoning": {"effort": REASONING_EFFORT},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "metadata": {"dataset_id": self.inputs.dataset_id, "call_id": call_id, "stage": STAGE, "attempt_id": attempt_id},
        }
