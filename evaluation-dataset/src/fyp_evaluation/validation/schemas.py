"""Local JSON Schema validation and small, single-document integrity checks.

This module does not compile gold answers, parse the coverage CSV, verify
dialogue meaning, or establish approval/provenance across documents.
"""

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .dates import is_canonical_dob
from .evidence import resolve_evidence
from .identification import is_invalid_test_identifier

SCHEMA_NAMES = (
    "checklist", "call_config", "check_plan", "dialogue", "ground_truth", "timings",
    "normalization", "transcript", "speaker_roles",
)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def load_json(path):
    """Read UTF-8 JSON, rejecting duplicate keys and nonfinite constants."""
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=_object,
        parse_constant=_invalid_constant,
    )


def _schemas(schema_dir):
    schemas = {
        name: load_json(Path(schema_dir) / f"{name}.schema.json")
        for name in SCHEMA_NAMES
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    # Only these local resources are registered. No network retrieval callback.
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema))
        for schema in schemas.values()
    )
    return schemas, registry


def check_schemas(schema_dir):
    """Check schema definitions against the Draft 2020-12 metaschema."""
    _schemas(schema_dir)


def _integrity(data, name):
    errors = []

    def unique(items, field):
        values = [item[field] for item in items]
        if len(values) != len(set(values)):
            errors.append(f"Duplicate {field}")

    def walk(value):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append("Numbers must be finite")
        if isinstance(value, dict):
            if value.get("kind") == "extraction" and value.get("value_type") == "dob_dd_month_yyyy":
                if not is_canonical_dob(value["expected_value"]):
                    errors.append("DOB must be a calendar-valid DD Month YYYY string")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    if name == "checklist":
        unique(data, "id")
    if name == "normalization":
        rules = data["rules"]
        compliance = rules.get(data["compliance_rule"])
        if compliance is None or compliance["value_type"] != "verdict":
            errors.append("compliance_rule must reference an existing verdict rule")
        for field, rule_id in data["field_rules"].items():
            rule = rules.get(rule_id)
            if rule is None:
                errors.append(f"{field}: unknown normalization rule {rule_id}")
            elif rule["value_type"] == "verdict":
                errors.append(f"{field}: extraction fields cannot use a verdict rule")
            elif field == "dob" and rule["value_type"] != "dob_dd_month_yyyy":
                errors.append("dob must use a DD Month YYYY string rule")
    if name == "call_config":
        unique(data["intended_answers"], "question_id")
        unique(data["scenario_facts"], "name")
        duration = data["target_duration_seconds"]
        if duration["min"] > duration["max"]:
            errors.append("target_duration_seconds.min must be <= max")
    if name in {"call_config", "check_plan", "ground_truth"}:
        answers = data["intended_answers" if name == "call_config" else "checks"]
        for answer in answers:
            if answer["question_id"] == "K01":
                value = answer["expected"]["expected_value"]
                if value is not None and not is_invalid_test_identifier(value):
                    errors.append("K01 must contain a checksum-invalid synthetic identifier")
    if name in {"check_plan", "ground_truth"}:
        unique(data["checks"], "question_id")
        for check in data["checks"]:
            if check["evidence_plan"]["mode"] == "presence":
                unique(check["evidence_plan"]["planned_evidence"], "evidence_id")
    if name in {"dialogue", "ground_truth"}:
        turns = data["turns" if name == "dialogue" else "dialogue"]
        for index, turn in enumerate(turns):
            for token in turn["critical_tokens"]:
                if token not in turn["text"]:
                    errors.append(f"Turn {index}: critical token is not in spoken text")
    if name in {"ground_truth", "timings"}:
        turns = data["dialogue" if name == "ground_truth" else "turns"]
        if any(type(turn["turn_id"]) is not int for turn in turns):
            errors.append("turn_id values must use integer JSON numbers, not decimals")
        if [turn["turn_id"] for turn in turns] != list(range(len(turns))):
            errors.append("turn_id values must be contiguous integers starting at zero")
    if name == "ground_truth":
        for check in data["checks"]:
            ids = check["gold_evidence_turns"]
            if any(type(i) is not int for i in ids):
                errors.append("gold_evidence_turns must use integer JSON numbers")
            if ids != sorted(ids):
                errors.append("gold_evidence_turns must be sorted")
            if any(i >= len(data["dialogue"]) for i in ids):
                errors.append("gold_evidence_turns references a missing turn")
            plan = check["evidence_plan"]
            if plan["mode"] == "presence":
                resolved, matching_errors = resolve_evidence(plan, data["dialogue"])
                matched_ids = set(resolved)
                for error in matching_errors:
                    errors.append("Planned evidence must match exactly one eligible turn")
                    # Preserve legacy diagnostics, including the ambiguous
                    # candidate union. Candidates are never selected as gold.
                    matched_ids.update(error["matching_turn_ids"])
                if ids != sorted(matched_ids):
                    errors.append("gold_evidence_turns disagrees with planned evidence")
                quote = "\n".join(e["text"] for e in plan["planned_evidence"])
                if check["gold_evidence_text"] != quote:
                    errors.append("gold_evidence_text disagrees with planned evidence")
    if name == "transcript":
        segments = data["audio_segments"]
        ids = [segment["id"] for segment in segments]
        if len(ids) != len(set(ids)):
            errors.append("audio_segments ids must be unique")
        for segment in segments:
            try:
                begin, finish = float(segment["start_time"]), float(segment["end_time"])
            except ValueError:
                errors.append(f"audio_segments[{segment['id']}]: start_time/end_time must be numeric strings")
                continue
            if not math.isfinite(begin) or not math.isfinite(finish) or begin < 0 or finish < begin:
                errors.append(f"audio_segments[{segment['id']}]: start_time/end_time must be finite and start_time <= end_time")
    if name == "timings":
        previous_end = 0
        for turn in data["turns"]:
            start, end = turn["start_seconds"], turn["end_seconds"]
            if not previous_end <= start <= end <= data["duration_seconds"]:
                errors.append("Turn timings must be ordered, nonoverlapping and within audio")
            previous_end = end
            word_end = start
            for word in turn.get("words", []):
                if not word_end <= word["start_seconds"] <= word["end_seconds"] <= end:
                    errors.append("Word timings must be ordered and within their turn")
                word_end = word["end_seconds"]
    return errors


def validate(data, schema_name, schema_dir):
    """Return errors, or [] for structural validity; never mutate the input.

    Schema files remain explicit repository resources. An installed caller can
    pass their path from another repository without relying on its cwd.
    """
    if schema_name not in SCHEMA_NAMES:
        raise ValueError(f"Unknown schema: {schema_name}")
    schemas, registry = _schemas(schema_dir)
    validator = Draft202012Validator(schemas[schema_name], registry=registry)
    errors = [
        f"{error.json_path}: {error.message}"
        for error in validator.iter_errors(data)
    ]
    return errors if errors else _integrity(data, schema_name)


def validate_checklist(checklist, normalization, schema_dir):
    """Validate the authoritative checklist and reusable field-rule references."""
    errors = []
    for name, data in [
        ("checklist", checklist), ("normalization", normalization),
    ]:
        errors.extend(f"{name}: {error}" for error in validate(data, name, schema_dir))
    if errors:
        return errors
    for item in checklist:
        if item["id"] == "K01":
            if item["type"] != "CLIENT_KYC" or item.get("field") != "client_identification_number":
                errors.append("K01 must be CLIENT_KYC field client_identification_number")
            rule = normalization["rules"].get(
                normalization["field_rules"].get("client_identification_number"),
            )
            if rule != {"operation": "exact", "value_type": "text"}:
                errors.append("K01 requires an exact-text normalization rule")
        if item["type"] != "COMPLIANCE" and item["field"] not in normalization["field_rules"]:
            errors.append(f"{item['id']}: no normalization rule for field {item['field']}")
    return errors


def to_langgraph_checklist(checklist, schema_dir):
    """Project a valid internal checklist without changing IDs, order or text.

    This pure adapter does not write a second checklist or call LangGraph.
    """
    if errors := validate(checklist, "checklist", schema_dir):
        raise ValueError("\n".join(errors))
    return [{"id": item["id"], "question": item["text"]} for item in checklist]


def validate_answers(data, schema_name, checklist, normalization, schema_dir):
    """Check one call document's expected-answer types against its checklist.

    Does not compile data or certify agreement with the coverage CSV or other
    call documents. Value preservation across stages remains a compiler duty.
    """
    if schema_name not in {"call_config", "check_plan", "ground_truth"}:
        raise ValueError("Answer validation requires call_config, check_plan or ground_truth")
    errors = validate_checklist(checklist, normalization, schema_dir)
    errors.extend(validate(data, schema_name, schema_dir))
    if errors:
        return errors
    items = {item["id"]: item for item in checklist}
    answers = data["intended_answers" if schema_name == "call_config" else "checks"]
    answer_ids = {answer["question_id"] for answer in answers}
    if missing := items.keys() - answer_ids:
        errors.append(f"Missing checklist answers: {sorted(missing)}")
    if extra := answer_ids - items.keys():
        errors.append(f"Unknown checklist answers: {sorted(extra)}")
    for answer in answers:
        item = items.get(answer["question_id"])
        if item is None:
            continue
        expected = answer["expected"]
        if item["type"] == "COMPLIANCE":
            if expected["kind"] != "compliance":
                errors.append(f"{item['id']}: COMPLIANCE requires a PASS/FAIL verdict")
        else:
            rule_id = normalization["field_rules"][item["field"]]
            value_type = normalization["rules"][rule_id]["value_type"]
            if expected["kind"] != "extraction" or expected["value_type"] != value_type:
                errors.append(f"{item['id']}: {item['type']} field {item['field']} requires extraction value_type {value_type}")
    return errors
