"""Cross-file call-config validation; never infers or repairs gold answers."""

import re

from fyp_evaluation.authoring.coverage import check_matrix, compile_expected
from .schemas import validate_answers


def contains_placeholder(text):
    """Shared authored-text check; preserves the Stage 3A marker convention."""
    placeholder = re.search(r"\b(?:TODO|TBD|FIXME|PLACEHOLDER)(?:\b|_)", text, flags=re.IGNORECASE)
    return bool(placeholder) or text.strip().upper() in {
        "TBC", "PENDING", "N/A", "NONE", "NULL", "...", "-", "TO BE AUTHORED",
    }


def validate_call_config(data, checklist, normalization, matrix, schema_dir, *, complete=False):
    """Return structural, matrix-preservation and optional completeness errors.

    Load the matrix with load_matrix using the approved call columns first.
    Complete mode detects empty facts and reserved placeholder markers, not
    narrative truth or semantic correctness; those still require review.
    """
    errors = validate_answers(data, "call_config", checklist, normalization, schema_dir)
    if errors:
        return errors
    try:
        check_matrix(matrix, checklist)
    except ValueError as error:
        return [str(error)]
    call_id = data["call_id"]
    if call_id not in matrix:
        return [f"{call_id}: no column in the approved matrix"]
    answers = data["intended_answers"]
    if [item["question_id"] for item in answers] != [item["id"] for item in checklist]:
        errors.append("intended_answers must follow approved checklist order")
    items = {item["id"]: item for item in checklist}
    for answer in answers:
        question_id = answer["question_id"]
        try:
            expected = compile_expected(items[question_id], normalization, matrix[call_id][question_id])
        except ValueError as error:
            errors.append(f"Matrix {call_id}/{question_id}: {error}")
            continue
        actual = answer["expected"]
        # Python equality alone conflates bool/int/float. Preserve JSON scalar
        # representations too (e.g. integer 1 must not become decimal 1.0).
        if actual != expected or type(actual.get("expected_value")) is not type(expected.get("expected_value")):
            errors.append(f"{call_id}/{question_id}: expected answer differs from approved matrix (value or type)")
    extraction_fields = {item["field"] for item in checklist if item["type"] != "COMPLIANCE"}
    for fact in data["scenario_facts"]:
        if fact["name"] in extraction_fields:
            errors.append(f"scenario_facts must not restate matrix extraction field {fact['name']}")
    if complete:
        if not data["scenario_facts"]:
            errors.append("Incomplete: scenario_facts must be populated")
        for answer in answers:
            intent = answer["scenario_intent"]
            if contains_placeholder(intent):
                errors.append(f"Incomplete: {answer['question_id']} scenario_intent contains a placeholder")
    return errors
