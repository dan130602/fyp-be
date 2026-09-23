"""Independent Entity WER annotation coverage, not scoring or evidence inference."""

import re
from pathlib import Path

from fyp_evaluation.authoring.coverage import check_matrix, compile_expected
from .schemas import validate, validate_checklist
from .spoken import verify_spoken_value

_SCHEMAS = Path(__file__).resolve().parents[3] / "schemas"
_FINANCIAL_LOOKING = re.compile(
    r"\b(?:SGD|USD|EUR|GBP|AUD)\b|[$£€]\s*[0-9][0-9,.]*|[0-9]+(?:[,.][0-9]+)*(?:\s*%)?"
)


def validate_critical_token_coverage(dialogue, checklist, normalization, matrix, *, schema_dir=_SCHEMAS):
    """Return (blocking error strings, separate nonblocking advisory records).

    Reuse existing dialogue validation for own-turn substring/uniqueness rules.
    Matrix selection uses only the validated internal call ID. No plan is read.
    Advisories are heuristic candidate spans, never proof of annotation quality.
    """
    errors = validate(dialogue, "dialogue", schema_dir)
    if errors:
        return errors, []
    call_id = dialogue["call_id"]
    if call_id not in matrix:
        return [f"{call_id}: no column in approved matrix"], []
    errors = validate_checklist(checklist, normalization, schema_dir)
    if errors:
        return errors, []
    try:
        check_matrix(matrix, checklist)
    except ValueError as error:
        return [str(error)], []
    tokens = [token for turn in dialogue["turns"] for token in turn["critical_tokens"]]
    for item in checklist:
        if item["type"] == "COMPLIANCE":
            continue
        try:
            expected = compile_expected(item, normalization, matrix[call_id][item["id"]])
        except ValueError as error:
            errors.append(f"{item['id']}: {error}")
            continue
        if expected["expected_value"] is not None and not any(
            verify_spoken_value(expected, token) is None for token in tokens
        ):
            errors.append(f"{item['id']}: no critical token covers the locked extraction value")
    advisory = []
    for index, turn in enumerate(dialogue["turns"]):
        for match in _FINANCIAL_LOOKING.finditer(turn["text"]):
            # String annotations have no occurrence offsets. This is a recall
            # aid only, not span tokenisation or an alignment implementation.
            if not any(match.group() in token for token in turn["critical_tokens"]):
                advisory.append({"turn_id": index, "text": match.group(),
                                 "message": "Financial-looking span has no covering critical token"})
    return errors, advisory
