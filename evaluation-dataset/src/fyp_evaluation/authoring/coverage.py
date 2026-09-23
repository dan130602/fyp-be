"""Read approved matrix cells and compile their existing JSON representation.

No answers are chosen here. Text is preserved; typed scalars are parsed, never
repaired. The matrix remains the sole source of expected answers.
"""

import csv
import json
import math
import re
from pathlib import Path

from fyp_evaluation.validation.dates import is_canonical_dob
from fyp_evaluation.validation.identification import is_invalid_test_identifier
from fyp_evaluation.validation.schemas import load_json

PILOT_CALL_IDS = tuple(f"CALL_{number:03}" for number in range(1, 6))


def check_matrix(matrix, checklist, *, call_ids=None):
    """Raise ValueError for incomplete/invalid call -> question -> raw-cell maps."""
    ids = [item["id"] for item in checklist]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Checklist IDs must be nonempty and unique")
    if not isinstance(matrix, dict) or not matrix:
        raise ValueError("Matrix must contain call columns")
    if call_ids is not None and list(matrix) != list(call_ids):
        raise ValueError(f"Matrix call columns must be exactly {list(call_ids)} in order")
    for call_id, cells in matrix.items():
        if not isinstance(call_id, str) or not re.fullmatch(r"CALL_[0-9]{3,}", call_id):
            raise ValueError(f"Invalid matrix call column: {call_id!r}")
        if not isinstance(cells, dict) or set(cells) != set(ids):
            raise ValueError(f"{call_id}: matrix must contain exactly the approved checklist IDs")
        for question_id, cell in cells.items():
            if not isinstance(cell, str) or not cell.strip() or "\x00" in cell:
                raise ValueError(f"{call_id}/{question_id}: missing or invalid string cell")


def load_matrix(path, *, checklist=None, call_ids=PILOT_CALL_IDS):
    """Load UTF-8 CSV as call -> question -> unchanged string cell.

    By default, use the project checklist adjacent to authoring/ and the five
    pilot columns. Other repositories/call sets can explicitly supply their
    approved checklist and call_ids. No copied question-ID catalogue is kept.
    """
    path = Path(path)
    if checklist is None:
        checklist = load_json(path.parent.parent / "evaluation_dataset/checklist.json")
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file, strict=True))
    except csv.Error as error:
        raise ValueError(f"Malformed coverage CSV: {error}") from error
    if not rows or not rows[0] or rows[0][0] != "question_id":
        raise ValueError("Matrix first header must be question_id")
    header = rows[0]
    if len(header) < 2 or len(header) != len(set(header)):
        raise ValueError("Matrix headers must include unique call columns")
    matrix = {call_id: {} for call_id in header[1:]}
    seen = set()
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f"Matrix row {line}: ragged row or missing cells")
        question_id = row[0]
        if question_id in seen:
            raise ValueError(f"Matrix row {line}: duplicate question ID {question_id}")
        seen.add(question_id)
        for call_id, cell in zip(header[1:], row[1:]):
            matrix[call_id][question_id] = cell
    check_matrix(matrix, checklist, call_ids=call_ids)
    return matrix


def parse_k01_cell(cell):
    """Decode K01 only: NOT_STATED -> None; preserve valid test text exactly.

    The compiler calls this only for CLIENT_KYC K01 with field
    client_identification_number. JSON serialization turns None into null.
    No whitespace, case or checksum correction is performed.
    """
    if cell == "NOT_STATED":
        return None
    if not is_invalid_test_identifier(cell):
        raise ValueError("K01 requires NOT_STATED or a checksum-invalid nine-character test identifier")
    return cell


def compile_expected(checklist_item, normalization, cell):
    """Convert one raw matrix cell; callers validate checklist/rules first."""
    if not isinstance(cell, str) or not cell.strip():
        raise ValueError("Expected a nonempty raw matrix string")
    item = checklist_item
    if item["id"] == "K01" and (
        item["type"] != "CLIENT_KYC" or item.get("field") != "client_identification_number"
    ):
        raise ValueError("K01 must be CLIENT_KYC field client_identification_number")
    if item["type"] == "COMPLIANCE":
        if cell not in {"PASS", "FAIL"}:
            raise ValueError(f"{item['id']}: expected exact PASS or FAIL")
        return {"kind": "compliance", "expected_verdict": cell}
    if item["type"] not in {"CLIENT_KYC", "TRANSACTION"}:
        raise ValueError(f"Unsupported checklist type: {item['type']}")
    try:
        rule = normalization["rules"][normalization["field_rules"][item["field"]]]
    except KeyError as error:
        raise ValueError(f"{item['id']}: missing field normalization rule") from error
    value_type = rule["value_type"]
    if rule["operation"] != "exact":
        raise ValueError("Only existing exact normalization rules are supported")
    if item["id"] == "K01":
        if value_type != "text":
            raise ValueError("K01 requires exact text")
        value = parse_k01_cell(cell)
    elif value_type == "text":
        value = cell
    elif value_type == "dob_dd_month_yyyy":
        if not is_canonical_dob(cell):
            raise ValueError(f"{item['id']}: invalid canonical DOB")
        value = cell
    elif value_type in {"integer", "number", "boolean"}:
        if cell != cell.strip():
            raise ValueError(f"{item['id']}: scalar syntax must not contain surrounding whitespace")
        try:
            value = json.loads(cell)
        except ValueError as error:
            raise ValueError(f"{item['id']}: invalid JSON scalar syntax") from error
        valid_type = {
            "integer": type(value) is int,
            "number": type(value) in {int, float},
            "boolean": type(value) is bool,
        }[value_type]
        if not valid_type or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{item['id']}: expected a finite {value_type} scalar")
    else:
        raise ValueError(f"Unsupported extraction value_type: {value_type}")
    return {"kind": "extraction", "value_type": value_type, "expected_value": value}
