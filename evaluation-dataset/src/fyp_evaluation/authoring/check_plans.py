"""Deterministic check-plan scaffolding; never authors evidence or gold values."""

import json
import math
from pathlib import Path

from .coverage import check_matrix, compile_expected
from fyp_evaluation.validation.schemas import validate_checklist

CHECK_PLAN_VERSION = "0.6.0"
PLANNED_EVIDENCE_PLACEHOLDER = "TODO_AUTHOR_PLANNED_EVIDENCE"
ABSENT_EVENT_PLACEHOLDER = "TODO_AUTHOR_ABSENT_EVENT"


def derive_evidence_mode(checklist_item, expected):
    """Derive presence/absence from an existing answer; never select an answer."""
    if not isinstance(checklist_item, dict) or not isinstance(expected, dict):
        raise ValueError("Checklist item and expected answer must be objects")
    category = checklist_item.get("type")
    if category == "COMPLIANCE":
        requirement = checklist_item.get("requirement_type")
        verdict = expected.get("expected_verdict")
        if (set(expected) != {"kind", "expected_verdict"} or
                expected.get("kind") != "compliance" or
                requirement not in ("OBLIGATION", "PROHIBITION") or verdict not in ("PASS", "FAIL")):
            raise ValueError("Unsupported compliance answer/requirement combination")
        presence = (requirement == "OBLIGATION") == (verdict == "PASS")
        return "presence" if presence else "absence"
    if category not in ("CLIENT_KYC", "TRANSACTION") or "requirement_type" in checklist_item:
        raise ValueError("Unsupported checklist type/requirement combination")
    if set(expected) != {"kind", "value_type", "expected_value"} or expected.get("kind") != "extraction":
        raise ValueError("Extraction requires a typed expected value")
    value, value_type = expected["expected_value"], expected["value_type"]
    is_k01 = (checklist_item.get("id") == "K01" and category == "CLIENT_KYC"
              and checklist_item.get("field") == "client_identification_number" and value_type == "text")
    if checklist_item.get("id") == "K01" and not is_k01:
        raise ValueError("K01 requires CLIENT_KYC identification with text value_type")
    if value is None:
        if is_k01:
            return "absence"
        raise ValueError("Only K01 identification extraction can be null")
    valid = (
        (value_type in ("text", "dob_dd_month_yyyy") and isinstance(value, str) and bool(value.strip())) or
        (value_type == "integer" and type(value) is int) or
        (value_type == "number" and type(value) in (int, float) and
         (type(value) is int or math.isfinite(value))) or
        (value_type == "boolean" and type(value) is bool)
    )
    if not valid:
        raise ValueError("Unsupported extraction value/type combination")
    return "presence"


def build_check_plan(call_id, checklist, normalization, matrix):
    """Build in checklist order from validated inputs; no source mutation or I/O."""
    check_matrix(matrix, checklist)
    if call_id not in matrix:
        raise ValueError(f"{call_id}: no column in approved matrix")
    checks = []
    for item in checklist:
        expected = compile_expected(item, normalization, matrix[call_id][item["id"]])
        mode = derive_evidence_mode(item, expected)
        if mode == "absence":
            evidence_plan = {
                "mode": mode, "scope": "entire_dialogue",
                "required_speaker": "ANY" if item["id"] == "K01" else "EMPLOYEE",
                "absent_event": ABSENT_EVENT_PLACEHOLDER,
            }
        else:
            evidence_plan = {
                "mode": mode,
                "planned_evidence": [{
                    "evidence_id": f"{item['id']}_E1",
                    "required_speaker": "EMPLOYEE" if item["type"] == "COMPLIANCE" else "CLIENT",
                    "text": PLANNED_EVIDENCE_PLACEHOLDER,
                }],
            }
        evidence_plan.update(content_constraints=[], forbidden_patterns=[])
        checks.append({"question_id": item["id"], "expected": expected, "evidence_plan": evidence_plan})
    return {"schema_version": CHECK_PLAN_VERSION, "call_id": call_id,
            "checks": checks, "ordering_constraints": []}


def write_check_plans(output_dir, checklist, normalization, matrix, call_configs,
                      schema_dir, *, only=None, overwrite=False):
    """Validate every matrix call before writing the selected batch.

    call_configs maps all approved matrix call IDs to approved config objects.
    only changes output selection, never the matrix or validation scope. Like
    call-config writing, writes are per file rather than a batch transaction.
    Existing symlinks are never followed, including with explicit overwrite.
    """
    # Local import keeps the pure derivation/builder reusable without a cycle.
    from fyp_evaluation.validation.check_plan import validate_check_plan

    if errors := validate_checklist(checklist, normalization, schema_dir):
        raise ValueError("\n".join(errors))
    check_matrix(matrix, checklist)
    if not isinstance(call_configs, dict) or set(call_configs) != set(matrix):
        raise ValueError("Approved call configs must cover every matrix column")
    selected = list(matrix) if only is None else list(only)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(matrix):
        raise ValueError("only must select unique existing approved matrix call IDs")
    selected = set(selected)
    output_dir = Path(output_dir)
    documents = []
    for call_id in matrix:
        data = build_check_plan(call_id, checklist, normalization, matrix)
        errors = validate_check_plan(data, call_configs[call_id], checklist, normalization, matrix, schema_dir)
        if errors:
            raise ValueError("\n".join(errors))
        if call_id not in selected:
            continue
        path = output_dir / f"{call_id}.json"
        if not overwrite and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"Refusing to overwrite {path}; use explicit overwrite=True / --overwrite")
        if path.is_symlink():
            raise ValueError(f"Refusing to write through symlink: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"Output is not a regular file: {path}")
        documents.append((path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, serialized in documents:
        with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
    return [path for path, _ in documents]
