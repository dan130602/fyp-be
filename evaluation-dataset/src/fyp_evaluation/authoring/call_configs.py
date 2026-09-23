"""Deterministic, unfilled call-config scaffolds; no narrative generation."""

import json
from pathlib import Path

from .coverage import check_matrix, compile_expected
from fyp_evaluation.validation.call_config import validate_call_config
from fyp_evaluation.validation.schemas import validate_checklist

CALL_CONFIG_VERSION = "0.5.0"
SCENARIO_INTENT_PLACEHOLDER = "TODO_AUTHOR_SCENARIO_INTENT"


def build_call_config(call_id, checklist, normalization, matrix):
    """Build in checklist order from validated inputs, without mutating them."""
    check_matrix(matrix, checklist)
    if call_id not in matrix:
        raise ValueError(f"{call_id}: no column in the approved matrix")
    return {
        "schema_version": CALL_CONFIG_VERSION,
        "call_id": call_id,
        "metadata": {
            "language": "en-SG",
            "scenario": "private_banking_investment_advice_and_order",
        },
        "target_duration_seconds": {"min": 120, "max": 180},
        "scenario_facts": [],
        "intended_answers": [
            {"question_id": item["id"],
             "expected": compile_expected(item, normalization, matrix[call_id][item["id"]]),
             "scenario_intent": SCENARIO_INTENT_PLACEHOLDER}
            for item in checklist
        ],
    }


def write_call_configs(output_dir, checklist, normalization, matrix, schema_dir, *, overwrite=False):
    """Validate the entire batch before writing. Refuse existing paths by default.

    Exclusive creation also prevents a file created after preflight from being
    overwritten. Writes are per file, not a transactional batch; on I/O failure,
    already-written scaffolds remain available and are protected on a retry.
    """
    if errors := validate_checklist(checklist, normalization, schema_dir):
        raise ValueError("\n".join(errors))
    check_matrix(matrix, checklist)
    output_dir = Path(output_dir)
    documents = []
    for call_id in matrix:
        data = build_call_config(call_id, checklist, normalization, matrix)
        if errors := validate_call_config(data, checklist, normalization, matrix, schema_dir):
            raise ValueError("\n".join(errors))
        path = output_dir / f"{call_id}.json"
        # Broken symlinks count as occupied paths, too.
        if not overwrite and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"Refusing to overwrite {path}; use explicit overwrite=True / --overwrite")
        if path.is_symlink():
            raise ValueError(f"Refusing to write through symlink: {path}")
        documents.append((path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, serialized in documents:
        with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
    return [path for path, _ in documents]
