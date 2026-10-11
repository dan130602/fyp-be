"""Compile canonical gold deterministically from approved authoring documents."""

from copy import deepcopy
from pathlib import Path

from .coverage import compile_expected
from fyp_evaluation.validation.call_config import validate_call_config
from fyp_evaluation.validation.dialogue import validate_speaker_alternation
from fyp_evaluation.validation.check_plan import _same_value, validate_check_plan
from fyp_evaluation.validation.evidence import resolve_evidence, validate_dialogue_against_plan
from fyp_evaluation.validation.schemas import validate
from fyp_evaluation.validation.spoken import verify_spoken_value
from fyp_evaluation.validation.tokens import validate_critical_token_coverage

_SCHEMAS = Path(__file__).resolve().parents[3] / "schemas"


def compile_ground_truth(call_config, check_plan, dialogue, checklist, normalization,
                         matrix, *, schema_dir=_SCHEMAS):
    """Return a validated 0.6.0 canonical object or raise ValueError; never write.

    No input is mutated. Evidence and annotation gates are separate. Semantic
    review/approval remains external; free-text constraints are not interpreted.
    """
    call_id = call_config.get("call_id")
    for name, document in (("check_plan", check_plan), ("dialogue", dialogue)):
        if document.get("call_id") != call_id:
            raise ValueError(f"call_config.call_id {call_id!r} must equal "
                             f"{name}.call_id {document.get('call_id')!r}")
    if call_id not in matrix:
        raise ValueError(f"{call_id}: no column in approved matrix")
    errors = validate_call_config(call_config, checklist, normalization, matrix, schema_dir, complete=True)
    errors.extend(validate_check_plan(check_plan, call_config, checklist, normalization, matrix,
                                      schema_dir, complete=True))
    errors.extend(validate(dialogue, "dialogue", schema_dir))
    if errors:
        raise ValueError("\n".join(errors))
    errors = validate_speaker_alternation(dialogue)
    errors.extend(validate_dialogue_against_plan(dialogue, check_plan))
    if errors:
        raise ValueError("\n".join(errors))
    turns = [{"turn_id": index, **deepcopy(turn)} for index, turn in enumerate(dialogue["turns"])]
    items = {item["id"]: item for item in checklist}
    checks = []
    for check in check_plan["checks"]:
        qid = check["question_id"]
        expected = deepcopy(check["expected"])
        locked = compile_expected(items[qid], normalization, matrix[call_id][qid])
        if not _same_value(expected, locked):
            raise ValueError(f"{qid}: expected differs from compiled matrix answer (value or type)")
        plan = deepcopy(check["evidence_plan"])
        ids, matching_errors = resolve_evidence(plan, turns)
        if matching_errors:
            raise ValueError("\n".join(error["message"] for error in matching_errors))
        if expected["kind"] == "extraction" and expected["expected_value"] is not None:
            if not any(verify_spoken_value(expected, item["text"]) is None
                       for item in plan["planned_evidence"]):
                raise ValueError(f"{qid}: planned evidence does not denote locked extraction value")
        checks.append({
            "question_id": qid, "expected": expected, "evidence_plan": plan,
            "gold_evidence_turns": sorted(set(ids)),
            "gold_evidence_text": "\n".join(item["text"] for item in plan["planned_evidence"])
                                  if plan["mode"] == "presence" else "",
        })
    by_id = {check["question_id"]: check["gold_evidence_turns"] for check in checks}
    for edge in check_plan["ordering_constraints"]:
        before, after = edge["before"], edge["after"]
        if max(by_id[before]) >= min(by_id[after]):
            raise ValueError(f"ordering_constraints: all {before} evidence must precede {after}")
    token_errors, _advisory = validate_critical_token_coverage(
        dialogue, checklist, normalization, matrix, schema_dir=schema_dir,
    )
    if token_errors:
        raise ValueError("\n".join(token_errors))
    result = {"schema_version": "0.6.0", "call_id": call_id,
              "metadata": deepcopy(call_config["metadata"]), "checks": checks, "dialogue": turns}
    if errors := validate(result, "ground_truth", schema_dir):
        raise ValueError("\n".join(errors))
    return result
