"""Check-plan integrity and cross-file checks; no evidence matching or execution."""

import re

from fyp_evaluation.authoring.coverage import compile_expected
from fyp_evaluation.authoring.check_plans import derive_evidence_mode
from .call_config import contains_placeholder, validate_call_config
from .schemas import validate_answers
from .spoken import verify_spoken_value


def _same_value(left, right):
    """Deep equality preserving Python types, including bool/int/float distinctions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_value(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


def validate_forbidden_patterns(document):
    """Check Python regex syntax on an already schema-valid plan or ground truth.

    Compile only; never search dialogue or treat regex absence as semantic proof.
    """
    errors = []
    for check in document["checks"]:
        for index, pattern in enumerate(check["evidence_plan"]["forbidden_patterns"]):
            try:
                re.compile(pattern)
            except (re.error, OverflowError, RecursionError) as error:
                errors.append(f"{check['question_id']}/forbidden_patterns/{index}: invalid Python regex: {error}")
    return errors


def _ordering_errors(plan):
    modes = {check["question_id"]: check["evidence_plan"]["mode"] for check in plan["checks"]}
    successors = {qid: set() for qid in modes}
    indegree = dict.fromkeys(modes, 0)
    errors = []
    for edge in plan["ordering_constraints"]:
        before, after = edge["before"], edge["after"]
        if before not in modes or after not in modes:
            errors.append("ordering_constraints must reference existing question IDs")
            continue
        if before == after:
            errors.append("ordering_constraints must not self-reference")
        if modes[before] != "presence" or modes[after] != "presence":
            errors.append("ordering_constraints may name only presence-mode checks")
        if after not in successors[before]:
            successors[before].add(after)
            indegree[after] += 1
    ready = [qid for qid, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        qid = ready.pop()
        visited += 1
        for after in successors[qid]:
            indegree[after] -= 1
            if indegree[after] == 0:
                ready.append(after)
    if visited != len(modes):
        errors.append("ordering_constraints must not contain a cycle")
    return errors


def validate_check_plan(plan, call_config, checklist, normalization, matrix,
                        schema_dir, *, complete=False):
    """Check identity first, then representation, locked values and derived modes.

    Complete mode also verifies planned spoken forms for the pilot's text/DOB
    extractions. Other scalar contracts retain their existing validation rules.
    """
    if isinstance(plan, dict) and isinstance(call_config, dict):
        if plan.get("call_id") != call_config.get("call_id"):
            return [f"check_plan.call_id {plan.get('call_id')!r} must equal "
                    f"call_config.call_id {call_config.get('call_id')!r}"]
    errors = validate_answers(plan, "check_plan", checklist, normalization, schema_dir)
    if errors:
        return errors
    config_errors = validate_call_config(call_config, checklist, normalization, matrix, schema_dir)
    if config_errors:
        return [f"call_config: {error}" for error in config_errors]
    call_id = plan["call_id"]
    if call_id not in matrix:
        return [f"{call_id}: no column in approved matrix"]
    if [check["question_id"] for check in plan["checks"]] != [item["id"] for item in checklist]:
        errors.append("checks must follow approved checklist order")
    items = {item["id"]: item for item in checklist}
    config_answers = {answer["question_id"]: answer["expected"] for answer in call_config["intended_answers"]}
    evidence_ids = set()
    authored_text = []
    for check in plan["checks"]:
        qid = check["question_id"]
        item = items[qid]
        try:
            expected = compile_expected(item, normalization, matrix[call_id][qid])
            mode = derive_evidence_mode(item, expected)
        except ValueError as error:
            errors.append(f"{qid}: {error}")
            continue
        if not _same_value(check["expected"], expected):
            errors.append(f"{qid}: expected differs from compiled matrix answer (value or type)")
        if not _same_value(check["expected"], config_answers[qid]):
            errors.append(f"{qid}: expected differs from approved call-config answer (value or type)")
        evidence = check["evidence_plan"]
        if evidence["mode"] != mode:
            errors.append(f"{qid}: evidence mode must be derived as {mode}")
        if evidence["mode"] == "absence":
            required_speaker = "ANY" if qid == "K01" else "EMPLOYEE"
            if evidence["required_speaker"] != required_speaker:
                errors.append(f"{qid}: absence required_speaker must be {required_speaker}")
            authored_text.append((f"{qid}/absent_event", evidence["absent_event"]))
        else:
            for index, phrase in enumerate(evidence["planned_evidence"], start=1):
                evidence_id = phrase["evidence_id"]
                if evidence_id != f"{qid}_E{index}":
                    errors.append(f"{qid}: evidence IDs must be contiguous {qid}_E1, {qid}_E2, ...")
                if evidence_id in evidence_ids:
                    errors.append(f"Duplicate plan-wide evidence ID: {evidence_id}")
                evidence_ids.add(evidence_id)
                # C01/C02 can need both prompt and response; no blanket rule
                # forces all compliance evidence to come from one speaker.
                if item["type"] != "COMPLIANCE" and phrase["required_speaker"] != "CLIENT":
                    errors.append(f"{qid}: extraction presence evidence requires CLIENT")
                authored_text.append((f"{qid}/{evidence_id}/text", phrase["text"]))
        for field in ("content_constraints", "forbidden_patterns"):
            authored_text.extend((f"{qid}/{field}/{i}", text) for i, text in enumerate(evidence[field]))
        if (complete and expected["kind"] == "extraction"
                and expected["expected_value"] is not None
                and expected["value_type"] in {"text", "dob_dd_month_yyyy"}
                and evidence["mode"] == "presence"):
            # Verify against the matrix-derived object, never a revised spoken
            # value. Context items need not each repeat the full value.
            if not any(verify_spoken_value(expected, phrase["text"]) is None
                       for phrase in evidence["planned_evidence"]):
                errors.append(f"{qid}: planned evidence does not denote locked extraction value")
    errors.extend(validate_forbidden_patterns(plan))
    errors.extend(_ordering_errors(plan))
    if complete:
        errors.extend(f"Incomplete: {path} contains a placeholder" for path, text in authored_text
                      if contains_placeholder(text))
    return errors
