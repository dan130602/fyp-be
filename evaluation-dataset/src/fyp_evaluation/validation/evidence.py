"""Exact evidence matching, shared by compilation and canonical validation.

These helpers examine speakers and text only, never Entity WER annotations.
Callers validate document schemas separately before evidence validation.
"""

import re


def resolve_evidence(evidence_plan, turns):
    """Return (IDs in planned-item order, structured matching errors).

    Successful IDs retain duplicates when multiple items resolve to one turn;
    canonical callers take the sorted union. Error candidate IDs are diagnostic,
    never a choice of evidence. Intermediate turns use their Python list index;
    canonical turns retain their supplied IDs for existing integrity checks.
    """
    resolved, errors = [], []
    if evidence_plan["mode"] == "absence":
        return resolved, errors
    for item in evidence_plan["planned_evidence"]:
        matches = [
            turn.get("turn_id", index) for index, turn in enumerate(turns)
            if turn["speaker"] == item["required_speaker"]
            and item["text"] in turn["text"]
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
        else:
            code = "not_found" if not matches else "ambiguous"
            errors.append({
                "code": code,
                "evidence_id": item["evidence_id"],
                "matching_turn_ids": matches,
                "message": f"{item['evidence_id']}: planned evidence {code.replace('_', ' ')}; eligible turns {matches}",
            })
    return resolved, errors


def validate_dialogue_against_plan(dialogue, check_plan):
    """Return evidence errors; never access or require critical_tokens.

    ID equality is checked before examining any checks or turns. Forbidden
    patterns are literal Python regex safeguards, not semantic absence proofs.
    """
    if dialogue.get("call_id") != check_plan.get("call_id"):
        return [f"dialogue.call_id {dialogue.get('call_id')!r} must equal "
                f"check_plan.call_id {check_plan.get('call_id')!r}"]
    errors = []
    for check in check_plan["checks"]:
        qid, plan = check["question_id"], check["evidence_plan"]
        if plan["mode"] == "presence":
            ids, matching_errors = resolve_evidence(plan, dialogue["turns"])
            errors.extend(f"{qid}: {error['message']}" for error in matching_errors)
            if not matching_errors and ids != sorted(ids):
                errors.append(f"{qid}: planned evidence order disagrees with resolved turn order")
        for pattern in plan["forbidden_patterns"]:
            try:
                compiled = re.compile(pattern)
            except (re.error, OverflowError, RecursionError) as error:
                errors.append(f"{qid}: invalid forbidden pattern: {error}")
                continue
            for index, turn in enumerate(dialogue["turns"]):
                if (plan["mode"] == "absence" and plan["required_speaker"] != "ANY"
                        and turn["speaker"] != plan["required_speaker"]):
                    continue
                if compiled.search(turn["text"]):
                    errors.append(f"{qid}: forbidden pattern {pattern!r} matched turn {index}")
    return errors
