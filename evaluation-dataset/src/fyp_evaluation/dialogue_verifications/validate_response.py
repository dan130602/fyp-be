"""The G7 response validator and evidence expansion.

`expand(parsed, ctx)` checks every deterministic rule a report must satisfy and, when it does, returns the rows, issues and
quality findings with each citation resolved against ONE version of the call's dialogue: the speaker, the exact source
substring of the cited turn, every matching occurrence span and the planned events those spans overlap. A report that fails
any rule is REJECTED (response budget); a report that passes is a completed verification whatever its verdict. Nothing here
judges semantics: Python checks coverage, order, linkage, severity consistency, sizes and that a quotation is real speech.

Citation rules (mechanical, never a semantic engine):
- every cited quote must be real speech of the cited turn;
- a PRESENCE check marked MATCH cites one of its own evidence events;
- an edge case marked MATCH cites an event that carries its assignment, when such an event exists;
- every MISMATCH issue cites the speech it rests on, except `MISSING_SUPPORT` (an ABSENT required act, value or realisation);
- every quality finding cites the turn it concerns;
- UNCLEAR issues, `MISSING_SUPPORT` issues and omission findings may cite nothing: absence is never "proved" by a quotation.
"""

import re

from jsonschema import Draft202012Validator

from .config import (
    ASSESSMENTS, MAX_EVIDENCE, MAX_EXPLANATION, MAX_ISSUES, MAX_QUALITY, MAX_QUOTE, MAX_RATIONALE, UNGROUNDED_MISMATCH_CODE,
)
from .grounding import locate

ISSUE_ID = re.compile(r"^I[1-9][0-9]*$")
QUALITY_ID = re.compile(r"^Q[1-9][0-9]*$")
SEVERITY = {"MATCH": 0, "UNCLEAR": 1, "MISMATCH": 2}


def _expand_evidence(ctx, item, where):
    """(expanded citation, error) for one model citation."""
    turn_id, quote = item["turn_id"], item["quote"]
    if len(quote) > MAX_QUOTE:
        return None, f"{where}: quote longer than {MAX_QUOTE} characters"
    if isinstance(turn_id, bool) or not isinstance(turn_id, int) or not 0 <= turn_id < len(ctx.turns):
        return None, f"{where}: turn_id {turn_id!r} is not a turn of this dialogue (0..{len(ctx.turns) - 1})"
    turn = ctx.turns[turn_id]
    exact, spans, reason = locate(quote, turn["text"])
    if reason:
        return None, f"{where}: {reason} (turn {turn_id})"
    overlapping = {span["event_id"] for span in ctx.spans_by_turn.get(turn_id, [])
                   if any(start < span["end"] and span["start"] < end for start, end in spans)}
    return {"turn_id": turn_id, "speaker": turn["speaker"], "quote": exact, "spans": spans,
            "event_ids": sorted(overlapping, key=ctx.event_index.get)}, None


def _expand_list(ctx, items, where, errors):
    if len(items) > MAX_EVIDENCE:
        errors.append(f"{where}: more than {MAX_EVIDENCE} evidence items")
    expanded = []
    for index, item in enumerate(items):
        result, error = _expand_evidence(ctx, item, f"{where}.evidence[{index}]")
        if error:
            errors.append(error)
        else:
            expanded.append(result)
    return expanded


def _text_errors(value, limit, where, label, errors):
    if not value.strip():
        errors.append(f"{where}: {label} is empty")
    elif len(value) > limit:
        errors.append(f"{where}: {label} longer than {limit} characters")


def _expand_rows(ctx, rows, key, link_field, label, issue_by_id, errors):
    """Common handling of checks and edge-case rows: text limits, issue linkage, severity agreement and evidence."""
    expanded = []
    for row in rows:
        name, assessment = row[key], row["assessment"]
        where = f"{label} {name}"
        if assessment not in ASSESSMENTS:
            errors.append(f"{where}: unknown assessment {assessment!r}")
            continue
        _text_errors(row["rationale"], MAX_RATIONALE, where, "rationale", errors)
        if len(set(row["issue_ids"])) != len(row["issue_ids"]):
            errors.append(f"{where}: issue_ids repeat")
        linked = []
        for issue_id in row["issue_ids"]:
            issue = issue_by_id.get(issue_id)
            if issue is None:
                errors.append(f"{where}: references unknown issue {issue_id!r}")
            elif name not in issue[link_field]:
                errors.append(f"{where}: references issue {issue_id} whose {link_field} do not include {name}")
            else:
                linked.append(issue)
        if assessment == "MATCH" and row["issue_ids"]:
            errors.append(f"{where}: a MATCH row must not reference issues")
        if assessment != "MATCH":
            if not row["issue_ids"]:
                errors.append(f"{where}: a {assessment} row needs at least one linked issue")
            elif linked and max(SEVERITY[i["assessment"]] for i in linked) != SEVERITY[assessment]:
                errors.append(f"{where}: assessment {assessment} must equal the most severe linked issue")
        expanded.append((row, _expand_list(ctx, row["evidence"], where, errors)))
    return expanded


def expand(parsed, ctx):
    """(checks, edge_cases, issues, quality_findings, errors): the expanded report parts, or the list of rule violations."""
    errors = []
    checks, edges, issues, quality = parsed["checks"], parsed["edge_cases"], parsed["issues"], parsed["quality_findings"]
    ids = [check["question_id"] for check in checks]
    if ids != ctx.order:
        errors.append(f"checks must list exactly the 20 checklist questions in checklist order; got {ids}")
    edge_ids = [edge["assignment_id"] for edge in edges]
    if edge_ids != ctx.assignment_ids:
        errors.append(f"edge_cases must list exactly the call's edge-case assignments in the given order {ctx.assignment_ids}; got {edge_ids}")
    if errors:
        return None, None, None, None, errors
    if len(issues) > MAX_ISSUES:  # not in the provider-facing schema (nested bounds are rejected by Gemini): enforced here
        errors.append(f"more than {MAX_ISSUES} issues")
    if len(quality) > MAX_QUALITY:
        errors.append(f"more than {MAX_QUALITY} quality findings")
    issue_ids = [issue["issue_id"] for issue in issues]
    if len(set(issue_ids)) != len(issue_ids):
        errors.append("issue_id values must be distinct")
    errors += [f"issue_id {i!r} must look like I1, I2, ..." for i in issue_ids if not ISSUE_ID.match(i)]
    quality_ids = [finding["quality_id"] for finding in quality]
    if len(set(quality_ids)) != len(quality_ids):
        errors.append("quality_id values must be distinct")
    errors += [f"quality_id {i!r} must look like Q1, Q2, ..." for i in quality_ids if not QUALITY_ID.match(i)]
    issue_by_id = {issue["issue_id"]: issue for issue in issues}
    assignments = set(ctx.assignment_ids)

    expanded_issues = []
    for issue in issues:
        where = f"issue {issue['issue_id']}"
        _text_errors(issue["explanation"], MAX_EXPLANATION, where, "explanation", errors)
        if len(set(issue["question_ids"])) != len(issue["question_ids"]):
            errors.append(f"{where}: question_ids repeat")
        if len(set(issue["assignment_ids"])) != len(issue["assignment_ids"]):
            errors.append(f"{where}: assignment_ids repeat")
        if unknown := [a for a in issue["assignment_ids"] if a not in assignments]:
            errors.append(f"{where}: assignment_ids {unknown} are not assignments of this call")
        evidence = _expand_list(ctx, issue["evidence"], where, errors)
        if issue["assessment"] == "MISMATCH" and issue["code"] != UNGROUNDED_MISMATCH_CODE and not issue["evidence"]:
            errors.append(f"{where}: a MISMATCH issue coded {issue['code']} must cite the dialogue speech it rests on "
                          f"(only {UNGROUNDED_MISMATCH_CODE} may have no citation)")
        expanded_issues.append({**issue, "evidence": evidence})

    expanded_checks = []
    for row, evidence in _expand_rows(ctx, checks, "question_id", "question_ids", "check", issue_by_id, errors):
        qid, plan_check = row["question_id"], ctx.checks[row["question_id"]]
        if row["assessment"] == "MATCH" and plan_check["evidence_mode"] == "PRESENCE":
            own = set(plan_check["evidence_event_ids"])
            if not any(own & set(item["event_ids"]) for item in evidence):
                errors.append(f"check {qid}: a PRESENCE check marked MATCH must cite speech of one of its own evidence events {sorted(own)}")
        expanded_checks.append({"question_id": qid, "kind": plan_check["kind"], "expected": ctx.gold[qid], "assessment": row["assessment"],
                                "rationale": row["rationale"], "evidence": evidence, "issue_ids": list(row["issue_ids"])})

    expanded_edges = []
    for row, evidence in _expand_rows(ctx, edges, "assignment_id", "assignment_ids", "edge case", issue_by_id, errors):
        aid, carriers = row["assignment_id"], ctx.events_of_assignment[row["assignment_id"]]
        if row["assessment"] == "MATCH" and carriers and not any(set(carriers) & set(item["event_ids"]) for item in evidence):
            errors.append(f"edge case {aid}: marked MATCH, it must cite speech of one of the events that carry it {carriers}")
        edge = ctx.edge_cases[aid]
        expanded_edges.append({"assignment_id": aid, "edge_case_id": edge["edge_case_id"], "target_question_ids": list(edge["target_question_ids"]),
                               "assessment": row["assessment"], "rationale": row["rationale"], "evidence": evidence,
                               "issue_ids": list(row["issue_ids"])})

    by_question = {c["question_id"]: c for c in checks}
    by_assignment = {e["assignment_id"]: e for e in edges}
    for issue in issues:  # linkage in the other direction: an issue naming a row must be referenced by that row
        for qid in issue["question_ids"]:
            if qid not in by_question:
                errors.append(f"issue {issue['issue_id']}: names unknown question {qid}")
            elif issue["issue_id"] not in by_question[qid]["issue_ids"]:
                errors.append(f"issue {issue['issue_id']}: names {qid} but check {qid} does not reference it")
        for aid in issue["assignment_ids"]:
            if aid in by_assignment and issue["issue_id"] not in by_assignment[aid]["issue_ids"]:
                errors.append(f"issue {issue['issue_id']}: names {aid} but edge case {aid} does not reference it")

    expanded_quality = []
    for finding in quality:
        where = f"quality finding {finding['quality_id']}"
        _text_errors(finding["explanation"], MAX_EXPLANATION, where, "explanation", errors)
        if not finding["evidence"]:
            errors.append(f"{where}: a quality finding must cite the turn it concerns")
        expanded_quality.append({**finding, "evidence": _expand_list(ctx, finding["evidence"], where, errors)})
    if errors:
        return None, None, None, None, errors
    return expanded_checks, expanded_edges, expanded_issues, expanded_quality, []


class DialogueVerificationValidator:
    def __init__(self, response_schema):
        Draft202012Validator.check_schema(response_schema)
        self._response = Draft202012Validator(response_schema)

    def errors(self, parsed, ctx):
        schema_errors = [f"schema: {e.json_path}: {e.message}" for e in self._response.iter_errors(parsed)]
        if schema_errors:
            return schema_errors
        return expand(parsed, ctx)[4]
