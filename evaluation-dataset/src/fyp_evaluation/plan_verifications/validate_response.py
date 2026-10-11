"""The G5 response validator and evidence expansion.

`expand(parsed, ctx)` checks every deterministic rule a report must satisfy and, when it does, returns the checks and issues
with each evidence item resolved against ONE version of the call's G3 config and G4 plan: the artifact JSON Pointer of the
cited value and the exact source substring of the quote. A report that fails any rule is REJECTED (response budget); a
report that passes is a completed verification whatever its verdict. Nothing here judges semantics.
"""

import re

from jsonschema import Draft202012Validator

from .config import ASSESSMENTS, MAX_EVIDENCE, MAX_EXPLANATION, MAX_ISSUES, MAX_QUOTE, MAX_RATIONALE
from .grounding import ground

ISSUE_ID = re.compile(r"^I[1-9][0-9]*$")
SEVERITY = {"MATCH": 0, "UNCLEAR": 1, "MISMATCH": 2}


def _pointer_tokens(pointer):
    if not pointer.startswith("/"):
        raise ValueError("not a JSON Pointer")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def _resolve_pointer(document, pointer):
    value = document
    for token in _pointer_tokens(pointer):
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and not (len(token) > 1 and token.startswith("0")) and int(token) < len(value):
            value = value[int(token)]
        else:
            raise KeyError(token)
    return value


def _expand_evidence(ctx, item, where):
    """(expanded evidence, error) for one model evidence item."""
    source, ref, quote = item["source"], item["ref"], item["quote"]
    if quote is not None and len(quote) > MAX_QUOTE:
        return None, f"{where}: quote longer than {MAX_QUOTE} characters"
    if source == "G4":
        if ref not in ctx.events_by_id:
            return None, f"{where}: G4 ref {ref!r} is not an event of this plan"
        text = ctx.planned_text(ref)
        pointer = f"/planned_events/{ctx.event_index[ref]}/authoring/planned_text"
        event_id = ref
    else:
        if not (ref in ("/products", "/scenario") or ref.startswith(("/products/", "/scenario/"))):
            return None, f"{where}: G3 ref {ref!r} must be a JSON Pointer under /products or /scenario"
        try:
            value = _resolve_pointer(ctx.g3, ref)
        except (KeyError, ValueError):
            return None, f"{where}: G3 ref {ref!r} does not resolve in the call config"
        text, pointer, event_id = value, ref, None
        if quote is not None and not isinstance(value, str):
            return None, f"{where}: a quote needs a G3 ref that points at a text value"
    if quote is None:
        return {"source": source, "event_id": event_id, "pointer": pointer, "quote": None}, None
    exact, reason = ground(quote, text)
    if reason:
        return None, f"{where}: {reason} ({ref})"
    return {"source": source, "event_id": event_id, "pointer": pointer, "quote": exact}, None


def _expand_list(ctx, items, where, errors):
    expanded = []
    for index, item in enumerate(items):
        result, error = _expand_evidence(ctx, item, f"{where}.evidence[{index}]")
        if error:
            errors.append(error)
        else:
            expanded.append(result)
    return expanded


def expand(parsed, ctx):
    """(checks, issues, errors): expanded report parts, or the list of rule violations."""
    errors = []
    checks, issues = parsed["checks"], parsed["issues"]
    ids = [check["question_id"] for check in checks]
    if ids != ctx.order:
        errors.append(f"checks must list exactly the 20 checklist questions in checklist order; got {ids}")
        return None, None, errors
    if len(issues) > MAX_ISSUES:  # not in the provider-facing schema (nested bounds are rejected by Gemini): enforced here
        errors.append(f"more than {MAX_ISSUES} issues")
    issue_ids = [issue["issue_id"] for issue in issues]
    if len(set(issue_ids)) != len(issue_ids):
        errors.append("issue_id values must be distinct")
    errors += [f"issue_id {i!r} must look like I1, I2, ..." for i in issue_ids if not ISSUE_ID.match(i)]
    issue_by_id = {issue["issue_id"]: issue for issue in issues}
    assignments = set(ctx.assignment_ids)

    expanded_issues = []
    for issue in issues:
        where = f"issue {issue['issue_id']}"
        if not issue["explanation"].strip():
            errors.append(f"{where}: explanation is empty")
        elif len(issue["explanation"]) > MAX_EXPLANATION:
            errors.append(f"{where}: explanation longer than {MAX_EXPLANATION} characters")
        if len(set(issue["question_ids"])) != len(issue["question_ids"]):
            errors.append(f"{where}: question_ids repeat")
        if unknown := [a for a in issue["assignment_ids"] if a not in assignments]:
            errors.append(f"{where}: assignment_ids {unknown} are not assignments of this call")
        if len(issue["evidence"]) > MAX_EVIDENCE:
            errors.append(f"{where}: more than {MAX_EVIDENCE} evidence items")
        expanded_issues.append({**issue, "evidence": _expand_list(ctx, issue["evidence"], where, errors)})

    expanded_checks = []
    for check in checks:
        qid, assessment = check["question_id"], check["assessment"]
        where = f"check {qid}"
        if assessment not in ASSESSMENTS:
            errors.append(f"{where}: unknown assessment {assessment!r}")
            continue
        rationale = check["rationale"]
        if not rationale.strip():
            errors.append(f"{where}: rationale is empty")
        elif len(rationale) > MAX_RATIONALE:
            errors.append(f"{where}: rationale longer than {MAX_RATIONALE} characters")
        if len(set(check["issue_ids"])) != len(check["issue_ids"]):
            errors.append(f"{where}: issue_ids repeat")
        linked = []
        for issue_id in check["issue_ids"]:
            issue = issue_by_id.get(issue_id)
            if issue is None:
                errors.append(f"{where}: references unknown issue {issue_id!r}")
            elif qid not in issue["question_ids"]:
                errors.append(f"{where}: references issue {issue_id} whose question_ids do not include {qid}")
            else:
                linked.append(issue)
        if assessment == "MATCH" and check["issue_ids"]:
            errors.append(f"{where}: a MATCH check must not reference issues")
        if assessment != "MATCH":
            if not check["issue_ids"]:
                errors.append(f"{where}: a {assessment} check needs at least one linked issue")
            elif linked and max(SEVERITY[i["assessment"]] for i in linked) != SEVERITY[assessment]:
                errors.append(f"{where}: assessment {assessment} must equal the most severe linked issue")
        if len(check["evidence"]) > MAX_EVIDENCE:
            errors.append(f"{where}: more than {MAX_EVIDENCE} evidence items")
        evidence = _expand_list(ctx, check["evidence"], where, errors)
        plan_check = ctx.checks[qid]
        if assessment == "MATCH" and plan_check["evidence_mode"] == "PRESENCE":
            own = set(plan_check["evidence_event_ids"])
            if not any(item["source"] == "G4" and item["event_id"] in own and item["quote"] for item in evidence):
                errors.append(f"{where}: a PRESENCE check marked MATCH must quote one of its own evidence events {sorted(own)}")
        expanded_checks.append({"question_id": qid, "kind": plan_check["kind"], "expected": ctx.gold[qid], "assessment": assessment,
                                "rationale": rationale, "evidence": evidence, "issue_ids": list(check["issue_ids"])})

    for issue in issues:  # linkage in the other direction: an issue naming a question must be referenced by that check
        for qid in issue["question_ids"]:
            check = next(c for c in checks if c["question_id"] == qid)
            if issue["issue_id"] not in check["issue_ids"]:
                errors.append(f"issue {issue['issue_id']}: names {qid} but check {qid} does not reference it")
    if errors:
        return None, None, errors
    return expanded_checks, expanded_issues, []


class VerificationValidator:
    def __init__(self, response_schema):
        Draft202012Validator.check_schema(response_schema)
        self._response = Draft202012Validator(response_schema)

    def errors(self, parsed, ctx):
        schema_errors = [f"schema: {e.json_path}: {e.message}" for e in self._response.iter_errors(parsed)]
        if schema_errors:
            return schema_errors
        return expand(parsed, ctx)[2]
