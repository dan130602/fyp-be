"""Compile one call spec into its check_plan_skeleton.

Walk order (deterministic, so event IDs are stable):
  1. compliance checks in checklist order: their presence events, then compliance edge cases;
  2. an order replacement, if assigned (it supplies the FINAL event of every transaction question);
  3. extraction questions in checklist order: supplier edge case or plain FINAL, then augmenting edge cases;
  4. cross-question edge cases (long distance);
  5. the G0 ordering rules;
  6. the 20 checks.
"""

from fyp_evaluation.call_specs.compile import SCHEMA_VERSION, HASHING, text_hash

from .edge_cases import COMPILERS, missing_compilers, unknown_compilers
from .languages import pin_languages
from .model import BOTH, CLIENT, CONSTRAINT_TYPES, EMPLOYEE, EVENT_KINDS, EVENT_STATUSES, Context, Plan

STAGE = "G2B"
GENERATOR_VERSION = "1.0.0"
SPEC_DIR = "authoring/check_plan_skeletons"
RECORD = "authoring/check_plan_skeleton_record.json"


def skeleton_path(call_id):
    return f"{SPEC_DIR}/{call_id}.json"


def _presence_events(plan, q):
    """Plain evidence for a compliance question whose frozen verdict requires its condition to occur."""
    ctx = plan.ctx
    ref, pattern = ctx.condition_ref(q), ctx.semantics[q]["presence_pattern"]
    if pattern == "EMPLOYEE_ONLY":
        ids = [plan.event("COMPLIANCE_ACT", "CONTEXT", [EMPLOYEE], [q], role="ACT", condition_ref=ref)]
    elif pattern == "ANY_SPEAKER":
        ids = [plan.event("COMPLIANCE_ACT", "CONTEXT", BOTH, [q], role="ACT", condition_ref=ref)]
    else:  # EXCHANGE: employee act, then client response
        act = plan.event("COMPLIANCE_ACT", "CONTEXT", [EMPLOYEE], [q], role="ACT", condition_ref=ref)
        response = plan.event("COMPLIANCE_ACT", "CONTEXT", [CLIENT], [q], role="RESPONSE", condition_ref=ref)
        plan.order(act, response, {"type": "PRESENCE_PATTERN", "question_id": q})
        ids = [act, response]
    plan.presence_events[q] = ids


def _compile_events(plan):
    ctx = plan.ctx
    unknown = sorted({a["edge_case_id"] for a in ctx.assignments} - ctx.catalog.keys())
    if unknown:
        raise ValueError(f"{ctx.spec['call_id']}: edge cases {unknown} are not in the catalog")
    by_stage = {}
    for a in ctx.assignments:
        by_stage.setdefault(COMPILERS[a["edge_case_id"]].stage, []).append(a)

    compiled = set()

    def run(a):
        COMPILERS[a["edge_case_id"]].fn(plan, a)
        compiled.add(a["assignment_id"])

    def targeting(stage, q):
        return [a for a in by_stage.get(stage, []) if q in a["target_question_ids"]]

    for q in ctx.compliance_ids:
        if ctx.compliance_mode(q) == "PRESENCE":
            _presence_events(plan, q)
        for a in targeting("COMPLIANCE", q):
            run(a)

    bundles = by_stage.get("BUNDLE", [])
    if len(bundles) > 1:
        raise ValueError(f"{ctx.spec['call_id']}: more than one order replacement")
    for a in bundles:
        run(a)

    for q in ctx.extraction_ids:
        suppliers = targeting("SUPPLY", q)
        if len(suppliers) > 1:
            raise ValueError(f"{ctx.spec['call_id']}/{q}: more than one supplier edge case")
        for a in suppliers:
            if q in plan.final_of:
                raise ValueError(f"{a['assignment_id']} ({a['edge_case_id']}): {q} already has a FINAL event from the order "
                                 f"replacement, so the two edge cases conflict")
            run(a)
        if q not in plan.final_of:
            plan.event("VALUE_STATEMENT", "FINAL", [CLIENT], [q], values=plan.values(q, ctx.gold[q]))
        for a in targeting("AUGMENT", q):
            run(a)

    for a in by_stage.get("CROSS", []):
        run(a)

    if skipped := sorted({a["assignment_id"] for a in ctx.assignments} - compiled):
        raise ValueError(f"{ctx.spec['call_id']}: assignments {skipped} were not compiled "
                         f"(their targets do not match their compiler's stage)")


def _compile_ordering_rules(plan):
    """C03 after the C02 disclosure; C08 after the final transaction state (G0 ordering_rules only)."""
    events = plan.events
    for rule in plan.ctx.ordering_rules:
        subjects = [e["event_id"] for e in events
                    if e["kind"] == "COMPLIANCE_ACT" and e["role"] == "ACT" and rule["question_id"] in e["question_ids"]]
        after = rule["after"]
        if after["type"] == "CHECK_EVIDENCE":
            references = [e["event_id"] for e in events if e["kind"] == "COMPLIANCE_ACT" and after["question_id"] in e["question_ids"]]
        else:
            references = list(dict.fromkeys(plan.final_of[q] for q in after["question_ids"]))
        for reference in references:
            for subject in subjects:
                plan.order(reference, subject, {"type": "ORDERING_RULE", "id": rule["id"]})


def _related_assignments(ctx, q):
    """Assignments that target q or whose frozen verdict requirements mention it."""
    related = []
    for a in ctx.assignments:
        requirements = ctx.catalog[a["edge_case_id"]].get("verdict_requirements", {})
        if q in a["target_question_ids"] or q in requirements:
            related.append(a["assignment_id"])
    return sorted(related)


def _compile_checks(plan):
    ctx, checks = plan.ctx, []
    for item in ctx.checklist:
        q = item["id"]
        if item["type"] == "COMPLIANCE":
            mode = ctx.compliance_mode(q)
            check = {
                "question_id": q, "kind": "COMPLIANCE", "expected": ctx.expected[q], "evidence_mode": mode,
                "presence_pattern": ctx.semantics[q]["presence_pattern"],
            }
            if mode == "PRESENCE":
                check["required_condition_ref"] = ctx.condition_ref(q)
            else:
                check["forbidden_condition_ref"] = ctx.condition_ref(q)
                check["absence"] = {"scope": "ENTIRE_DIALOGUE"}
            check["evidence_event_ids"] = list(plan.presence_events.get(q, []))
            if q in plan.wording:
                check["wording_constraint"] = plan.wording[q]
        else:
            assigned = q in ctx.edge_form
            check = {
                "question_id": q, "kind": "EXTRACTION", "expected": ctx.expected[q], "evidence_mode": "PRESENCE",
                "resolution_ref": f"{q}.resolution_rule",
                "evidence_event_ids": [e["event_id"] for e in plan.events if q in e["question_ids"]],
                "final_event_id": plan.final_of[q],
                "gold_restatement": {"policy": "ASSIGNED_FORM_ONLY" if assigned else "ANY_PERMITTED_FORM",
                                     "spoken_form": ctx.gold_form[q]},
                "non_gold_spoken_form": ctx.non_gold_form[q],
                "forbidden_spoken_forms": ctx.forbidden_forms[q],
            }
        check["assignment_ids"] = _related_assignments(ctx, q)
        checks.append(check)
    return checks


def _attach_spoken_form_assignments(plan):
    """Mark every event that restates a gold value under an assigned spoken form with that assignment."""
    for q, (_, assignment) in plan.ctx.edge_form.items():
        for event in plan.events:
            entry = event.get("values", {}).get(q)
            if entry and entry["spoken_form"]["requirement"] == "REQUIRED":
                plan.attach([event["event_id"]], assignment)


def compile_skeleton(spec, spec_sha256, documents, g0_hashes):
    """Return the check_plan_skeleton for one call spec (pure; raises ValueError on any contradiction)."""
    if gap := missing_compilers(documents["edge_case_catalog"]):
        raise ValueError(f"No G2B compiler is registered for edge cases {gap}")
    if extra := unknown_compilers(documents["edge_case_catalog"]):
        raise ValueError(f"G2B compilers exist for edge cases missing from the catalog: {extra}")
    ctx = Context(spec, documents)
    plan = Plan(ctx)
    _compile_events(plan)
    _compile_ordering_rules(plan)
    _attach_spoken_form_assignments(plan)
    pin_languages(plan)
    checks = _compile_checks(plan)
    return {
        "schema_version": spec["schema_version"],
        "stage": STAGE,
        "dataset_id": spec["dataset_id"],
        "call_id": spec["call_id"],
        "call_spec_sha256": spec_sha256,
        "source": dict(g0_hashes),
        "planned_events": plan.events,
        "checks": checks,
        "constraints": plan.constraints,
    }


def build_skeletons(inputs):
    return [
        compile_skeleton(inputs.specs[call_id], inputs.spec_hashes[call_id], inputs.documents, inputs.g0_hashes)
        for call_id in inputs.call_ids
    ]


def _count(items, key):
    counts = {}
    for item in items:
        counts[item[key]] = counts.get(item[key], 0) + 1
    return counts


def build_record(inputs, skeletons, skeleton_texts):
    """Provenance record: input hashes and one hash per skeleton, plus deterministic summary counts."""
    events = [e for s in skeletons for e in s["planned_events"]]
    constraints = [c for s in skeletons for c in s["constraints"]]

    def ordered(counts, vocabulary):
        return {key: counts.get(key, 0) for key in vocabulary}

    multilingual = "language_profiles_sha256" in inputs.g0_hashes
    record = {
        "schema_version": skeletons[0]["schema_version"] if skeletons else SCHEMA_VERSION,
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "generator": {"package": "fyp_evaluation.check_plan_skeletons", "version": GENERATOR_VERSION},
        "hashing": HASHING,
        "g2a_inputs": {"call_spec_record_sha256": inputs.record_sha256},
        "g0_inputs": dict(inputs.g0_hashes),
        "skeletons": {
            s["call_id"]: {"call_spec_sha256": s["call_spec_sha256"], "skeleton_sha256": text_hash(skeleton_texts[s["call_id"]])}
            for s in skeletons
        },
        "summary": {
            "skeletons": len(skeletons),
            "checks_per_skeleton": len(skeletons[0]["checks"]) if skeletons else 0,
            "planned_events": len(events),
            "constraints": len(constraints),
            "event_kinds": ordered(_count(events, "kind"), EVENT_KINDS),
            "event_statuses": ordered(_count(events, "status"), EVENT_STATUSES),
            "constraint_types": ordered(_count(constraints, "type"), CONSTRAINT_TYPES),
            "edge_case_assignments": sum(len(inputs.specs[c]["edge_cases"]) for c in inputs.call_ids),
        },
    }
    if multilingual:
        record["summary"]["events_with_required_language"] = sum(1 for e in events if "required_language" in e)
    return record
