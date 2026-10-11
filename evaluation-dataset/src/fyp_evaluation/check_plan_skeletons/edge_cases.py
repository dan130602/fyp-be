"""Edge-case compilers, registered directly by edge_case_id (no metadata in G0).

Each compiler turns one assignment's locked G1 parameters into planned events and relative
constraints. A compiler never invents a value: final values are checked against gold, and
superseded / rejected / distractor values must differ from it. `stage` tells the compile
walk when to run it:

  BUNDLE      an order replacement that supplies the FINAL event of every transaction question
  SUPPLY      supplies the FINAL event of its single target question
  AUGMENT     adds structure around the FINAL event that the plain compile (or a supplier) created
  COMPLIANCE  adds structure around a compliance question's own presence events
  CROSS       relates the FINAL events of several questions
"""

from collections import namedtuple
from itertools import combinations

from .model import CLIENT, EMPLOYEE, NOT_EXPLICIT

Compiler = namedtuple("Compiler", "stage fn")


def _single_target(assignment):
    targets = assignment["target_question_ids"]
    if len(targets) != 1:
        raise ValueError(f"{assignment['assignment_id']}: {assignment['edge_case_id']} needs exactly one target")
    return targets[0]


# ---- correction family (SUPPLY) --------------------------------------------------------------------
def _single_correction(plan, a):
    """SAME_TURN and CROSS_TURN: superseded value, then a final correction; relative placement from the catalog."""
    q, p, template = _single_target(a), a["parameters"], plan.template(a)
    plan.require_gold(a, q, p["final_value"])
    first = plan.event("VALUE_STATEMENT", "SUPERSEDED", [CLIENT], [q], a, values=plan.nongold(a, q, p["initial_value"]))
    final = plan.event("VALUE_CORRECTION", "FINAL", [CLIENT], [q], a, values=plan.values(q, p["final_value"]),
                       corrects_event_id=first)
    source = plan.edge_source(a)
    plan.order(first, final, source)
    (plan.same_turn if template["same_turn"] else plan.not_same_turn)(first, final, source)
    if template.get("min_intervening_turns"):
        plan.min_gap(first, final, template["min_intervening_turns"], source)


def _multiple_corrections(plan, a):
    """Ordered superseded states, then FINAL; no turn gap is prescribed."""
    q, p, template = _single_target(a), a["parameters"], plan.template(a)
    plan.require_gold(a, q, p["final_value"])
    states = p["superseded_values"]
    if len(states) + 1 < template["minimum_state_count"]:
        raise ValueError(f"{a['assignment_id']}: needs at least {template['minimum_state_count'] - 1} superseded values")
    previous, ids = None, []
    for position, value in enumerate(states):
        if position == 0:
            event_id = plan.event("VALUE_STATEMENT", "SUPERSEDED", [CLIENT], [q], a, values=plan.nongold(a, q, value))
        else:
            event_id = plan.event("VALUE_CORRECTION", "SUPERSEDED", [CLIENT], [q], a, values=plan.nongold(a, q, value),
                                  corrects_event_id=previous)
        ids.append(event_id)
        previous = event_id
    final = plan.event("VALUE_CORRECTION", "FINAL", [CLIENT], [q], a, values=plan.values(q, p["final_value"]),
                       corrects_event_id=previous)
    for before, after in zip(ids, ids[1:] + [final]):
        plan.order(before, after, plan.edge_source(a))


def _readback(min_fields_key):
    def compile_readback(plan, a):
        """CONVENTION: the earlier client STATED instruction, the employee READBACK, the client correction."""
        q, p, template = _single_target(a), a["parameters"], plan.template(a)
        plan.require_gold(a, q, p["final_value"])
        minimum = template.get(min_fields_key, 1) if min_fields_key else 1
        stated = plan.event("VALUE_STATEMENT", "STATED", [CLIENT], [q], a, values=plan.values(q, p["final_value"]))
        wrong = plan.nongold(a, q, p["employee_wrong_value"])[q]
        readback = plan.event("READBACK", "REJECTED", [EMPLOYEE], [q], a, readback={
            "must_include_question_ids": [q],
            "min_field_count": minimum,
            "wrong_question_id": q,
            "wrong_value": wrong["value"],
            "wrong_value_spoken_form": wrong["spoken_form"],
            "other_included_fields_must_match_final": True,
            "other_included_fields_must_follow_their_final_events": True,
        })
        final = plan.event("VALUE_CORRECTION", "FINAL", [CLIENT], [q], a, values=plan.values(q, p["final_value"]),
                           corrects_event_id=readback)
        source = plan.edge_source(a)
        plan.order(stated, readback, source)
        plan.order(readback, final, source)
    return compile_readback


def _order_replacement(plan, a):
    """Initial ORDER_STATEMENT, ORDER_INVALIDATION, final ORDER_STATEMENT covering every transaction question."""
    ctx, p = plan.ctx, a["parameters"]
    order_ids = ctx.order_ids
    initial, final = p["initial_order"], p["final_order"]
    if set(initial) != set(order_ids) or set(final) != set(order_ids):
        raise ValueError(f"{a['assignment_id']}: orders must cover exactly {order_ids}")
    for q in order_ids:
        plan.require_gold(a, q, final[q], "final order value")
    changed = [q for q in order_ids if initial[q] != final[q]]
    if changed != a["target_question_ids"]:
        raise ValueError(f"{a['assignment_id']}: changed fields {changed} differ from targets {a['target_question_ids']}")
    first = plan.event("ORDER_STATEMENT", "SUPERSEDED", [CLIENT], order_ids, a,
                       values={q: plan.values(q, initial[q])[q] for q in order_ids})
    invalidation = plan.event("ORDER_INVALIDATION", "CONTEXT", [CLIENT], order_ids, a, invalidates_event_id=first)
    last = plan.event("ORDER_STATEMENT", "FINAL", [CLIENT], order_ids, a,
                      values={q: plan.values(q, final[q])[q] for q in order_ids})
    source = plan.edge_source(a)
    plan.order(first, invalidation, source)
    plan.order(invalidation, last, source)


# ---- distractors (AUGMENT) -------------------------------------------------------------------------
def _distractor_speakers(plan, a):
    return plan.ctx.catalog[a["edge_case_id"]]["allowed_distractor_speakers"]


def _currency_distractor(plan, a):
    q, p = _single_target(a), a["parameters"]
    plan.require_gold(a, q, p["final_currency"])
    plan.event("DISTRACTOR_MENTION", "DISTRACTOR", _distractor_speakers(plan, a), [q], a,
               values=plan.nongold(a, q, p["distractor_currency"]), distractor_role=p["distractor_role"])


def _monetary_distractors(plan, a):
    q, p = _single_target(a), a["parameters"]
    plan.require_gold(a, q, p["final_transaction_amount"])
    for distractor in p["distractor_values"]:
        plan.event("DISTRACTOR_MENTION", "DISTRACTOR", _distractor_speakers(plan, a), [q], a,
                   values=plan.nongold(a, q, distractor["amount"]), distractor_role=distractor["role"],
                   currency=distractor["currency"])


# ---- references (SUPPLY) ---------------------------------------------------------------------------
def _antecedent_speaker(plan, a):
    speaker = a["parameters"]["antecedent_speaker"]
    if speaker not in plan.ctx.catalog[a["edge_case_id"]]["allowed_antecedent_speakers"]:
        raise ValueError(f"{a['assignment_id']}: antecedent speaker {speaker} is not allowed")
    return [speaker]


def _selection(plan, a, q, antecedent):
    return plan.event("REFERENCE_SELECTION", "FINAL", [CLIENT], [q], a,
                      values={q: {"value": plan.ctx.gold[q], "spoken_form": NOT_EXPLICIT}}, refers_to_event_id=antecedent)


def _value_reference(parameter):
    def compile_reference(plan, a):
        """Antecedent mention, then a client reference that does not restate the value."""
        q, p = _single_target(a), a["parameters"]
        plan.require_gold(a, q, p[parameter])
        antecedent = plan.event("ANTECEDENT_MENTION", "ANTECEDENT", _antecedent_speaker(plan, a), [q], a,
                                values=plan.values(q, p[parameter]))
        selection = _selection(plan, a, q, antecedent)
        plan.order(antecedent, selection, plan.edge_source(a))
    return compile_reference


def _product_reference(plan, a):
    """G1's candidate_products order is the locked mention order; each candidate has its own antecedent event."""
    q, p = _single_target(a), a["parameters"]
    plan.require_gold(a, q, p["selected_product"], "selected product")
    candidates = p["candidate_products"]
    if len(candidates) < 2 or p["selected_product"] not in candidates:
        raise ValueError(f"{a['assignment_id']}: needs at least two candidates including the selection")
    speaker = _antecedent_speaker(plan, a)
    ids = [plan.event("ANTECEDENT_MENTION", "ANTECEDENT", speaker, [q], a, values=plan.values(q, name))
           for name in candidates]
    selection = _selection(plan, a, q, ids[candidates.index(p["selected_product"])])
    source = plan.edge_source(a)
    for before, after in zip(ids, ids[1:]):
        plan.order(before, after, source)
    plan.order(ids[-1], selection, source)


# ---- spoken forms, false start, long distance --------------------------------------------------------
def _spoken_form(plan, a):
    """The REQUIRED form was fixed in the context; this verifies the assignment's own parameters agree with it."""
    q = _single_target(a)
    if q not in plan.ctx.edge_form:
        raise ValueError(f"{a['assignment_id']}: no spoken form in normalization.json is enabled by {a['edge_case_id']}")
    parameters = a["parameters"]
    plan.require_gold(a, q, parameters["canonical_value"], "canonical value")
    if "spoken_form" in parameters and plan.ctx.gold_form[q].get("text") != parameters["spoken_form"]:
        raise ValueError(f"{a['assignment_id']}: locked spoken text was not carried into the required form")


def _false_start(plan, a):
    q, p = _single_target(a), a["parameters"]
    plan.require_gold(a, q, p["final_value"])
    template = plan.template(a)
    fragment = plan.event("FALSE_START", "ABANDONED", [CLIENT], [q], a, incomplete=True)
    final = plan.final_of[q]
    source = plan.edge_source(a)
    plan.order(fragment, final, source)
    if template["same_turn"]:
        plan.same_turn(fragment, final, source)


def _long_distance(plan, a):
    """At least min_turn_gap complete intervening turns between the FINAL events of the targets (unordered)."""
    p, template = a["parameters"], plan.template(a)
    targets = a["target_question_ids"]
    if p["target_question_ids"] != targets:
        raise ValueError(f"{a['assignment_id']}: parameter targets differ from the assignment targets")
    finals = [plan.final_of[q] for q in targets]
    if len(set(finals)) < len(finals):
        raise ValueError(f"{a['assignment_id']}: unsatisfiable long-distance dependency; targets {targets} "
                         f"resolve to the same FINAL event {finals}")
    for first, second in combinations(finals, 2):
        plan.min_gap(first, second, p["min_turn_gap"], plan.edge_source(a))


# ---- compliance semantics (COMPLIANCE) ---------------------------------------------------------------
def _guarantee(plan, a):
    q = _single_target(a)
    request = plan.event("GUARANTEE_REQUEST", "CONTEXT", [CLIENT], [q], a)
    refusal = plan.event("GUARANTEE_REFUSAL", "CONTEXT", [EMPLOYEE], [q], a)
    plan.order(request, refusal, plan.edge_source(a))


def _paraphrase(plan, a):
    q = _single_target(a)
    plan.attach(plan.presence_events.get(q, []), a)
    plan.wording[q] = plan.template(a)["wording_constraint"]


def _risk_not_verified(plan, a):
    """C02 presence plus C03 absence are carried by the frozen gold; this only links the assignment to C02's evidence."""
    for q, verdict in plan.ctx.catalog[a["edge_case_id"]]["verdict_requirements"].items():
        if verdict == "PASS":
            plan.attach(plan.presence_events.get(q, []), a)


def _weak_confirmation(plan, a):
    """The confirmation request and a weak acknowledgement may exist; the complete C08 exchange stays absent."""
    q = _single_target(a)
    request = plan.event("COMPLIANCE_ACT", "CONTEXT", [EMPLOYEE], [q], a, role="ACT", condition_ref=plan.ctx.condition_ref(q))
    weak = plan.event("WEAK_ACKNOWLEDGEMENT", "CONTEXT", [CLIENT], [q], a, strength="WEAK")
    plan.order(request, weak, plan.edge_source(a))


COMPILERS = {
    "EC_CORRECTION_SAME_TURN": Compiler("SUPPLY", _single_correction),
    "EC_CORRECTION_CROSS_TURN": Compiler("SUPPLY", _single_correction),
    "EC_MULTIPLE_CORRECTIONS": Compiler("SUPPLY", _multiple_corrections),
    "EC_EMPLOYEE_READBACK_ERROR": Compiler("SUPPLY", _readback(None)),
    "EC_ONE_FIELD_READBACK_ERROR": Compiler("SUPPLY", _readback("min_readback_fields")),
    "EC_COMPLETE_ORDER_REPLACEMENT": Compiler("BUNDLE", _order_replacement),
    "EC_MULTIPLE_CURRENCIES": Compiler("AUGMENT", _currency_distractor),
    "EC_MULTIPLE_MONETARY_VALUES": Compiler("AUGMENT", _monetary_distractors),
    "EC_PREVIOUS_CURRENCY_REFERENCE": Compiler("SUPPLY", _value_reference("final_currency")),
    "EC_PRODUCT_REFERENCE": Compiler("SUPPLY", _product_reference),
    "EC_PREVIOUS_QUANTITY_REFERENCE": Compiler("SUPPLY", _value_reference("final_quantity")),
    "EC_DOUBLE_TRIPLE_DIGITS": Compiler("AUGMENT", _spoken_form),
    "EC_OH_VS_ZERO": Compiler("AUGMENT", _spoken_form),
    "EC_FINANCIAL_SHORTHAND": Compiler("AUGMENT", _spoken_form),
    "EC_LARGE_NUMBER_EXPRESSION": Compiler("AUGMENT", _spoken_form),
    "EC_FALSE_START": Compiler("AUGMENT", _false_start),
    "EC_LONG_DISTANCE_DEPENDENCY": Compiler("CROSS", _long_distance),
    "EC_GUARANTEE_REQUEST_REFUSAL": Compiler("COMPLIANCE", _guarantee),
    "EC_PARAPHRASED_COMPLIANCE_STATEMENT": Compiler("COMPLIANCE", _paraphrase),
    "EC_RISK_EXPLAINED_UNDERSTANDING_NOT_VERIFIED": Compiler("COMPLIANCE", _risk_not_verified),
    "EC_WEAK_CONFIRMATION": Compiler("COMPLIANCE", _weak_confirmation),
}


def missing_compilers(catalog):
    """Catalog edge-case IDs that have no compiler (G2B refuses to run while any exist)."""
    return sorted({case["id"] for case in catalog["edge_cases"]} - COMPILERS.keys())


def unknown_compilers(catalog):
    return sorted(COMPILERS.keys() - {case["id"] for case in catalog["edge_cases"]})
