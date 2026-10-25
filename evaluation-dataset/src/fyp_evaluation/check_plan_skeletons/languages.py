"""Language pins for edge-case language variants (multilingual releases only).

A catalog variant with `event_languages` fixes the spoken language of specific planned events: `source` for the
earlier state of the mechanism and `target` for the later one (cross-language variants), or only `target` for the
value mentions that carry an assigned English spoken form (EN_SURFACE). Ordinary code-switching is never pinned;
every other event's language is left to G4 within the call's language profile. Pinning one event to two languages
is a contradiction and raises.
"""


def _own(plan, a):
    return [e for e in plan.events if a["assignment_id"] in e["assignment_ids"]]


def _ids(events):
    return [e["event_id"] for e in events]


def _single_correction(plan, a):
    events = _own(plan, a)
    return _ids(e for e in events if e["status"] == "SUPERSEDED"), _ids(e for e in events if e["status"] == "FINAL")


def _multiple_corrections(plan, a):
    events = _own(plan, a)
    return _ids([next(e for e in events if e["status"] == "SUPERSEDED")]), _ids(e for e in events if e["status"] == "FINAL")


def _readback(plan, a):
    events = _own(plan, a)
    return _ids(e for e in events if e["kind"] == "VALUE_STATEMENT"), _ids(e for e in events if e["kind"] == "READBACK")


def _order_replacement(plan, a):
    events = [e for e in _own(plan, a) if e["kind"] == "ORDER_STATEMENT"]
    return _ids(e for e in events if e["status"] == "SUPERSEDED"), _ids(e for e in events if e["status"] == "FINAL")


def _reference(plan, a):
    events = _own(plan, a)
    return _ids(e for e in events if e["kind"] == "ANTECEDENT_MENTION"), _ids(e for e in events if e["kind"] == "REFERENCE_SELECTION")


def _against_final(kind):
    def select(plan, a):
        (q,) = a["target_question_ids"]
        return _ids(e for e in _own(plan, a) if e["kind"] == kind), [plan.final_of[q]]
    return select


def _assigned_form(plan, a):
    """EN_SURFACE: every mention of the target that must use the assigned spoken form."""
    return [], [e["event_id"] for e in plan.events for q in a["target_question_ids"]
                if e.get("values", {}).get(q, {}).get("spoken_form", {}).get("requirement") == "REQUIRED"]


SELECTORS = {
    "EC_CORRECTION_SAME_TURN": _single_correction,
    "EC_CORRECTION_CROSS_TURN": _single_correction,
    "EC_MULTIPLE_CORRECTIONS": _multiple_corrections,
    "EC_EMPLOYEE_READBACK_ERROR": _readback,
    "EC_ONE_FIELD_READBACK_ERROR": _readback,
    "EC_COMPLETE_ORDER_REPLACEMENT": _order_replacement,
    "EC_MULTIPLE_CURRENCIES": _against_final("DISTRACTOR_MENTION"),
    "EC_MULTIPLE_MONETARY_VALUES": _against_final("DISTRACTOR_MENTION"),
    "EC_PREVIOUS_CURRENCY_REFERENCE": _reference,
    "EC_PRODUCT_REFERENCE": _reference,
    "EC_PREVIOUS_QUANTITY_REFERENCE": _reference,
    "EC_FALSE_START": _against_final("FALSE_START"),
    "EC_DOUBLE_TRIPLE_DIGITS": _assigned_form,
    "EC_OH_VS_ZERO": _assigned_form,
    "EC_FINANCIAL_SHORTHAND": _assigned_form,
    "EC_LARGE_NUMBER_EXPRESSION": _assigned_form,
}


def pin_languages(plan):
    """Apply every assigned variant's event languages (no-op in a monolingual release)."""
    for a in plan.ctx.assignments:
        languages = (plan.ctx.variants.get(a["assignment_id"]) or {}).get("event_languages")
        if not languages:
            continue
        select = SELECTORS.get(a["edge_case_id"])
        if select is None:
            raise ValueError(f"{a['assignment_id']}: no language pins are defined for {a['edge_case_id']}")
        source, target = select(plan, a)
        if ("source" in languages and not source) or not target:
            raise ValueError(f"{a['assignment_id']}: variant {a['language_variant']} has no events to pin")
        for event_id in source if "source" in languages else []:
            plan.pin(event_id, languages["source"], a)
        for event_id in target:
            plan.pin(event_id, languages["target"], a)
