"""The temporal meaning of a cross-language edge-case variant, as placement constraints the G6 author and validator both apply.

A frozen language variant such as CROSS_LANGUAGE_ZH_TO_EN says the earlier state is spoken in one language and the later state in
the other. For edge cases with their own ordering (corrections, replacements, readbacks, references) the plan already carries ORDER
constraints, so the pins and the order agree. A competing-values distractor (family DISTRACTOR_COMPETING_VALUES) has no order
against the final transaction value, so the variant's "earlier state" (the distractor mentions) was never enforced. This module
derives it deterministically from the frozen call spec and the accepted G4 plan alone: every distractor mention of such an
assignment is placed before the final event of each target question. Nothing in G0-G4 changes, and no model is involved.
"""

CROSS_LANGUAGE = "CROSS_LANGUAGE"
DISTRACTOR_FAMILY = "DISTRACTOR_COMPETING_VALUES"


def language_order_constraints(spec, plan):
    """ORDER constraints (`before` distractor, `after` final event) for every cross-language distractor assignment of the call."""
    events, checks = plan["planned_events"], {check["question_id"]: check for check in plan["checks"]}
    constraints, seen = [], set()
    for edge in spec["edge_cases"]:
        variant = edge.get("language_variant") or "NONE"
        if not variant.startswith(CROSS_LANGUAGE) or edge.get("family") != DISTRACTOR_FAMILY:
            continue
        distractors = [event["event_id"] for event in events if edge["assignment_id"] in event["assignment_ids"] and event["kind"] == "DISTRACTOR_MENTION"]
        finals = sorted({checks[q]["final_event_id"] for q in edge["target_question_ids"] if q in checks and checks[q].get("final_event_id")})
        for before in distractors:
            for after in finals:
                if (before, after) not in seen:
                    seen.add((before, after))
                    constraints.append({"type": "ORDER", "before": before, "after": after,
                                        "source": {"type": "LANGUAGE_VARIANT", "id": edge["assignment_id"], "variant": variant}})
    return constraints
