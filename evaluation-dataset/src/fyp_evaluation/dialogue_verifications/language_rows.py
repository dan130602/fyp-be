"""The language rows of a G7 report (schema 2.0): requirements Python derives from the frozen call spec, judged by the verifier.

A row is a blocking, separately reported language requirement: a MISMATCH makes the verdict INVALID exactly like a checklist
mismatch. Rows exist only where the frozen language profile states the requirement, so ENGLISH and MANDARIN calls have none
(their script rules are enforced mechanically in G6) unless an edge case carries a language variant. `mechanical_errors` are
the deterministic citation rules a MATCH must satisfy: they prove only that the cited speech is real and of the right kind
(for example that a "within-turn switch" quote does contain both languages), never that the requirement is met.
"""

from fyp_evaluation.validation.stage0 import HAN, LATIN

MIXED_BOTH = "MIXED_CLIENT_BOTH_LANGUAGES"
MIXED_SWITCH = "MIXED_CLIENT_WITHIN_TURN_SWITCH"
MIXED_SPREAD = "MIXED_BILINGUAL_DISTRIBUTED"
SINGLISH_CLIENT = "SINGLISH_CLIENT_REGISTER"
SINGLISH_EMPLOYEE = "SINGLISH_EMPLOYEE_PROFESSIONAL_ENGLISH"
VARIANT_PREFIX = "VARIANT_"

REQUIREMENTS = {
    MIXED_BOTH: "The client personally speaks substantive English and substantive Mandarin (names, identifiers, acronyms, currency codes and product names alone do not count).",
    MIXED_SWITCH: "At least one client turn contains a meaningful switch between English and Mandarin inside the same turn, at a natural boundary or for a financial term.",
    MIXED_SPREAD: "The client's bilingual speech is spread across the call, not confined to one place.",
    SINGLISH_CLIENT: "The client's speech is realised as natural colloquial Singapore English across the call.",
    SINGLISH_EMPLOYEE: "Every employee turn is professional English with no colloquial Singapore English phrasing or particles.",
}


def rows_for(spec):
    """[{row_id, requirement}] in report order for one call spec (empty when the call has no language row)."""
    profile = spec["metadata"].get("language_profile")
    ids = {"EN_ZH_MIXED": [MIXED_BOTH, MIXED_SWITCH, MIXED_SPREAD], "SINGLISH": [SINGLISH_CLIENT, SINGLISH_EMPLOYEE]}.get(profile, [])
    rows = [{"row_id": row_id, "requirement": REQUIREMENTS[row_id]} for row_id in ids]
    for edge in spec["edge_cases"]:
        variant = edge.get("language_variant")
        if variant and variant != "NONE":
            rows.append({"row_id": f"{VARIANT_PREFIX}{edge['assignment_id']}",
                         "requirement": f"Edge case {edge['assignment_id']} ({edge['edge_case_id']}) is realised in the language or switch direction "
                                        f"its language variant {variant} names."})
    return rows


def _has_both(text):
    return bool(HAN.search(text)) and bool(LATIN.search(text))


def mechanical_errors(ctx, row_id, evidence):
    """Citation rules for a MATCH row (evidence items are the expanded ones: turn_id, speaker, quote, event_ids)."""
    clients = [e for e in evidence if e["speaker"] == "CLIENT"]
    where = f"language row {row_id}"
    if row_id == MIXED_BOTH:
        if not (any(HAN.search(e["quote"]) for e in clients) and any(LATIN.search(e["quote"]) for e in clients)):
            return [f"{where}: a MATCH must quote client speech in Mandarin and client speech in English"]
    elif row_id == MIXED_SWITCH:
        if not any(_has_both(e["quote"]) for e in clients):
            return [f"{where}: a MATCH must quote one client turn that contains both English and Mandarin"]
    elif row_id == MIXED_SPREAD:
        turns = sorted({e["turn_id"] for e in clients})
        if len(turns) < 2 or turns[-1] - turns[0] < max(1, len(ctx.turns) // 3):
            return [f"{where}: a MATCH must quote client turns far apart in the call (at least a third of the dialogue apart)"]
    elif row_id == SINGLISH_CLIENT:
        if not clients:
            return [f"{where}: a MATCH must quote client speech"]
    elif row_id == SINGLISH_EMPLOYEE:
        if not any(e["speaker"] == "EMPLOYEE" for e in evidence):
            return [f"{where}: a MATCH must quote employee speech"]
    elif row_id.startswith(VARIANT_PREFIX):
        carriers = ctx.events_of_assignment.get(row_id[len(VARIANT_PREFIX):], [])
        if carriers and not any(set(carriers) & set(e["event_ids"]) for e in evidence):
            return [f"{where}: a MATCH must quote speech of one of the events that carry the edge case {carriers}"]
    return []
