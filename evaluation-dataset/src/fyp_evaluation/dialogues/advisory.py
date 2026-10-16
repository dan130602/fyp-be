"""Python-derived advisory-recommendation data (shared by the G6 brief and the G7 brief), offline conflict detection and an advisory lint.

Every call has `call_intents` ADVISORY then EXECUTION. When the plan has no planned recommendation event (the C07 plan structure; C07 FAIL
is an absence check), the dialogue must still contain a clear employee recommendation of the frozen product/action WITHOUT a concrete
rationale. Everything the author and the verifier need is derived here from frozen inputs only (the call spec, the G3 final product and the
G4 events and constraints); there is no keyword, no model and no call ID:

* `product_name`: the G3 FINAL product; `direction`: BUY/SELL from gold T05, only when EXECUTION is also an intent.
* `place_after_event_ids`: events carrying an edge-case assignment that targets the product (T01) or the direction (T05) (reference
  selection, corrections, replacement). Naming the product before them would give the answer away.
* `place_before_event_ids`: the final confirmation exchange (events that carry C08).
* `keep_apart_from_event_ids`: events that carry C01 or C06 (objective, willingness to accept loss); adjacency would read as a reason.

`advisory_conflicts` finds contradictory placement requirements offline (before any paid request). `advisory_lint` is ADVISORY ONLY: it
lists signals for a human; G6 acceptance stays structural and meaning belongs to G7.
"""

import re

PRODUCT_QUESTION, DIRECTION_QUESTION = "T01", "T05"
FINAL_CONFIRMATION_QUESTION = "C08"
KEEP_APART_QUESTIONS = ("C01", "C06")
RECOMMENDATION_QUESTION = "C07"

BUY_WORDS = ("buy", "buying", "purchase", "purchasing", "invest in", "investing in", "subscribe", "subscribing")
SELL_WORDS = ("sell", "selling", "sale", "reduce", "reducing", "redeem", "redeeming", "dispose", "disposing")
RATIONALE_CUES = ("because", "since", "given", "as you", "as your", "so that", "which is why", "that is why", "for you", "your goal", "your objective",
                  "your need", "your risk", "suits", "suit you", "fits", "fit your", "ideal", "suitable", "appropriate for", "diversif", "benefit",
                  "performance", "returns", "growth", "income", "stable", "track record", "low risk", "good option", "attractive", "well positioned")
CONFIRMATION_REACTIONS = ("understand", "agree", "confirm", "proceed", "go ahead", "please do", "yes please", "let's do", "lets do", "sounds good")


def advisory_recommendation(spec, config, events, constraints=None):
    """The brief data for the unplanned recommendation, or None when no unplanned recommendation is required.

    None when ADVISORY is not an intent or the plan already carries the recommendation in planned events.
    """
    intents = list(spec["metadata"]["call_intents"])
    if "ADVISORY" not in intents or any(RECOMMENDATION_QUESTION in event["question_ids"] for event in events):
        return None
    final = next(product for product in config["products"] if "FINAL" in product["roles"])
    data = {"product_name": final["name"]}
    if "EXECUTION" in intents:
        gold = {entry["question_id"]: entry["expected"] for entry in spec["gold"]}
        direction = gold.get(DIRECTION_QUESTION, {}).get("expected_value")
        if direction in ("BUY", "SELL"):
            data["direction"] = direction
    determining = {edge["assignment_id"] for edge in spec["edge_cases"] if {PRODUCT_QUESTION, DIRECTION_QUESTION} & set(edge["target_question_ids"])}
    data["place_after_event_ids"] = [e["event_id"] for e in events if determining & set(e["assignment_ids"])]
    data["place_before_event_ids"] = [e["event_id"] for e in events if FINAL_CONFIRMATION_QUESTION in e["question_ids"]]
    data["keep_apart_from_event_ids"] = [e["event_id"] for e in events if set(KEEP_APART_QUESTIONS) & set(e["question_ids"])]
    return data


def _groups(events, constraints):
    parent = {e["event_id"]: e["event_id"] for e in events}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in constraints:
        if c["type"] == "SAME_TURN":
            a, b = find(c["events"][0]), find(c["events"][1])
            if a != b:
                parent[b] = a
    return {e["event_id"]: find(e["event_id"]) for e in events}


def _reaches(edges, start):
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        for before, after in edges:
            if before == node and after not in seen:
                seen.add(after)
                stack.append(after)
    return seen


def advisory_conflicts(data, events, constraints):
    """Contradictory placement requirements for this call (empty list = a valid placement exists). Offline, from frozen constraints."""
    if data is None:
        return []
    speakers = {e["event_id"]: e["authoring"]["speaker"] for e in events}
    edges = [(c["before"], c["after"]) for c in constraints if c["type"] == "ORDER"]
    group = _groups(events, constraints)
    after, before, apart = data["place_after_event_ids"], data["place_before_event_ids"], set(data["keep_apart_from_event_ids"])
    errors = []
    for settled in after:
        for confirmation in before:
            if settled == confirmation or confirmation in _reaches(edges, settled) and settled in _reaches(edges, confirmation):
                errors.append(f"{settled} must precede the recommendation but is itself a final-confirmation event or cyclic with {confirmation}")
            elif settled in _reaches(edges, confirmation):
                errors.append(f"{settled} must precede the recommendation, but ORDER constraints place it after the final confirmation event {confirmation}")
            elif group[settled] == group[confirmation]:
                members = [e for e in group if group[e] == group[settled]]
                if {speakers[m] for m in members} != {"EMPLOYEE"}:
                    errors.append(f"{settled} and {confirmation} share a turn that is not an employee turn, so no employee recommendation fits between them")
                elif apart & set(members):
                    errors.append(f"the only turn left for the recommendation (shared by {settled} and {confirmation}) also holds "
                                  f"objective/loss-tolerance evidence {sorted(apart & set(members))}")
    return errors


# ---- advisory lint (report-only) ------------------------------------------------------------------------------------------------------------

def _outside_events(turn_text, spans):
    chars = list(turn_text)
    for span in spans:
        for i in range(span["start"], span["end"]):
            chars[i] = " "
    return "".join(chars)


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def advisory_lint(data, artifact):
    """Report-only signals about the unplanned recommendation in an accepted G6 dialogue; never a gate and never a verdict.

    Returns {"required": bool, "signals": [str], "recommendation": {...} | None}. Meaning, rationale and naturalness belong to G7.
    """
    if data is None:
        return {"required": False, "signals": [], "recommendation": None}
    turns, alignment = artifact["turns"], artifact["event_alignment"]
    spans = {}
    turn_of = {}
    for span in alignment:
        spans.setdefault(span["turn_id"], []).append(span)
        turn_of[span["event_id"]] = span["turn_id"]
    name = data["product_name"].lower()
    found = None
    for turn in turns:
        if turn["speaker"] != "EMPLOYEE":
            continue
        for sentence in _sentences(_outside_events(turn["text"], spans.get(turn["turn_id"], []))):
            if name in sentence.lower():
                found = {"turn_id": turn["turn_id"], "sentence": sentence}
                break
        if found:
            break
    signals = []
    if not found:
        return {"required": True, "signals": ["no employee sentence outside the planned events names the product"], "recommendation": None}
    lowered, t = found["sentence"].lower(), found["turn_id"]
    words = BUY_WORDS if data.get("direction") == "BUY" else SELL_WORDS if data.get("direction") == "SELL" else ()
    if words and not any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words):
        signals.append(f"turn {t}: no {data['direction']} action word in the sentence")
    claim = lowered.replace(name, " ")  # the product's own name (e.g. "... Income Fund", "... Bond 2031") is neither rationale wording nor an order detail
    cues = [cue for cue in RATIONALE_CUES if re.search(rf"\b{re.escape(cue)}", claim)]
    if cues:
        signals.append(f"turn {t}: possible rationale wording {cues} (a lint signal only; G7 judges by meaning)")
    if re.search(r"\d", claim) or re.search(r"\b(units?|shares?|euros?|dollars?|yen|yuan|limit order|market order)\b", claim):
        signals.append(f"turn {t}: the sentence appears to state an order detail")
    for event_id in data["place_before_event_ids"]:
        if turn_of.get(event_id, 10**9) < t:
            signals.append(f"turn {t}: comes after final-confirmation event {event_id} (turn {turn_of[event_id]})")
    for event_id in data["place_after_event_ids"]:
        if turn_of.get(event_id, -1) > t:
            signals.append(f"turn {t}: comes before {event_id} (turn {turn_of[event_id]}), which must precede it")
    for event_id in data["keep_apart_from_event_ids"]:
        if turn_of.get(event_id) in (t, t - 1):
            signals.append(f"turn {t}: adjacent to objective/loss-tolerance event {event_id} (turn {turn_of[event_id]})")
    reply = next((x for x in turns if x["turn_id"] == t + 1), None)
    if reply is None or reply["speaker"] != "CLIENT":
        signals.append(f"turn {t}: no client turn follows the recommendation")
    else:
        opening = (_sentences(_outside_events(reply["text"], spans.get(reply["turn_id"], []))) or [""])[0].lower()
        if not opening:
            signals.append(f"turn {reply['turn_id']}: the client turn opens with a planned event, not a brief acknowledgement")
        elif any(word in opening for word in CONFIRMATION_REACTIONS) or re.search(r"\d", opening):
            signals.append(f"turn {reply['turn_id']}: the client's reaction may be more than a mild acknowledgement: {opening!r}")
    return {"required": True, "signals": signals, "recommendation": found}
