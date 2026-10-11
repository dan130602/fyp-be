"""Deterministic structural acceptance of a G6 dialogue response. G6 acceptance is NOT semantic verification (that is G7).

Rules: schema; at least two non-blank turns; well-formed whole-token markers; every planned event exactly once in a turn
by its G4 speaker; strict EMPLOYEE/CLIENT alternation; every frozen and derived placement constraint; no planned text
repeated outside event spans; no unplanned K01/K02 value surface; length within the recorded policy's band. Other value
surfaces outside events are not errors: `review_notes` records them for G7.
"""

from jsonschema import Draft202012Validator

from fyp_evaluation.validation.dialogue import validate_speaker_alternation

from .render import occurrences_outside, render, spans_by_turn, word_count


def _schema_errors(validator, parsed):
    return [f"schema: {error.json_path}: {error.message}"[:300] for error in validator.iter_errors(parsed)]


def _constraint_errors(ctx, positions):
    errors = []
    for c in ctx.constraints:
        kind = c["type"]
        if kind == "ORDER":
            before, after = positions[c["before"]], positions[c["after"]]
            if not before < after:
                errors.append(f"ORDER: {c['before']} must come before {c['after']}")
            continue
        first, second = (positions[event_id][0] for event_id in c["events"])
        if kind == "SAME_TURN" and first != second:
            errors.append(f"SAME_TURN: {c['events'][0]} and {c['events'][1]} must be in the same turn")
        elif kind == "NOT_SAME_TURN" and first == second:
            errors.append(f"NOT_SAME_TURN: {c['events'][0]} and {c['events'][1]} must be in different turns")
        elif kind == "MIN_INTERVENING_TURNS" and abs(first - second) - 1 < c["turns"]:
            errors.append(f"MIN_INTERVENING_TURNS: at least {c['turns']} turns must lie between {c['events'][0]} and {c['events'][1]} "
                          f"(found {max(abs(first - second) - 1, 0)})")
    return errors


def review_notes(ctx, turns, alignment):
    """Non-identity value surfaces outside event spans (for G7 review; never a rejection)."""
    spans, notes = spans_by_turn(alignment), set()
    for turn in turns:
        for question_id, surface in ctx.value_surfaces:
            if occurrences_outside(turn["text"], surface, spans.get(turn["turn_id"], [])):
                notes.add((turn["turn_id"], question_id, surface))
    return [{"turn_id": t, "question_id": q, "surface": s} for t, q, s in sorted(notes)]


class DialogueValidator:
    def __init__(self, response_schema):
        self.schema = Draft202012Validator(response_schema)

    def errors(self, parsed, ctx):
        errors = _schema_errors(self.schema, parsed)
        if errors:
            return errors
        if len(parsed["turns"]) < 2:
            return ["a dialogue needs at least two turns"]
        turns, alignment, errors = render(ctx, parsed)
        errors += [f"turn {t['turn_id']}: empty turn" for t in turns if not t["text"].strip()]
        counts = {}
        for span in alignment:
            counts[span["event_id"]] = counts.get(span["event_id"], 0) + 1
        errors += [f"{event_id}: placed {n} times (exactly once required)" for event_id, n in sorted(counts.items()) if n > 1]
        missing = [event_id for event_id in ctx.texts if event_id not in counts]
        errors += [f"{event_id}: missing from the dialogue" for event_id in missing]
        for span in alignment:
            speaker = turns[span["turn_id"]]["speaker"]
            if speaker != ctx.speakers[span["event_id"]]:
                errors.append(f"{span['event_id']}: placed in a {speaker} turn but its speaker is {ctx.speakers[span['event_id']]}")
        errors += validate_speaker_alternation({"turns": turns})
        if not missing and all(n == 1 for n in counts.values()):
            positions = {span["event_id"]: (span["turn_id"], span["start"]) for span in alignment}
            errors += _constraint_errors(ctx, positions)
        spans = spans_by_turn(alignment)
        for turn in turns:
            inside = spans.get(turn["turn_id"], [])
            for event_id, text in ctx.texts.items():
                if occurrences_outside(turn["text"], text, inside):
                    errors.append(f"turn {turn['turn_id']}: the text of {event_id} is repeated outside its event marker")
            for question_id, surface in ctx.identity_surfaces:
                if occurrences_outside(turn["text"], surface, inside):
                    errors.append(f"turn {turn['turn_id']}: unplanned {question_id} value {surface!r} outside the planned events")
        low, high = ctx.bounds["accepted_words"]
        words = word_count(turns)
        if not low <= words <= high:
            errors.append(f"length: {words} words; the {ctx.profile} band of {ctx.length_policy['version']} is {low}-{high}")
        return errors
