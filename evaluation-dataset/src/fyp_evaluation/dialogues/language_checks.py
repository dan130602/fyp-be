"""Schema-2.0 deterministic checks on the speech the G6 model writes itself (everything outside the planned event spans).

Mechanical and narrow, like G4's guards: they enforce the frozen script rules on the model's own words and stop a planned
number from being extended or changed by the words glued next to it. They never judge meaning: whether supporting speech adds a
confirmation, a fee mention or market commentary, or whether a call really uses both languages, belongs to G7.
"""

import re

from fyp_evaluation.check_plans.guards import NUMERALS, VALUE_MARKERS
from fyp_evaluation.validation.stage0 import HAN, LATIN

from .render import outside_text

DIGIT = re.compile(r"[0-9]")
_NUMBER_CHARS = set(NUMERALS) | set("点0123456789")


def script_errors(ctx, turns, spans):
    """The frozen script rules applied to the model's own words (event spans are exempt: G4 already guarded them)."""
    rules = ctx.language_requirements["script_rules"]
    found = []
    for turn in turns:
        outside = outside_text(turn["text"], spans.get(turn["turn_id"], []))
        if rules["han_characters"] == "FORBIDDEN" and HAN.search(outside):
            found.append(f"turn {turn['turn_id']}: Chinese characters are not allowed in this call's speech (frozen script rule)")
        if rules["latin_letters"] == "K01_IDENTIFIER_SPAN_ONLY" and LATIN.search(outside):
            found.append(f"turn {turn['turn_id']}: Latin letters are only allowed inside the identifier event in this call's speech")
        if rules["arabic_digits"] == "FORBIDDEN" and DIGIT.search(outside):
            found.append(f"turn {turn['turn_id']}: Arabic digits are not allowed in this call's speech")
    return found


def numeral_extension_errors(ctx, turns, alignment):
    """A planned event whose text starts or ends with a number must not be extended by a numeral or value word glued to it.

    Marker adjacency to Han characters is allowed in a Chinese call, so a filler sentence could otherwise continue "七百" into
    "七百五十" or add a currency word. Only the character directly beside the span is looked at.
    """
    errors = []
    for span in alignment:
        text = turns[span["turn_id"]]["text"]
        planned = text[span["start"]:span["end"]]
        before = text[span["start"] - 1] if span["start"] > 0 else ""
        after = text[span["end"]] if span["end"] < len(text) else ""
        if planned and planned[-1] in _NUMBER_CHARS and after and (after in _NUMBER_CHARS or after in VALUE_MARKERS):
            errors.append(f"turn {span['turn_id']}: {after!r} directly after {span['event_id']} would extend or change the number it ends with")
        if planned and planned[0] in _NUMBER_CHARS and before and before in _NUMBER_CHARS:
            errors.append(f"turn {span['turn_id']}: {before!r} directly before {span['event_id']} would extend or change the number it starts with")
    return errors


def language_errors(ctx, turns, alignment, spans):
    if not ctx.multilingual:
        return []
    return script_errors(ctx, turns, spans) + numeral_extension_errors(ctx, turns, alignment)
