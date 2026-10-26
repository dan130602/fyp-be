"""Marker substitution: the model writes `[[EV_nnn]]`, Python inserts the exact G4 `planned_text`.

Evidence is therefore preserved by construction, every span offset is exact (zero-based Unicode code points, start
inclusive, end exclusive, relative to the rendered turn text) and the order of events inside a turn is the marker order.
"""

import re

from fyp_evaluation.validation.stage0 import HAN

MARKER = re.compile(r"\[\[(EV_[0-9]{3})\]\]")
# Multilingual calls: a marker may also sit directly beside Han characters or CJK punctuation (Chinese is not written with
# spaces); it must still never touch a Latin letter, a digit or another marker.
CJK_PUNCTUATION = "，。！？；：、…—「」『』（）《》〈〉“”‘’"
ENGLISH_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.%-]*")


def _boundary_ok(character, multilingual):
    return character.isspace() or (multilingual and (bool(HAN.match(character)) or character in CJK_PUNCTUATION))


def render(ctx, parsed):
    """(turns, alignment, errors): rendered turns with Python-assigned IDs and one span per event marker found.

    Each model turn's text is stripped of surrounding whitespace before rendering; nothing else is altered. Marker problems
    (stray brackets, a marker glued to a word, an unknown event) are reported as errors and that marker is not rendered.
    """
    turns, alignment, errors = [], [], []
    for turn_id, turn in enumerate(parsed["turns"]):
        text = turn["text"].strip()
        pieces, last, rendered_length = [], 0, 0
        for match in MARKER.finditer(text):
            start, end, event_id = match.start(), match.end(), match.group(1)
            multilingual = getattr(ctx, "multilingual", False)
            if (start > 0 and not _boundary_ok(text[start - 1], multilingual)) or (end < len(text) and not _boundary_ok(text[end], multilingual)):
                errors.append(f"turn {turn_id}: marker {match.group(0)} must be a separate token (whitespace on both sides"
                              + (", or Chinese characters or punctuation)" if multilingual else ")"))
            if event_id not in ctx.texts:
                errors.append(f"turn {turn_id}: unknown event marker {match.group(0)}")
                continue
            filler = text[last:start]
            pieces.append(filler)
            rendered_length += len(filler)
            alignment.append({"event_id": event_id, "turn_id": turn_id, "start": rendered_length,
                              "end": rendered_length + len(ctx.texts[event_id])})
            pieces.append(ctx.texts[event_id])
            rendered_length += len(ctx.texts[event_id])
            last = end
        pieces.append(text[last:])
        stray = MARKER.sub("", text)
        if "[[" in stray or "]]" in stray:
            errors.append(f"turn {turn_id}: stray '[[' or ']]' outside a well-formed event marker")
        turns.append({"turn_id": turn_id, "speaker": turn["speaker"], "text": "".join(pieces)})
    return turns, alignment, errors


def word_count(turns):
    """The length policy's word count: whitespace-separated tokens over the rendered turn texts."""
    return sum(len(turn["text"].split()) for turn in turns)


def length_count(turns, unit="words"):
    """The policy's length count of the rendered turns: words, Han characters, or Han characters plus English words."""
    if unit == "words":
        return word_count(turns)
    han = sum(len(HAN.findall(turn["text"])) for turn in turns)
    if unit == "han_chars":
        return han
    return han + sum(len(ENGLISH_WORD.findall(HAN.sub(" ", turn["text"]))) for turn in turns)


def outside_text(text, spans):
    """`text` with every event span blanked (same length), i.e. only the speech the model itself wrote."""
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = " "
    return "".join(chars)


def spans_by_turn(alignment):
    result = {}
    for span in alignment:
        result.setdefault(span["turn_id"], []).append((span["start"], span["end"]))
    return result


def occurrences_outside(text, needle, spans):
    """Start offsets of `needle` in `text` that are not fully inside any of the given (start, end) spans."""
    found, position = [], text.find(needle)
    while needle and position != -1:
        end = position + len(needle)
        if not any(start <= position and end <= stop for start, stop in spans):
            found.append(position)
        position = text.find(needle, position + 1)
    return found
