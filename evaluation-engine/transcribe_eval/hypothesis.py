"""The hypothesis side: Transcribe's `results.items` as words, each traced back to its original pronunciation item."""

import json
import unicodedata
from pathlib import Path

from .cjk import has_han, split_han

PHRASE_BREAK_PUNCTUATION = set(".?!;:,")
COMMA = ","  # the one break a date may span (all others end a multi-word match)


def _seconds(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_items(path):
    """Pronunciation items [{item, content, confidence, speaker}] in order, the items followed by a phrase break, the subset of those
    followed by a comma, and the raw transcript text for inspection."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
    items, breaks_after_item, commas_after_item = [], set(), set()
    for raw in data["items"]:
        alternative = raw["alternatives"][0]
        if raw["type"] == "punctuation":
            mark = unicodedata.normalize("NFKC", alternative["content"]).replace("。", ".").replace("、", ",")  # Chinese full stop / list comma
            if items and mark in PHRASE_BREAK_PUNCTUATION:
                breaks_after_item.add(len(items) - 1)
                if mark == COMMA:
                    commas_after_item.add(len(items) - 1)
                else:
                    commas_after_item.discard(len(items) - 1)
            continue
        confidence = alternative.get("confidence")
        items.append({"item": len(items), "content": alternative["content"], "speaker": raw.get("speaker_label"),
                      "confidence": float(confidence) if confidence not in (None, "") else None,
                      "start": _seconds(raw.get("start_time")), "end": _seconds(raw.get("end_time"))})
    text = " ".join(t["transcript"] for t in data.get("transcripts", []))
    return items, breaks_after_item, commas_after_item, text


def raw_words(items, breaks_after_item, commas_after_item=()):
    """Words of the pronunciation items, the item each came from, the phrase breaks after punctuated items, which of those are commas,
    and the speaker boundaries (after the last word of an item whose speaker label differs from the next item's)."""
    words, item_of_word, breaks, commas, boundaries = [], [], set(), set(), set()
    for position, item in enumerate(items):
        for part in item["content"].split():
            for piece in ([p for p, _, _ in split_han(part)] if has_han(part) else [part]):  # Han characters are scored one by one
                words.append(piece)
                item_of_word.append(item["item"])
        if not words:
            continue
        if item["item"] in breaks_after_item:
            breaks.add(len(words) - 1)
            if item["item"] in commas_after_item:
                commas.add(len(words) - 1)
        following = items[position + 1]["speaker"] if position + 1 < len(items) else None
        if item["speaker"] and following and item["speaker"] != following:
            boundaries.add(len(words) - 1)
    return words, item_of_word, breaks, commas, boundaries
