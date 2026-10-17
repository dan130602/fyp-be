"""The hypothesis side: Transcribe's `results.items` as words, each traced back to its original pronunciation item."""

import json
from pathlib import Path

PHRASE_BREAK_PUNCTUATION = set(".?!;:,")


def load_items(path):
    """Pronunciation items [{item, content, confidence, speaker}] in order, plus the raw transcript text for inspection."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
    items, breaks_after_item = [], set()
    for raw in data["items"]:
        alternative = raw["alternatives"][0]
        if raw["type"] == "punctuation":
            if items and alternative["content"] in PHRASE_BREAK_PUNCTUATION:
                breaks_after_item.add(len(items) - 1)
            continue
        confidence = alternative.get("confidence")
        items.append({"item": len(items), "content": alternative["content"], "speaker": raw.get("speaker_label"),
                      "confidence": float(confidence) if confidence not in (None, "") else None})
    text = " ".join(t["transcript"] for t in data.get("transcripts", []))
    return items, breaks_after_item, text


def raw_words(items, breaks_after_item):
    """Words of the pronunciation items, the item each came from, and phrase breaks after punctuated items."""
    words, item_of_word, breaks = [], [], set()
    for item in items:
        for part in item["content"].split():
            words.append(part)
            item_of_word.append(item["item"])
        if item["item"] in breaks_after_item and words:
            breaks.add(len(words) - 1)
    return words, item_of_word, breaks
