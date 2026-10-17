"""The reference side: canonical dialogue turns and, where a dataset annotates them, the spans entity WER scores.

Entity WER selects (project definition, see README):
  (a) every annotated value mention (check plan `value_mentions` placed in the turn through the dialogue's `event_alignment`),
  (b) the whole aligned span of every critical event (corrections, cancellations, readbacks, false starts) and of the events it links to,
  (c) occurrences of terms from a REVIEWED domain-term list.
Every span is checked against the actual reference turn text; a mismatch is an error, never silently skipped.
A dataset without these annotations (the legacy 5-call pilot) has no entity spans: `selected_spans` returns None.
"""

import json
import re
from pathlib import Path

CRITICAL_KINDS = ("VALUE_CORRECTION", "ORDER_INVALIDATION", "READBACK", "FALSE_START")
LINK_FIELDS = ("corrects_event_id", "invalidates_event_id", "refers_to_event_id")


class ReferenceDataError(ValueError):
    pass


def load_turns(dataset_root, call_id):
    """[{turn_id, speaker, text}] from the call's canonical ground truth."""
    path = Path(dataset_root) / "calls" / call_id / "ground_truth.json"
    turns = json.loads(path.read_text(encoding="utf-8"))["dialogue"]
    if [t["turn_id"] for t in turns] != list(range(len(turns))):
        raise ReferenceDataError(f"{call_id}: dialogue turn ids are not 0..n-1")
    return turns


def raw_words(turns):
    """Whitespace words of every turn, with (turn_id, start, end) positions, the turn speaker, and a phrase break after each turn."""
    words, positions, speakers, breaks = [], [], [], set()
    for turn in turns:
        for match in re.finditer(r"\S+", turn["text"]):
            words.append(match.group())
            positions.append((turn["turn_id"], match.start(), match.end()))
            speakers.append(turn["speaker"])
        if words:
            breaks.add(len(words) - 1)
    return words, positions, speakers, breaks


def load_terms(path):
    """The domain terms, but only from a list marked reviewed; an unreviewed or missing list yields no terms."""
    path = Path(path)
    if not path.is_file():
        return False, []
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("reviewed")), list(data.get("terms") or []) if data.get("reviewed") else []


def _annotation_files(dataset_root, call_id):
    plan = Path(dataset_root) / "authoring" / "check_plans" / f"{call_id}.json"
    dialogue = Path(dataset_root) / "authoring" / "dialogues" / f"{call_id}.json"
    if not (plan.is_file() and dialogue.is_file()):
        return None
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    dialogue_data = json.loads(dialogue.read_text(encoding="utf-8"))
    if "planned_events" not in plan_data or "event_alignment" not in dialogue_data:
        return None
    return plan_data, dialogue_data


def selected_spans(dataset_root, call_id, turns, terms=()):
    """The entity-WER spans [{turn_id, start, end, group, label, text}], or None when the dataset has no annotations."""
    files = _annotation_files(dataset_root, call_id)
    if files is None:
        return None
    plan, dialogue = files
    texts = {t["turn_id"]: t["text"] for t in turns}
    if [t["text"] for t in dialogue["turns"]] != [t["text"] for t in turns]:
        raise ReferenceDataError(f"{call_id}: the authoring dialogue differs from the ground-truth dialogue")
    events = {e["event_id"]: e for e in plan["planned_events"]}
    aligned = {a["event_id"]: a for a in dialogue["event_alignment"]}

    def event_span(event_id, group):
        a, event = aligned[event_id], events[event_id]
        text = texts[a["turn_id"]][a["start"]:a["end"]]
        if text != event["authoring"]["planned_text"]:
            raise ReferenceDataError(f"{call_id}/{event_id}: aligned text differs from the planned text")
        return {"turn_id": a["turn_id"], "start": a["start"], "end": a["end"], "group": group, "label": event_id, "text": text}

    spans = []
    for event_id, event in events.items():
        for mention in event["authoring"].get("value_mentions", []):
            a = aligned[event_id]
            start, end = a["start"] + mention["start"], a["start"] + mention["end"]
            text = texts[a["turn_id"]][start:end]
            if text != mention["surface"]:
                raise ReferenceDataError(f"{call_id}/{event_id}: mention {mention['surface']!r} not found at its offsets")
            group = f"value:{mention['question_id'] or mention['source']}"
            spans.append({"turn_id": a["turn_id"], "start": start, "end": end, "group": group, "label": event_id, "text": text})
    for event_id, event in events.items():
        if event["kind"] not in CRITICAL_KINDS:
            continue
        spans.append(event_span(event_id, f"event:{event['kind']}"))
        seen, queue = {event_id}, [event_id]
        while queue:  # linked events, followed transitively
            current = queue.pop(0)
            for field in LINK_FIELDS:
                linked = events[current].get(field)
                if linked and linked not in seen:
                    seen.add(linked)
                    queue.append(linked)
                    spans.append(event_span(linked, f"event:{event['kind']}:linked"))
    for term in terms:
        pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
        for turn in turns:
            for match in pattern.finditer(turn["text"]):
                spans.append({"turn_id": turn["turn_id"], "start": match.start(), "end": match.end(), "group": f"term:{term}",
                              "label": term, "text": match.group()})
    return spans
