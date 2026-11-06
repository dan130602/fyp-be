"""Display units: the scoring alignment re-expressed as reference-anchored units a person can read.

Built from the SAME alignment that produces the scores (never a second alignment), so the UI can only highlight errors the
scoring counted. A unit groups the normalised tokens of one reference word, and merges words that share a Transcribe item
(reference "S six five ... A" against the single item "S6532539A") or a multi-word token ("nineteen ninety-eight").

Merging never hides an error: every unit keeps a token-level `parts` list [{ref, hyp, op, entity}] with insertions included.
A unit is "equal" only when every part is, and its error counts are the sums over its parts. Per call, summing the parts over all
units reproduces the exported S/D/I exactly (and the entity-counted parts reproduce entity S/D/I).
"""

from collections import defaultdict

LOW_CONFIDENCE = 0.5  # only a flag for the UI; it never changes a score


class _Sets:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_units(*, words, positions, speakers, ref, ref_origins, ref_ops, hyp, hyp_origins, item_of_word, hyp_ops, insertions,
                items, selected, token_groups, label_mapping, turn_times=None, ref_rules=None, hyp_rules=None):
    """The ordered display units of one call. `selected` is the set of entity-selected reference tokens, or None."""
    turn_of_word = [position[0] for position in positions]
    items_by_id = {item["item"]: item for item in items}
    sets = _Sets(len(words))

    for origin in ref_origins:  # a token that spans several reference words ties them together
        for word in origin[1:]:
            if turn_of_word[word] == turn_of_word[origin[0]]:
                sets.union(origin[0], word)

    hyp_items = [sorted({item_of_word[w] for w in origin}) for origin in hyp_origins]  # original AWS items behind each hyp token
    item_ref_words = defaultdict(set)
    for r, (_, k) in enumerate(ref_ops):
        if k is not None:
            for item in hyp_items[k]:
                item_ref_words[item].update(ref_origins[r])
    for words_of_item in item_ref_words.values():  # tokens of one Transcribe item that align to different reference words
        ordered = sorted(words_of_item)
        for word in ordered[1:]:
            if turn_of_word[word] == turn_of_word[ordered[0]]:
                sets.union(ordered[0], word)

    changed = True
    while changed:  # a merged unit covers every reference word between its first and last, so words are never shown out of order
        changed = False
        spans_of = defaultdict(list)
        for word in range(len(words)):
            spans_of[sets.find(word)].append(word)
        for members in spans_of.values():
            for word in range(members[0], members[-1] + 1):
                if turn_of_word[word] == turn_of_word[members[0]] and sets.find(word) != sets.find(members[0]):
                    sets.union(members[0], word)
                    changed = True
    unit_of_token = [sets.find(origin[0]) for origin in ref_origins]
    units = {}

    def unit(key, turn_id):
        return units.setdefault(key, {"parts": [], "sort": None, "turn_id": turn_id, "items": set(), "words": set(), "pure_insert": False})

    for r, (op, k) in enumerate(ref_ops):
        u = unit(unit_of_token[r], turn_of_word[ref_origins[r][0]])
        u["parts"].append({"key": r, "ref": ref[r], "hyp": hyp[k] if k is not None else None, "op": op, "r": r,
                           "entity": bool(selected is not None and r in selected and op != "equal")})
        u["words"].update(ref_origins[r])
        if k is not None:
            u["items"].update(hyp_items[k])

    for anchor, k in insertions:
        owner = next((sets.find(min(item_ref_words[i])) for i in hyp_items[k] if i in item_ref_words), None)
        before = unit_of_token[anchor - 1] if anchor > 0 else None
        if owner is None and before is not None and anchor < len(ref) and before == unit_of_token[anchor]:
            owner = before  # interior to a unit
        if owner is not None:
            u = unit(owner, units[owner]["turn_id"])
        else:  # between units: one insert unit per Transcribe item, placed after the unit holding the previous reference token
            turn_id = turn_of_word[ref_origins[anchor - 1][0]] if anchor > 0 else (turn_of_word[0] if turn_of_word else 0)
            u = unit(("insert", anchor, tuple(hyp_items[k])), turn_id)
            u["pure_insert"], u["sort"] = True, anchor - 0.5
        counted = bool(selected is not None and anchor - 1 in selected and anchor in selected)
        u["parts"].append({"key": anchor - 0.5 + k * 1e-6, "ref": None, "hyp": hyp[k], "op": "insert", "anchor": anchor, "entity": counted})
        u["items"].update(hyp_items[k])

    ordered = sorted(units.values(), key=lambda u: (u["sort"] if u["sort"] is not None else min(p["r"] for p in u["parts"] if "r" in p)))
    eligible_items = {i for i, item in items_by_id.items()
                      if (tokens := [k for k, group in enumerate(hyp_items) if i in group]) and all(hyp_ops[k] == "equal" for k in tokens)}
    out = []
    for index, u in enumerate(ordered):
        u["parts"].sort(key=lambda p: p["key"])
        parts = [{key: value for key, value in p.items() if key in ("ref", "hyp", "op", "entity")} for p in u["parts"]]
        counts = {"substitute": 0, "delete": 0, "insert": 0}
        entity_errors = 0
        for part in parts:
            if part["op"] in counts:
                counts[part["op"]] += 1
                entity_errors += part["entity"]
        errors = sum(counts.values())
        kinds = [name for name, n in counts.items() if n]
        ref_words = sorted(u["words"])
        turn_id = u["turn_id"]
        ref_role = speakers[ref_words[0]] if ref_words else None
        unit_items = []
        for i in sorted(u["items"]):
            item = items_by_id[i]
            unit_items.append({"item": i, "content": item["content"], "confidence": item["confidence"], "speaker": item["speaker"],
                               "start": item.get("start"), "end": item.get("end"),
                               "mapped_role": label_mapping.get(item["speaker"]), "eligible": i in eligible_items})
        confidences = [x["confidence"] for x in unit_items if x["confidence"] is not None]
        groups = sorted({g for p in u["parts"] if "r" in p for g in token_groups.get(p["r"], ())})
        starts = [x["start"] for x in unit_items if x["start"] is not None]
        ends = [x["end"] for x in unit_items if x["end"] is not None]
        out.append({
            "id": index, "turn_id": turn_id,
            "ref_text": " ".join(words[w] for w in ref_words), "hyp_text": " ".join(x["content"] for x in unit_items),
            "op": "equal" if not errors else kinds[0] if len(kinds) == 1 else "mixed",
            "errors": errors, "substitutions": counts["substitute"], "deletions": counts["delete"], "insertions": counts["insert"],
            "entity_errors": entity_errors, "entity_groups": groups, "parts": parts,
            "ref_role": ref_role,
            "speaker_mismatch": any(x["eligible"] and x["mapped_role"] is not None and ref_role is not None and x["mapped_role"] != ref_role
                                    for x in unit_items),
            "min_confidence": min(confidences) if confidences else None,
            "low_confidence": bool(confidences and min(confidences) < LOW_CONFIDENCE),
            "items": unit_items,
            "scoring_rules": sorted({rule for w in u["words"] for rule in (ref_rules or {}).get(w, ())}
                                    | {rule for i in u["items"] for rule in (hyp_rules or {}).get(i, ())}),
            "start": min(starts) if starts else None, "end": max(ends) if ends else None,
            "time_source": "aws" if starts else None, "approximate": False,
        })
    _fill_missing_times(out, turn_times or {})
    return out


def _fill_missing_times(units, turn_times):
    """Units with no AWS word time (deletions) borrow the neighbouring aligned words' times, else the reference turn's; marked approximate."""
    for index, u in enumerate(units):
        if u["start"] is not None:
            continue
        previous = next((units[j]["end"] for j in range(index - 1, -1, -1) if units[j]["time_source"] == "aws"), None)
        following = next((units[j]["start"] for j in range(index + 1, len(units)) if units[j]["time_source"] == "aws"), None)
        if previous is not None or following is not None:
            u["start"] = previous if previous is not None else following
            u["end"] = following if following is not None else previous
            u["time_source"] = "neighbours"
        elif u["turn_id"] in turn_times:
            u["start"], u["end"] = turn_times[u["turn_id"]]
            u["time_source"] = "reference_turn"
        u["approximate"] = u["start"] is not None


def error_totals(units):
    """S/D/I and entity-counted S/D/I summed over the parts of all units (must equal the exported scores)."""
    total = {"substitutions": 0, "deletions": 0, "insertions": 0, "entity_errors": 0}
    for u in units:
        total["substitutions"] += u["substitutions"]
        total["deletions"] += u["deletions"]
        total["insertions"] += u["insertions"]
        total["entity_errors"] += u["entity_errors"]
    return total
