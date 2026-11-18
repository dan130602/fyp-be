"""Conservative hypothesis-only mapping. No ground-truth imports or access."""
from collections import Counter
import math
import re
import unicodedata

TOKEN = re.compile(r"(?:[A-Za-z]*[$\u00a3\u20ac\u00a5\u20b9])?[+-]?\w+(?:['’.,]\w+)*%?|[^\w\s]")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")


def canonical(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def token_spans(text):
    for match in TOKEN.finditer(text):
        left = match.start()
        # Separate Han characters for timestamps, as the project scorer already
        # scores them per character. Keep complete numeric/currency tokens.
        for han in HAN.finditer(match.group()):
            position = match.start() + han.start()
            if left < position:
                yield left, position, text[left:position]
            yield position, position + 1, text[position:position + 1]
            left = position + 1
        if left < match.end():
            yield left, match.end(), text[left:match.end()]


def speaker_for(start, end, segments):
    overlap = Counter()
    for row in segments:
        overlap[row['speaker']] += max(0., min(end, row['end']) - max(start, row['start']))
    positive = {speaker: amount for speaker, amount in overlap.items() if amount > 1e-9}
    # Do not assign a word that straddles a predicted speaker change.
    if len(positive) != 1:
        return None, 'speaker_boundary_or_no_speech'
    label, amount = next(iter(positive.items()))
    if amount <= (end - start) / 2:
        return None, 'insufficient_speech_overlap'
    return label, None


def chunk_items(text, units, offset, duration, segments):
    spans = list(token_spans(text))
    items = []
    diagnostics = []
    for left, right, token in spans:
        pronunciation = bool(canonical(token)) or token in '$%+-\u00a3\u20ac\u00a5\u20b9'
        items.append({'type': 'pronunciation' if pronunciation else 'punctuation',
                      'alternatives': [{'content': token}]})
        diagnostics.append({'start_char': left, 'end_char': right})
    source = [(char, index) for index, original in enumerate(text) for char in canonical(original)]
    lexical = [(unit, canonical(unit['text'])) for unit in units if canonical(unit['text'])]
    same_text = ''.join(char for char, _ in source) == ''.join(word for _, word in lexical)
    owners = [[] for _ in text]
    validity = []
    cursor = 0
    previous_start = -1.
    if same_text:
        for unit_index, (unit, word) in enumerate(lexical):
            begin, end = float(unit['start_time']), float(unit['end_time'])
            valid = (math.isfinite(begin) and math.isfinite(end) and
                     0 <= begin < end <= duration + .05 and begin < duration and begin >= previous_start)
            validity.append(valid)
            if math.isfinite(begin):
                previous_start = begin
            for _, position in source[cursor:cursor + len(word)]:
                owners[position].append(unit_index)
            cursor += len(word)
    unit_tokens = {}
    for token_index, (left, right, _) in enumerate(spans):
        if items[token_index]['type'] != 'pronunciation':
            continue
        for owner in {owner for position in range(left,right) for owner in owners[position]}:
            unit_tokens.setdefault(owner,set()).add(token_index)
    for token_index, ((left,right,token), item, diagnostic) in enumerate(zip(spans,items,diagnostics)):
        if item['type'] != 'pronunciation':
            continue
        reason = None
        used = sorted({owner for position in range(left,right) for owner in owners[position]})
        if not same_text:
            reason = 'alignment_text_mismatch'
        elif not used:
            reason = 'no_lexical_alignment'
        elif any(not validity[owner] for owner in used):
            reason = 'invalid_alignment_time'
        elif any(len(unit_tokens[owner]) != 1 for owner in used):
            reason = 'alignment_unit_spans_multiple_items'
        if reason:
            diagnostic['reason'] = reason
            continue
        start = offset + min(float(lexical[owner][0]['start_time']) for owner in used)
        end = offset + min(duration, max(float(lexical[owner][0]['end_time']) for owner in used))
        item.update(start_time=str(round(start,4)), end_time=str(round(end,4)))
        label, reason = speaker_for(start,end,segments)
        if label is not None:
            item['speaker_label'] = label
        diagnostic['reason'] = reason
    return items, diagnostics


def document(transcript, items):
    return {'results': {'transcripts':[{'transcript':transcript}], 'items':items}}
