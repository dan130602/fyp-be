"""Quote grounding: a model quote must be a substring of the cited source text, after a narrow, declared normalisation.

Normalisation (and nothing else): curly single/double quotes -> ' and "; the ellipsis character -> "..."; em and en
dashes -> "-"; every run of whitespace (including NBSP) -> one space; the quote's ends are trimmed. Matching is
case-sensitive, with no Unicode normalisation and no punctuation stripping. The stored quote is always the EXACT
source substring a match covers, recovered through an index map, never the model's spelling.
"""

_CHARACTER_MAP = {"‘": "'", "’": "'", "“": '"', "”": '"', "…": "...", "—": "-", "–": "-"}


def _is_space(character):
    return character.isspace() or character == " "


def normalize(text):
    """(normalised text, spans): spans[i] is the (start, end) source range that produced normalised character i."""
    out, spans = [], []
    index, length = 0, len(text)
    while index < length:
        character = text[index]
        if _is_space(character):
            end = index
            while end < length and _is_space(text[end]):
                end += 1
            out.append(" ")
            spans.append((index, end))
            index = end
            continue
        replacement = _CHARACTER_MAP.get(character, character)
        for piece in replacement:
            out.append(piece)
            spans.append((index, index + 1))
        index += 1
    return "".join(out), spans


def ground(quote, source):
    """(exact source substring, None) for a uniquely grounded quote, else (None, reason)."""
    if not isinstance(quote, str) or not quote.strip():
        return None, "quote is empty or whitespace-only (use null when no quote is cited)"
    needle = normalize(quote)[0].strip()
    haystack, spans = normalize(source)
    exact = set()
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle) - 1
        exact.add(source[spans[start][0]:spans[end][1]])
        start = haystack.find(needle, start + 1)
    if not exact:
        return None, "quote is not a substring of the cited source text"
    if len(exact) > 1:
        return None, "quote is ambiguous: it matches different source substrings after normalisation"
    return exact.pop(), None
