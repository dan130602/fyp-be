"""Quote grounding against ONE actual dialogue turn, with turn-local Unicode character spans.

The narrow, declared normalisation is G5's (`plan_verifications.grounding.normalize`: curly quotes, ellipsis, dashes and
whitespace runs only; case-sensitive; no punctuation stripping). The stored quote is the EXACT source substring, recovered
through the index map, never the model's spelling. A quote that occurs several times in the turn keeps EVERY matching
occurrence (no arbitrary first match); a quote whose normalised form matches DIFFERENT exact substrings is rejected as ambiguous.
"""

from fyp_evaluation.plan_verifications.grounding import normalize


def locate(quote, text):
    """(exact substring, [[start, end], ...], None) for a grounded quote, else (None, None, reason).

    Spans are zero-based code-point offsets into `text` (start inclusive, end exclusive), ordered, without duplicates.
    """
    if not isinstance(quote, str) or not quote.strip():
        return None, None, "quote is empty or whitespace-only"
    needle = normalize(quote)[0].strip()
    haystack, spans = normalize(text)
    found, start = set(), haystack.find(needle)
    while start != -1:
        found.add((spans[start][0], spans[start + len(needle) - 1][1]))
        start = haystack.find(needle, start + 1)
    if not found:
        return None, None, "quote is not a substring of the cited turn"
    exact = {text[a:b] for a, b in found}
    if len(exact) > 1:
        return None, None, "quote is ambiguous: it matches different source substrings after normalisation"
    return exact.pop(), [list(span) for span in sorted(found)], None
