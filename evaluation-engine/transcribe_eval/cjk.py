"""Han-character tokenisation for Mandarin (and the Chinese parts of mixed) calls.

English is scored per word; Chinese has no spaces, so each Han character is one scoring unit (a character error rate for the Chinese parts,
a word error rate for the English parts of the same call). Text without Han characters is returned unchanged, so English and Singlish scores
are byte-identical to before. Offsets are exact so entity spans (character ranges in the turn text) keep working.
"""

import re

HAN_RANGES = "〇㐀-䶿一-鿿豈-﫿"  # includes 〇 (U+3007), the zero of 二〇三一
HAN = re.compile(f"[{HAN_RANGES}]")
_UNIT = re.compile(rf"[{HAN_RANGES}]|[^\s{HAN_RANGES}]+")
CJK_TOKENISATION = "han-character-v1"


def has_han(text):
    return HAN.search(text) is not None


def split_han(text, offset=0):
    """[(piece, start, end)]: every Han character on its own, other non-space runs kept whole, offsets relative to `offset`."""
    return [(m.group(), offset + m.start(), offset + m.end()) for m in _UNIT.finditer(text)]
