"""Strict, versioned normalisation of reference and Transcribe text for WER.

Principle: change how something is WRITTEN, never what was SAID. The same rules run independently on the reference and on Transcribe's
output and never consult expected values. Anything ambiguous is left unchanged and reported in `notes`.

Rules (version NORMALIZER_VERSION):
- NFKC, casefold, curly apostrophes to straight. In-word apostrophes stay ("don't" is not expanded; reported as a contraction).
- Numbers are recognised BEFORE punctuation is stripped, so decimals, signs and leading zeros survive:
  "1.2" -> 1 point 2; "-5" -> minus 5; "00123" -> 0 0 1 2 3; thousands separators only in valid groups ("1,200,000").
- Spelled numbers become digits: cardinals ("fifty three"), multipliers ("one million", "1.2 million"), the "k" shorthand ("500k"),
  ordinals ("sixteenth", "16th" -> 16 ord), decimals ("one point two"), an explicit year rule ("nineteen ninety-eight" -> 1998),
  "double"/"triple" + digit, and "oh" -> 0 only between digits. Single-digit words in a row stay separate digits ("six five" -> 6 5).
- Letter-digit identifiers keep their letters ("S6532539A" -> s 6 5 3 2 5 3 9 a). "%" -> percent.
- Every number is written one digit per token. Documented consequences: a seven-digit identifier counts as seven WER tokens, and
  a cardinal and its digit-by-digit reading become the same tokens ("sixty five" and "six five" are both 6 5), so that
  particular misrecognition is not counted. Splitting is what lets a spelled identifier match Transcribe's digits.
- Hyphenated words are split ("ninety-eight", "long-term" -> long term). On the first real run Transcribe wrote "long term"
  and "7 year" (split) far more often than "cutoff" (joined), so splitting is kept; "cut-off" vs "cutoff" stays an error.
- A bare currency symbol attached to a number becomes the number followed by its spoken word ("€200" -> 2 0 0 euros,
  "$40,000" -> 4 0 0 0 0 dollars): Transcribe writes the symbol for a spoken "200 euros". Only symbols with one possible word are
  mapped (€ $ £ ₹); "¥" and letter-prefixed forms ("S$50") keep the symbol as its own token. All are still reported.
- Left unchanged and reported: currency symbols, digits in an unrecognised shape, a multiplier with no number, contractions.

Changes: 1.1.0 separates currency symbols from numbers (seen in the first real run: "$40,000" for spoken "forty thousand").
         1.2.0 writes a bare € $ £ ₹ on a number as the number plus its word (advisory CALL_001: "€200" for spoken "200 euros").
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

NORMALIZER_VERSION = "1.2.0"

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
            "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
            "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
MULTIPLIERS = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
REPEATS = {"double": 2, "triple": 3}
ORD = "ord"
CURRENCY_SYMBOLS = set("$€£¥₹")
# Symbols with exactly one spoken word. "¥" (yen or yuan) and letter-prefixed forms such as "S$" are ambiguous: left as tokens.
CURRENCY_WORDS = {"€": "euros", "$": "dollars", "£": "pounds", "₹": "rupees"}
TRAILING_BREAK = set(".,!?;:")
NUMBER = re.compile(r"(?P<sign>[-−])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?(?P<suffix>%|k|st|nd|rd|th)?")


class Piece:
    """An intermediate unit: a word, or a number kept as its digit STRING (so leading zeros survive)."""

    __slots__ = ("kind", "text", "origin")

    def __init__(self, kind, text, origin):
        self.kind, self.text, self.origin = kind, text, frozenset(origin)

    def __repr__(self):
        return f"{self.kind}:{self.text}"


def _number_string(value):
    value = Decimal(value)
    return str(int(value)) if value == value.to_integral_value() else format(value.normalize(), "f")


def _pieces(raw, index, notes):
    """One raw word -> pieces, and whether a phrase break follows it."""
    word = unicodedata.normalize("NFKC", raw).casefold().replace("’", "'").replace("‘", "'")
    breaks = bool(word) and word[-1] in TRAILING_BREAK
    core = word.strip(".,!?;:\"()[]{}…'")
    if not core:
        return [], True
    if any(symbol in core for symbol in CURRENCY_SYMBOLS):
        notes.append(("currency_symbol", raw))
        attached = re.fullmatch(r"([a-z]*[$€£¥₹])(.+)", core)
        if attached and NUMBER.fullmatch(attached[2]):
            rest, _ = _pieces(attached[2], index, notes)
            if attached[1] in CURRENCY_WORDS:
                return [*rest, Piece("word", CURRENCY_WORDS[attached[1]], {index})], breaks
            return [Piece("word", attached[1], {index}), *rest], breaks
        return [Piece("word", core, {index})], breaks
    match = NUMBER.fullmatch(core)
    if match:
        out = []
        if match["sign"]:
            out.append(Piece("word", "minus", {index}))
        number = match["int"].replace(",", "") + (match["frac"] or "")
        suffix = match["suffix"]
        if suffix == "k":
            number = _number_string(Decimal(number) * 1000)
        out.append(Piece("num", number, {index}))
        if suffix in ("st", "nd", "rd", "th"):
            out.append(Piece("ord", ORD, {index}))
        elif suffix == "%":
            out.append(Piece("word", "percent", {index}))
        return out, breaks
    if re.fullmatch(r"[a-z0-9]+", core) and re.search(r"\d", core) and re.search(r"[a-z]", core):
        return [Piece("num" if part.isdigit() else "word", part, {index}) for part in re.findall(r"\d+|[a-z]+", core)], breaks
    if "-" in core and re.fullmatch(r"[a-z]+(?:-[a-z]+)+", core):
        return [Piece("word", part, {index}) for part in core.split("-")], breaks
    if re.search(r"\d", core):
        notes.append(("unrecognised_number", raw))
        return [Piece("word", core, {index})], breaks
    cleaned = re.sub(r"[^a-z0-9']", "", core).strip("'")
    if not cleaned:
        return [], breaks
    if "'" in cleaned:
        notes.append(("contraction", cleaned))
    return [Piece("word", cleaned, {index})], breaks


def _two_digit(pieces, i):
    """A spelled 10-99 starting at i: (value, next index) or None."""
    if i < len(pieces) and pieces[i].kind == "word":
        text = pieces[i].text
        if text in TEENS:
            return TEENS[text], i + 1
        if text in TENS:
            if i + 1 < len(pieces) and pieces[i + 1].kind == "word" and pieces[i + 1].text in UNITS and UNITS[pieces[i + 1].text] > 0:
                return TENS[text] + UNITS[pieces[i + 1].text], i + 2
            return TENS[text], i + 1
    return None


def _year(pieces, i):
    """Explicit year rule: '<10-99> <10-99 | oh unit | hundred>', e.g. nineteen ninety-eight -> 1998."""
    first = _two_digit(pieces, i)
    if not first:
        return None
    high, j = first
    if j >= len(pieces) or pieces[j].kind != "word":
        return None
    if pieces[j].text == "hundred":
        if j + 1 < len(pieces) and pieces[j + 1].kind == "word" and pieces[j + 1].text in (*MULTIPLIERS, "and"):
            return None  # 'nineteen hundred thousand' / 'hundred and five' are cardinals
        return str(high * 100), j + 1
    if pieces[j].text == "oh" and j + 1 < len(pieces) and pieces[j + 1].kind == "word" and pieces[j + 1].text in UNITS:
        return str(high * 100 + UNITS[pieces[j + 1].text]), j + 2
    second = _two_digit(pieces, j)
    if not second:
        return None
    low, k = second
    if k < len(pieces) and pieces[k].kind == "word" and pieces[k].text in (*MULTIPLIERS, "hundred"):
        return None
    return str(high * 100 + low), k


def _cardinal(pieces, i):
    """A spelled (or digit + multiplier) number starting at i: (number string, is ordinal, next index) or None.
    Adjacent small numbers that cannot combine end the phrase, so 'six five' is two numbers, never 65 or 11."""
    total, current, last, j, ordinal = Decimal(0), None, None, i, False
    while j < len(pieces):
        piece = pieces[j]
        text = piece.text
        if piece.kind == "num" and current is None and last is None:
            nxt = pieces[j + 1].text if j + 1 < len(pieces) and pieces[j + 1].kind == "word" else None
            if nxt not in MULTIPLIERS and nxt != "hundred":
                break
            try:
                current, last, j = Decimal(text), "num", j + 1
            except InvalidOperation:
                break
            continue
        if piece.kind != "word":
            break
        small = UNITS.get(text, TEENS.get(text, TENS.get(text)))
        if small is not None:
            if current is None:
                current, last = Decimal(small), ("tens" if text in TENS else "small")
            elif last == "tens" and text in UNITS and small > 0:
                current, last = current + small, "small"
            elif last == "hundred" and current % 100 == 0:
                current, last = current + small, ("tens" if text in TENS else "small")
            elif last == "mult":
                current, last = Decimal(small), ("tens" if text in TENS else "small")
            else:
                break
            j += 1
        elif text == "hundred" and current is not None and current < 100 and last not in ("hundred",):
            current, last, j = current * 100, "hundred", j + 1
        elif text in MULTIPLIERS and current is not None:
            total, current, last, j = total + current * MULTIPLIERS[text], None, "mult", j + 1
        elif text == "and" and last in ("hundred", "mult") and j + 1 < len(pieces) and pieces[j + 1].kind == "word" and (
                pieces[j + 1].text in UNITS or pieces[j + 1].text in TEENS or pieces[j + 1].text in TENS):
            j += 1
        elif text in ORDINALS:
            value = ORDINALS[text]
            if current is None:
                current = Decimal(value)
            elif (last == "tens" and value < 10) or (last == "hundred" and current % 100 == 0) or last == "mult":
                current = current + value if last != "mult" else Decimal(value)
            else:
                break
            ordinal, j = True, j + 1
            break
        else:
            break
    if j == i or (current is None and last != "mult"):
        return None
    if last == "mult" and current is None and pieces[j - 1].kind == "word" and pieces[j - 1].text in MULTIPLIERS and j - 1 == i:
        return None
    value = total + (current or 0)
    if not ordinal and j < len(pieces) and pieces[j].kind == "word" and pieces[j].text == "point" and j + 1 < len(pieces) and \
            pieces[j + 1].kind == "word" and pieces[j + 1].text in UNITS and value == value.to_integral_value():
        k = j + 1
        digits = []
        while k < len(pieces) and pieces[k].kind == "word" and pieces[k].text in UNITS:
            digits.append(str(UNITS[pieces[k].text]))
            k += 1
        decimal = f"{int(value)}.{''.join(digits)}"
        if k < len(pieces) and pieces[k].kind == "word" and pieces[k].text in MULTIPLIERS:  # 'one point two million'
            return _number_string(Decimal(decimal) * MULTIPLIERS[pieces[k].text]), False, k + 1
        return decimal, False, k
    return _number_string(value), ordinal, j


def _combine(pieces, notes):
    """Phrase-level rules over one break-free run of pieces."""
    out, i = [], 0
    while i < len(pieces):
        piece = pieces[i]
        if piece.kind == "word" and piece.text in REPEATS and i + 1 < len(pieces):
            nxt = pieces[i + 1]
            digit = str(UNITS[nxt.text]) if nxt.kind == "word" and nxt.text in UNITS else nxt.text if nxt.kind == "num" and len(nxt.text) == 1 else None
            if digit is not None:
                out.append(Piece("num", digit * REPEATS[piece.text], piece.origin | nxt.origin))
                i += 2
                continue
        if piece.kind == "word" and piece.text == "oh" and out and out[-1].kind == "num" and out[-1].text.isdigit() and i + 1 < len(pieces):
            nxt = pieces[i + 1]
            if (nxt.kind == "word" and nxt.text in UNITS) or (nxt.kind == "num" and nxt.text.isdigit()) or (nxt.kind == "word" and nxt.text in REPEATS):
                out.append(Piece("num", "0", piece.origin))
                i += 1
                continue
        year = _year(pieces, i)
        if year:
            text, j = year
            out.append(Piece("num", text, frozenset().union(*(p.origin for p in pieces[i:j]))))
            i = j
            continue
        number = _cardinal(pieces, i)
        if number:
            text, ordinal, j = number
            origin = frozenset().union(*(p.origin for p in pieces[i:j]))
            out.append(Piece("num", text, origin))
            if ordinal:
                out.append(Piece("ord", ORD, origin))
            i = j
            continue
        if piece.kind == "word" and (piece.text in MULTIPLIERS or piece.text == "hundred"):
            notes.append(("multiplier_without_number", piece.text))
        out.append(piece)
        i += 1
    return out


def normalize_words(raw_words, breaks_after=()):
    """Raw words -> (tokens, origins, notes). `origins[k]` is the sorted tuple of raw-word indices token k came from.
    `breaks_after` adds phrase breaks after those raw indices (e.g. Transcribe punctuation items, turn ends)."""
    notes, runs, run = [], [], []
    extra = set(breaks_after)
    for index, raw in enumerate(raw_words):
        pieces, breaks = _pieces(raw, index, notes)
        run.extend(pieces)
        if breaks or index in extra:
            runs.append(run)
            run = []
    runs.append(run)
    tokens, origins = [], []
    for piece in (p for run in runs for p in _combine(run, notes)):
        origin = tuple(sorted(piece.origin))
        if piece.kind == "num":
            for character in piece.text:
                tokens.append("point" if character == "." else character)
                origins.append(origin)
        else:
            tokens.append(piece.text)
            origins.append(origin)
    return tokens, origins, notes


def normalize_text(text):
    """Convenience for tests and inspection: one string -> tokens."""
    return normalize_words(text.split())[0]
