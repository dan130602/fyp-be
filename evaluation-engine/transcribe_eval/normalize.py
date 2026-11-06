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
- Left unchanged and reported: currency symbols, digits in an unrecognised shape, a multiplier with no number, contractions that
  have no single meaning ('s, 'd).

Scoring-only equivalences (1.3.0). They exist so that the same thing written two ways is not counted as a recognition error. They change
only the tokens used for alignment; raw text is never modified, and nothing here is sent anywhere else. Each rule can be switched off by
name (`rules=`), which is how the score changes are attributed to rules. Every rule is applied independently to the reference and to
Transcribe's output and never looks at the other side.
- dates: a VALID date written day-first or month-first ("the fourth of August 1991", "4 August 1991", "August 4th, 1991") becomes one
  token order, month day [year]; "the", "of" and the ordinal marker are absorbed only inside a matched date. The day must exist in
  that month. "may" and "march" are also ordinary words, so they only count as a month with an ordinal, an "of" or a year.
- spelling: British -> American (OpenAI Whisper's pinned english.json as data, minus reviewed exclusions) and reviewed project
  variants (spelling_variants.json: "ok" -> "okay", "cutoff" -> "cut off").
- contractions: the unambiguous ones only ('ll 're 've 'm, auxiliary + n't, won't, can't); negation is always kept ("won't" is
  "will not", never "will"). 's and 'd stay as they are.
- currency: "$50,000 in SGD" has a redundant symbol, because the explicit currency after it says the same thing. When a currency
  symbol on a number is directly followed (optionally after "in") by an explicit mention that agrees with it, the symbol adds no token.
  Closed table (CURRENCY_TABLE_VERSION): "$" agrees with SGD, USD, dollar(s), Singapore/US dollars; "€" agrees with EUR, euro(s). A
  disagreement ("$30,000 in EUR") keeps the 1.2.0 behaviour (the symbol's word stays) and is reported as `currency_conflict`. A bare
  symbol is never dropped and a bare number never gains a currency.
Multi-word rules match only inside one reference turn or one speaker's words and never across a sentence end; a comma may sit inside a
date ("August 4th, 1991").

Changes: 1.1.0 separates currency symbols from numbers (seen in the first real run: "$40,000" for spoken "forty thousand").
         1.2.0 writes a bare € $ £ ₹ on a number as the number plus its word (advisory CALL_001: "€200" for spoken "200 euros").
         1.3.0 adds the dates, spelling, contractions and currency equivalences above (all can be switched off to reproduce 1.2.0).
         1.4.0 scores Mandarin: every Han character is one token (cjk.py), English stays per word; Chinese punctuation (，。、) ends a phrase.
         English and Singlish scores are identical to 1.3.0.
"""

import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from .cjk import CJK_TOKENISATION, HAN

NORMALIZER_VERSION = "1.4.0"  # 1.4.0: Han characters are scoring units (cjk.py); 1.3.0 rules and all English/Singlish scores unchanged
RULES = ("dates", "spelling", "contractions", "variants", "currency", "zh_numbers")
ALL_RULES = frozenset(RULES)
CURRENCY_TABLE_VERSION = "1"

DATA = Path(__file__).parent / "data"
SPELLING_SOURCE = {"project": "openai/whisper", "tag": "v20240930", "path": "whisper/normalizers/english.json", "licence": "MIT",
                   "sha256": "6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465"}


def _load_spelling():
    raw = (DATA / "english_spelling.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPELLING_SOURCE["sha256"]:
        raise RuntimeError(f"data/english_spelling.json is not the pinned Whisper {SPELLING_SOURCE['tag']} file (sha256 {digest}); "
                           "update SPELLING_SOURCE deliberately and bump NORMALIZER_VERSION")
    config_bytes = (DATA / "spelling_variants.json").read_bytes()
    config = json.loads(config_bytes)
    excluded = set(config.get("british_american_exclusions", {}))
    pairs = {k: v for k, v in json.loads(raw).items()
             if k not in excluded and re.fullmatch(r"[a-z]+", k) and re.fullmatch(r"[a-z]+", v) and k != v}
    reviewed = bool(config.get("reviewed"))
    variants = {k: v.split() for k, v in config.get("variants", {}).items()} if reviewed else {}
    info = {"spelling": {**SPELLING_SOURCE, "entries": len(pairs), "excluded": len(excluded)},
            "variants": {"sha256": hashlib.sha256(config_bytes).hexdigest(), "reviewed": reviewed, "entries": len(variants)},
            "currency_table_version": CURRENCY_TABLE_VERSION}
    return pairs, variants, info


BRITISH_TO_AMERICAN, PROJECT_VARIANTS, _RULES_INFO = _load_spelling()


def rules_info():
    """The exact scoring-rule data behind this version, recorded in every score summary so old and new scores stay distinguishable."""
    return {"normalizer_version": NORMALIZER_VERSION, "rules": list(RULES), "cjk_tokenisation": CJK_TOKENISATION, **_RULES_INFO}

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
TRAILING_BREAK = set(".,!?;:。、")
NUMBER = re.compile(r"(?P<sign>[-−])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?(?P<suffix>%|k|st|nd|rd|th)?")

MONTH_DAYS = {"january": 31, "february": 29, "march": 31, "april": 30, "may": 31, "june": 30, "july": 31, "august": 31,
              "september": 30, "october": 31, "november": 30, "december": 31}
AMBIGUOUS_MONTHS = {"may", "march"}  # also ordinary words: need an ordinal, an "of" or a year to count as a month
YEAR_RANGE = (1900, 2100)
# The closed currency table (CURRENCY_TABLE_VERSION): which explicit mentions agree with which symbol. Nothing else is recognised.
SYMBOL_OF_WORD = {"dollars": "$", "euros": "€"}
MENTION_CODES = {"sgd": "$", "usd": "$", "eur": "€"}
MENTION_NAMES = {"dollar": "$", "dollars": "$", "euro": "€", "euros": "€"}
MENTION_QUALIFIERS = {"singapore", "us"}  # "Singapore dollars", "US dollars"
CONTRACTION_SUFFIXES = (("'ll", "will"), ("'re", "are"), ("'ve", "have"))
CONTRACTION_SPECIAL = {"won't": ["will", "not"], "can't": ["can", "not"]}
NEGATED_AUXILIARIES = {"do", "does", "did", "is", "are", "was", "were", "has", "have", "had", "would", "could", "should", "must", "need", "might"}


class Piece:
    """An intermediate unit: a word, or a number kept as its digit STRING (so leading zeros survive)."""

    __slots__ = ("kind", "text", "origin", "raw")

    def __init__(self, kind, text, origin, raw=None):
        self.kind, self.text, self.origin, self.raw = kind, text, frozenset(origin), raw

    def __repr__(self):
        return f"{self.kind}:{self.text}"


def _expand_contraction(word):
    """The words of an unambiguous contraction, or None ('s, 'd and anything unrecognised stay as written)."""
    if word in CONTRACTION_SPECIAL:
        return list(CONTRACTION_SPECIAL[word])
    if word.endswith("n't") and word[:-3] in NEGATED_AUXILIARIES:
        return [word[:-3], "not"]
    if word.endswith("'m") and word[:-2] == "i":
        return ["i", "am"]
    for suffix, expansion in CONTRACTION_SUFFIXES:
        if word.endswith(suffix) and word[:-len(suffix)].isalpha():
            return [word[:-len(suffix)], expansion]
    return None


def _spelled(word, index, applied, rules):
    """One plain word -> its pieces after the spelling rules (British -> American, then reviewed project variants)."""
    words = [word]
    if "spelling" in rules and word in BRITISH_TO_AMERICAN:
        words = [BRITISH_TO_AMERICAN[word]]
        applied.append(("spelling", f"{word} -> {words[0]}", frozenset({index})))
    if "variants" in rules and len(words) == 1 and words[0] in PROJECT_VARIANTS:
        applied.append(("variants", f"{words[0]} -> {' '.join(PROJECT_VARIANTS[words[0]])}", frozenset({index})))
        words = list(PROJECT_VARIANTS[words[0]])
    return [Piece("word", w, {index}) for w in words]


def _number_string(value):
    value = Decimal(value)
    return str(int(value)) if value == value.to_integral_value() else format(value.normalize(), "f")


def _pieces(raw, index, notes, rules=ALL_RULES, applied=None):
    """One raw word -> pieces, and whether a phrase break follows it. Scoring-only rules that fire are appended to `applied`."""
    applied = [] if applied is None else applied
    word = unicodedata.normalize("NFKC", raw).casefold().replace("’", "'").replace("‘", "'")
    breaks = bool(word) and word[-1] in TRAILING_BREAK
    core = word.strip(".,!?;:\"()[]{}…'")
    if not core:
        return [], True
    if len(core) == 1 and HAN.fullmatch(core):  # a single Han character is one token, unchanged
        return [Piece("word", core, {index})], breaks
    if any(symbol in core for symbol in CURRENCY_SYMBOLS):
        attached = re.fullmatch(r"([a-z]*[$€£¥₹])(.+)", core)
        if attached and NUMBER.fullmatch(attached[2]):
            rest, _ = _pieces(attached[2], index, notes, rules, applied)
            if attached[1] in CURRENCY_WORDS:  # kept as its own kind until the currency pass decides whether it is redundant
                return [*rest, Piece("cursym", CURRENCY_WORDS[attached[1]], {index}, raw=raw)], breaks
            notes.append(("currency_symbol", raw))
            return [Piece("word", attached[1], {index}), *rest], breaks
        notes.append(("currency_symbol", raw))
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
        return [piece for part in core.split("-") for piece in _spelled(part, index, applied, rules)], breaks
    if re.search(r"\d", core):
        notes.append(("unrecognised_number", raw))
        return [Piece("word", core, {index})], breaks
    cleaned = re.sub(r"[^a-z0-9']", "", core).strip("'")
    if not cleaned:
        return [], breaks
    if "'" in cleaned:
        expanded = _expand_contraction(cleaned) if "contractions" in rules else None
        if expanded:
            applied.append(("contractions", f"{cleaned} -> {' '.join(expanded)}", frozenset({index})))
            return [Piece("word", word, {index}) for word in expanded], breaks
        notes.append(("contraction", cleaned))
        return [Piece("word", cleaned, {index})], breaks
    return _spelled(cleaned, index, applied, rules), breaks


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


def _union(pieces):
    return frozenset().union(*(piece.origin for piece in pieces))


def _day_valid(day, month, year):
    if not (day.text.isdigit() and day.text == str(int(day.text)) and 1 <= int(day.text) <= MONTH_DAYS[month]):
        return False
    if month == "february" and int(day.text) == 29 and year is not None:
        y = int(year.text)
        return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    return True


def _date_at(seq, i, connected):
    """A valid date starting at seq[i]: (end index, month, day, year or None) or None. Month-first and day-first, optional article/ordinal/'of'."""
    n = len(seq)

    def at(k, kind, texts=None):
        if k >= n or seq[k].kind != kind or (texts is not None and seq[k].text not in texts):
            return False
        return k == i or connected(seq[k - 1], seq[k])

    def year_at(k):
        return at(k, "num") and re.fullmatch(r"\d{4}", seq[k].text) and YEAR_RANGE[0] <= int(seq[k].text) <= YEAR_RANGE[1]

    k = i
    if at(k, "word", {"the"}):  # day-first: [the] D[th] [of] Month [YYYY]
        k += 1
    if at(k, "num"):
        day, k = seq[k], k + 1
        has_ord = at(k, "ord")
        k += has_ord
        has_of = at(k, "word", {"of"})
        k += has_of
        if at(k, "word", MONTH_DAYS):
            month, k = seq[k].text, k + 1
            year = seq[k] if year_at(k) else None
            k += year is not None
            if _day_valid(day, month, year) and (month not in AMBIGUOUS_MONTHS or has_ord or has_of or year):
                return k, month, day.text, year.text if year else None
    k = i
    if at(k, "word", MONTH_DAYS):  # month-first: Month [the] D[th][,] [YYYY]
        month, k = seq[k].text, k + 1
        k += at(k, "word", {"the"})
        if at(k, "num"):
            day, k = seq[k], k + 1
            has_ord = at(k, "ord")
            k += has_ord
            year = seq[k] if year_at(k) else None
            k += year is not None
            if _day_valid(day, month, year) and (month not in AMBIGUOUS_MONTHS or has_ord or year):
                return k, month, day.text, year.text if year else None
    return None


def _apply_dates(seq, connected, raw_words, applied):
    out, i = [], 0
    while i < len(seq):
        match = _date_at(seq, i, connected)
        if match:
            end, month, day, year = match
            origin = _union(seq[i:end])  # one origin for the whole date, like a multi-word number, so it reads as one unit
            out.extend(Piece(kind, text, origin) for kind, text in [("word", month), ("num", day), *([("num", year)] if year else [])])
            applied.append(("dates", " ".join(raw_words[w] for w in sorted(origin)), origin))
            i = end
        else:
            out.append(seq[i])
            i += 1
    return out


def _mention_at(seq, k, connected):
    """An explicit currency mention at seq[k] (optionally after 'in'): (end, symbol it agrees with) or None. Closed table only."""
    n = len(seq)

    def word(j, texts):
        return j < n and seq[j].kind == "word" and seq[j].text in texts and connected(seq[j - 1], seq[j])

    if word(k, {"in"}):
        k += 1
    if word(k, MENTION_CODES):
        return k + 1, MENTION_CODES[seq[k].text]
    if word(k, MENTION_QUALIFIERS) and word(k + 1, {"dollar", "dollars"}):
        return k + 2, "$"
    if word(k, MENTION_NAMES):
        return k + 1, MENTION_NAMES[seq[k].text]
    return None


ZH_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
ZH_SMALL = {"十": 10, "百": 100, "千": 1000}
ZH_BIG = {"万": 10_000, "亿": 100_000_000}


def _zh_kind(piece):
    if piece.kind == "num" and re.fullmatch(r"\d+(?:\.\d+)?", piece.text):
        return "arabic"
    if piece.kind == "word" and len(piece.text) == 1:
        text = piece.text
        return ("digit" if text in ZH_DIGIT else "small" if text in ZH_SMALL else "big" if text in ZH_BIG else "point" if text == "点" else None)
    return None


def _zh_integer(run, allow_single):
    """The digit string of a Chinese / Arabic integer written with `run` (pieces), or None when it is not clearly a number."""
    kinds = [_zh_kind(p) for p in run]
    if not run or None in kinds or "point" in kinds:
        return None
    if all(k == "digit" for k in kinds):  # positional digits: 一二九六七一七, 二〇三一
        return "".join(str(ZH_DIGIT[p.text]) for p in run)  # a lone 五 equals 5 on both sides, so it costs nothing and fixes 五欧元 vs 5欧元
    if kinds == ["arabic"]:
        return run[0].text if "." not in run[0].text else None
    if "small" not in kinds and "big" not in kinds:
        return None
    yi, wan, section, current, seen_unit, scale = 0, 0, 0, None, False, 0  # 亿 level, 万 level, below 万, a pending digit, 一万五-style scale
    for piece, kind in zip(run, kinds):
        if kind == "arabic":
            if "." in piece.text or current is not None:
                return None
            current = int(piece.text)
        elif kind == "digit":
            if ZH_DIGIT[piece.text] == 0:  # 零 inside a quantity (一千零五十, 一万零五) only marks a gap
                scale = 0
                continue
            if current is not None:
                return None
            current = ZH_DIGIT[piece.text]
            scale = scale if section == 0 and wan + yi > 0 else 0  # a digit right after 万 / 亿 with no unit (一万五 = 15000) counts a tenth of it
        elif kind == "small":
            if current is None and not (piece.text == "十" and section == 0 and not seen_unit):
                return None
            section += (current if current is not None else 1) * ZH_SMALL[piece.text]
            current, seen_unit, scale = None, True, 0
        elif piece.text == "万":
            if current is None and section == 0:
                return None
            wan, section, current, seen_unit, scale = wan + (section + (current or 0)) * ZH_BIG["万"], 0, None, True, ZH_BIG["万"] // 10
        else:  # 亿
            if current is None and section == 0 and wan == 0:
                return None
            yi, wan, section, current, seen_unit, scale = yi + (wan + section + (current or 0)) * ZH_BIG["亿"], 0, 0, None, True, ZH_BIG["亿"] // 10
    return str(yi + wan + section + (current or 0) * (scale if section == 0 and kinds[-1] in ("digit", "arabic") and scale else 1))


def _zh_number(run):
    """The number a run of numeral pieces writes, as a digit string with an optional '.', or None."""
    kinds = [_zh_kind(p) for p in run]
    if not any(k != "arabic" for k in kinds):
        return None  # plain Arabic digits are the other side's spelling already
    if "point" in kinds:
        at = kinds.index("point")
        whole, fraction = run[:at], run[at + 1:]
        digits = _zh_integer(whole, True)
        if digits is None or not fraction:
            return None
        tail = []
        for piece in fraction:
            kind = _zh_kind(piece)
            if kind == "digit":
                tail.append(str(ZH_DIGIT[piece.text]))
            elif kind == "arabic" and "." not in piece.text:
                tail.append(piece.text)
            else:
                return None
        return f"{digits}.{''.join(tail)}"
    return _zh_integer(run, False)


def _apply_zh_numbers(seq, connected, raw_words, applied):
    """Chinese numerals and "7万5千"-style mixes become the digit string the other spelling gives; 百分之N becomes N percent."""
    def run_from(start):
        run = []
        for k in range(start, len(seq)):
            if _zh_kind(seq[k]) is None or (run and not connected(seq[k - 1], seq[k])):
                break
            run.append(seq[k])
        return run

    out, i = [], 0
    while i < len(seq):
        percent = (i + 3 < len(seq) and [p.text for p in seq[i:i + 3]] == ["百", "分", "之"] and all(p.kind == "word" for p in seq[i:i + 3])
                   and connected(seq[i], seq[i + 1]) and connected(seq[i + 1], seq[i + 2]) and connected(seq[i + 2], seq[i + 3]))
        start = i + 3 if percent else i
        run = run_from(start) if (percent or _zh_kind(seq[i]) is not None) else []
        if run:
            value = _zh_number(run) if (percent or any(_zh_kind(p) != "arabic" for p in run)) else None
            if percent and value is None and all(_zh_kind(p) == "arabic" for p in run):
                value = run[0].text if len(run) == 1 else None  # 百分之5: the number is already digits
            if value is not None:
                span = seq[i:start + len(run)]
                origin = _union(span)
                applied.append(("zh_numbers", " ".join(raw_words[w] for w in sorted(origin)), origin))
                out.append(Piece("num", value, origin))
                if percent:
                    out.append(Piece("word", "percent", origin))
                i = start + len(run)
                continue
        out.append(seq[i])
        i += 1
    return out


def _apply_currency(seq, connected, raw_words, notes, applied, rules):
    out = []
    for i, piece in enumerate(seq):
        if piece.kind != "cursym":
            out.append(piece)
            continue
        mention = _mention_at(seq, i + 1, connected) if "currency" in rules else None
        if mention and mention[1] == SYMBOL_OF_WORD[piece.text]:  # the explicit currency already says it: the symbol is formatting
            origin = _union(seq[i:mention[0]])
            applied.append(("currency", " ".join(raw_words[w] for w in sorted(origin)), origin))
            continue
        notes.append(("currency_symbol", piece.raw))
        if mention:
            notes.append(("currency_conflict", " ".join(raw_words[w] for w in sorted(_union(seq[i:mention[0]])))))
        out.append(Piece("word", piece.text, piece.origin))
    return out


def normalize_words_traced(raw_words, breaks_after=(), *, boundaries_after=(), commas_after=(), rules=ALL_RULES):
    """Raw words -> (tokens, origins, notes, applied). `origins[k]` is the sorted tuple of raw-word indices token k came from.

    `breaks_after` adds phrase breaks after those raw indices (Transcribe punctuation items, turn ends). For the multi-word rules (dates,
    currency) a break ends the match unless it is a comma (a trailing "," on the raw word, or an index in `commas_after`);
    `boundaries_after` always ends a match (turn ends, speaker changes) without adding a phrase break. `applied` lists the scoring-only
    rules that fired: (rule, raw text, origin) so a display can say which rule explains a match. `rules` switches rules off by name.
    """
    rules = frozenset(rules)
    notes, applied, runs, run = [], [], [], []
    extra, boundaries, commas = set(breaks_after), set(boundaries_after), set(commas_after)
    hard_after = [False] * len(raw_words)  # a break the multi-word rules must not cross
    for index, raw in enumerate(raw_words):
        pieces, breaks = _pieces(raw, index, notes, rules, applied)
        run.extend(pieces)
        broke = breaks or index in extra
        if broke:
            runs.append(run)
            run = []
        comma = index in commas or unicodedata.normalize("NFKC", raw)[-1:] == ","
        hard_after[index] = index in boundaries or (broke and not comma)
    runs.append(run)
    seq = [piece for run in runs for piece in _combine(run, notes)]

    def connected(a, b):
        return not any(hard_after[w] for w in range(max(a.origin), min(b.origin)))

    if "zh_numbers" in rules:
        seq = _apply_zh_numbers(seq, connected, raw_words, applied)
    if "dates" in rules:
        seq = _apply_dates(seq, connected, raw_words, applied)
    seq = _apply_currency(seq, connected, raw_words, notes, applied, rules)
    tokens, origins = [], []
    for piece in seq:
        origin = tuple(sorted(piece.origin))
        if piece.kind == "num":
            for character in piece.text:
                tokens.append("point" if character == "." else character)
                origins.append(origin)
        else:
            tokens.append(piece.text)
            origins.append(origin)
    return tokens, origins, [*notes, *((f"applied_{rule}", text) for rule, text, _ in applied)], applied


def normalize_words(raw_words, breaks_after=(), **options):
    """Raw words -> (tokens, origins, notes); see normalize_words_traced."""
    tokens, origins, notes, _ = normalize_words_traced(raw_words, breaks_after, **options)
    return tokens, origins, notes


def normalize_text(text, **options):
    """Convenience for tests and inspection: one string -> tokens."""
    return normalize_words(text.split(), **options)[0]
