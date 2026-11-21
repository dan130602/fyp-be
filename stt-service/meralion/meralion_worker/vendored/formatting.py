import re

HALF_PERCENT = re.compile(r"\bhalf\s+a\s+per\s*cent\b", re.I)
FRACTION_PREFIX = re.compile(
    r"(?:\w+[-–]|\b(?:and|one|two|three|four|five|six|seven|eight|nine|[0-9]+)\s+(?:a\s+)?)$",
    re.I,
)

def format_fee(text):
    """Formatter v1: render standalone half-percent phrases equivalently; leave compound fractions alone."""
    def replace(match):
        before=text[max(0,match.start()-40):match.start()]
        return match.group() if FRACTION_PREFIX.search(before) else '0.5 percent'
    return HALF_PERCENT.sub(replace,text)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Formatter v2: English spoken percentage fractions -> decimal percent. One closed grammar; anything ambiguous is left exactly as it was.
#   half (a) percent | a half percent        -> 0.5 percent        a quarter (of a) percent | one quarter ...   -> 0.25 percent      three quarters (of a) percent -> 0.75 percent
#   <N> and (a) half|quarter|three quarters percent  -> <N>.5 | <N>.25 | <N>.75 percent   (N = 0-99, digits or spelled; parsed BEFORE the simple forms: longest match wins)
# The whole expression must end with the unit `percent` / `per cent`. Abstains (text unchanged) when the expression is part of a range or list (to, or, -, between, from ... ranging ...,
# `and` + another number), when it follows a number or hyphenated word, or when anything else about it is not exactly this grammar. Words around a rewrite (not, about, up to, minus ...) are untouched,
# so meaning is preserved. Scope: English percentage/fraction wording only; not general number, date or currency normalisation. Chinese and written forms (0.5%) are never touched.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FORMATTER_VERSION = "2"
_UNIT = r"per[\s-]?cent\b"
_ONES = "one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_NUMBER_WORDS = {w: i + 1 for i, w in enumerate(_ONES)}
_NUMBER_WORDS.update({w: 10 + i for i, w in enumerate(_TEENS)})
_NUMBER_WORDS.update({"zero": 0})
_TENS_VALUE = {w: 20 + 10 * i for i, w in enumerate(_TENS)}
_NUMBER = r"(?:\d{1,2}|(?:%s)(?:[\s-](?:%s))?|(?:%s))" % ("|".join(_TENS), "|".join(_ONES), "|".join(_ONES + _TEENS + ["zero"]))
_FRACTION = r"(?:(?:a\s+)?half|(?:a\s+|one\s+)?quarter|three[\s-]quarters?)"
_COMPOUND = re.compile(rf"(?<![\w.%-])(?P<n>{_NUMBER})\s+and\s+(?P<f>(?:a\s+)?half|(?:a\s+)?quarter|three[\s-]quarters?)\s+{_UNIT}", re.I)
_SIMPLE = re.compile(rf"(?<![\w.%-])(?P<f>(?:a\s+)?half(?:\s+a)?|(?:a\s+|one\s+)?quarter(?:\s+of\s+a)?|three[\s-]quarters?(?:\s+of\s+a)?)\s+{_UNIT}", re.I)
_NUMBERISH = re.compile(rf"(?:\b(?:{'|'.join(_ONES + _TEENS + _TENS + ['zero', 'hundred', 'half', 'quarter', 'quarters', 'and', 'a', 'to', 'or', 'between', 'from'])})|[0-9%.]|[-–—])\s*$", re.I)
_RANGE_BEFORE = re.compile(r"\b(?:between|ranging|range|ranges)\b", re.I)         # "from X to Y" is caught by the to/or/- guards on both ends
_NUMBER_BEFORE_TO = re.compile(rf"(?:\d|\b(?:{'|'.join(_ONES + _TEENS + _TENS + ['zero', 'half', 'quarter', 'quarters', 'percent', 'cent'])}))[\s%]*\s+(?:to|through|until|or)\s+$", re.I)
_AFTER_RANGE = re.compile(rf"^\s*(?:[-–—]|(?:to|or|through|until)\b|and\s+(?:a\s+)?(?:\d|(?:{'|'.join(_ONES + _TEENS + _TENS + ['zero', 'half', 'quarter', 'quarters', 'three'])})\b))", re.I)


def _spelled(word):
    parts = re.split(r"[\s-]+", word.lower().strip())
    if len(parts) == 1:
        return _TENS_VALUE.get(parts[0]) if parts[0] in _TENS_VALUE else _NUMBER_WORDS.get(parts[0])
    if len(parts) == 2 and parts[0] in _TENS_VALUE and parts[1] in _NUMBER_WORDS and _NUMBER_WORDS[parts[1]] < 10:
        return _TENS_VALUE[parts[0]] + _NUMBER_WORDS[parts[1]]
    return None


def _fraction(f):
    f = f.lower()
    if "three" in f:
        return "75"
    if "quarter" in f:
        return "25"
    return "5"


def _abstain(text, start, end):
    """True when the expression text[start:end] sits in a range, a list or a number it would corrupt."""
    before, after = text[max(0, start - 70):start], text[end:end + 40]
    before = re.split(r"[.!?;:。！？]\s*", before)[-1] if re.search(r"[.!?;:。！？]\s*\S", before) else before     # only the current sentence can make this a range
    if _RANGE_BEFORE.search(before[-60:]):
        return True
    if _NUMBER_BEFORE_TO.search(before) or re.search(r"(?:[-–—]|\b(?:and|or|to)\b)\s*$", before, re.I) and not re.search(r"\b(?:up|down|close|next|due)\s+to\s*$", before, re.I):
        return True
    return bool(_AFTER_RANGE.search(after)) or bool(re.match(r"\s*[-–—]\s*\d", after))


def format_percentages(text):
    """Formatter v2 (see the grammar above). Raw text is never modified in place; a new string is returned."""
    def compound(match):
        if _abstain(text, match.start(), match.end()):
            return match.group()
        n = match.group("n")
        value = int(n) if n.isdigit() else _spelled(n)
        if value is None:
            return match.group()
        return f"{value}.{_fraction(match.group('f'))} percent"
    first = _COMPOUND.sub(compound, text)

    def simple(match):
        if _abstain(first, match.start(), match.end()):
            return match.group()
        before = first[max(0, match.start() - 40):match.start()]
        if FRACTION_PREFIX.search(before):                                    # "one and a half a percent", "one-half a percent": not this grammar
            return match.group()
        if re.search(r"\b(?:" + "|".join(_ONES + _TEENS + _TENS + ["zero", "hundred", "thousand", "million"]) + r")\s*$", before, re.I):
            return match.group()
        return f"0.{_fraction(match.group('f'))} percent"
    return _SIMPLE.sub(simple, first)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Amount formatter amount-v1 (separate entry point; formatter v2 above is untouched): remove ONE comma that splits a spelled English amount, "one million, two hundred thousand" ->
# "one million two hundred thousand". The words are kept; only that comma goes; digits are never produced. Descending magnitudes alone prove nothing ("Two transactions: one million, two
# hundred thousand" can be two amounts), so the join needs POSITIVE evidence of a single amount and otherwise ABSTAINS:
#   shape     A = a well-formed spelled number with a multiplier that ends in thousand/million/billion; B = a well-formed spelled number smaller than A's scale unit ("and" allowed);
#             B is not followed by another `, number` / `and number`; A is not preceded by `number,` / `number and`
#   evidence  B is immediately followed by a currency or unit word (dollars, US dollars, yuan, yen, units, shares, lots, ...), OR A follows a singular-amount cue in the same sentence
#             (amount ... is/of, price of/at/is, total is, balance is, shows)
#   blockers  anywhere in the same sentence: a plural count noun (transactions, amounts, orders, trades, figures, items, payments), a list word (both, each, either, neither, respectively,
#             versus), a range cue (between, from ... to, number to/or number), or a colon before A
# Latin text only. Nothing outside the matched comma is changed.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AMOUNT_FORMATTER_VERSION = "amount-v1"
_SCALES = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
_AMOUNT_WORD = "|".join(_ONES + _TEENS + _TENS + ["zero", "hundred", "thousand", "million", "billion"])
_AMOUNT_TOKEN = rf"(?:(?:{_AMOUNT_WORD})(?:-(?:{'|'.join(_ONES)}))?)"
_AMOUNT_GROUP = rf"{_AMOUNT_TOKEN}(?:\s+(?:and\s+)?{_AMOUNT_TOKEN})*"
_SPLIT_AMOUNT = re.compile(rf"(?<![\w-])(?P<a>{_AMOUNT_GROUP})[ \t]*[,，][ \t]*(?P<b>{_AMOUNT_GROUP})(?![\w-])", re.I)
_CURRENCY_ADJ = r"(?:us|u\.s\.|singapore|japanese|chinese|hong\s+kong|australian|canadian|new\s+zealand)"
_AMOUNT_UNIT_AFTER = re.compile(rf"^[ \t]+(?:{_CURRENCY_ADJ}[ \t]+)?(?:dollars?|yuan|renminbi|rmb|yen|euros?|pounds?|sterling|units?|shares?|lots?)\b", re.I)
_AMOUNT_CUE_BEFORE = re.compile(r"(?:\bamount(?:\s+[\w']+){0,6}\s+(?:is|of)|\b(?:limit\s+)?price\s+(?:of|at|is)|\btotal\s+(?:is|of)|\bbalance\s+(?:is|of)|\bshows)\s*$", re.I)
_AMOUNT_NUM = rf"(?:\d+|{_AMOUNT_TOKEN})"
_AMOUNT_BLOCKERS = [
    (re.compile(r"\b(?:transactions|amounts|orders|trades|figures|items|payments)\b", re.I), "plural count noun in the sentence"),
    (re.compile(r"\b(?:both|each|either|neither|respectively|versus|vs)\b", re.I), "list word in the sentence"),
    (re.compile(r"\bbetween\b", re.I), "range word 'between'"),
    (re.compile(r"\bfrom\b(?:\s+[\w,'.-]+){0,12}?\s+(?:to|through|until)\b", re.I), "range 'from ... to'"),
    (re.compile(rf"{_AMOUNT_NUM}\s+(?:to|or|through|until)\b|\b(?:to|or|through|until)\s+{_AMOUNT_NUM}\b", re.I), "number to/or number"),
]
_BEFORE_NUMBER_LIST = re.compile(rf"(?:{_AMOUNT_NUM}\s*[,，]|\b{_AMOUNT_NUM}\s+(?:and|or))\s*$", re.I)
_AFTER_NUMBER_LIST = re.compile(rf"^\s*(?:[,，]\s*(?:(?:and|or)\s+)?|\s(?:and|or)\s+){_AMOUNT_NUM}\b", re.I)


def _spoken_value(words):
    """Value of a well-formed spelled English number (hyphenated tens allowed, "and" after hundred or a scale), else None. Scales must descend, hundred follows 1-9 once per group."""
    total, current, last_scale, seen_hundred, last = 0, 0, None, False, None
    for raw in words:
        w = raw.lower()
        if w == "and":
            if last not in ("hundred", "scale"):
                return None
            continue
        parts = w.split("-")
        if len(parts) == 2 and parts[0] in _TENS_VALUE and parts[1] in _ONES:
            if current % 100 != 0:
                return None
            current += _TENS_VALUE[parts[0]] + _NUMBER_WORDS[parts[1]]
            last = "ones"
        elif w in _SCALES:
            if current == 0 or (last_scale is not None and _SCALES[w] >= last_scale):
                return None
            total += current * _SCALES[w]
            current, last_scale, seen_hundred, last = 0, _SCALES[w], False, "scale"
        elif w == "hundred":
            if seen_hundred or not 1 <= current <= 9:
                return None
            current *= 100
            seen_hundred, last = True, "hundred"
        elif w in _TENS_VALUE:
            if current % 100 != 0:
                return None
            current += _TENS_VALUE[w]
            last = "tens"
        elif w in _NUMBER_WORDS:
            value = _NUMBER_WORDS[w]
            if value >= 10:                                              # teens and ten
                if current % 100 != 0:
                    return None
            elif last != "tens" and current % 100 != 0:                  # ones after a tens word, or after nothing in this hundreds group
                return None
            current += value
            last = "ones"
        else:
            return None
    return total + current


def _sentence_around(text, start, end):
    left = max((text.rfind(c, 0, start) for c in ".!?;。！？\n"), default=-1)
    right_positions = [p for p in (text.find(c, end) for c in ".!?;。！？\n") if p != -1]
    return left + 1, (min(right_positions) if right_positions else len(text))


def join_split_amounts_report(text):
    """(new text, events). Every candidate comma between two spelled number groups is an event: {"at": position, "text": the matched span, "action": "joined" | "abstained", "reason": ...}."""
    events, pieces, last = [], [], 0
    for m in _SPLIT_AMOUNT.finditer(text):
        a_words, b_words = re.split(r"\s+", m.group("a").strip()), re.split(r"\s+", m.group("b").strip())
        scale = _SCALES.get(a_words[-1].lower())
        a_value, b_value = _spoken_value(a_words), _spoken_value(b_words)

        def event(action, reason, m=m):
            events.append({"at": m.start(), "text": m.group(0), "action": action, "reason": reason})
        if scale is None:
            event("abstained", "A does not end in a scale word")
            continue
        if a_value is None or b_value is None or a_value < scale:
            event("abstained", "A or B is not a well-formed spelled number with a multiplier")
            continue
        if b_value >= scale:
            event("abstained", "B is not smaller than A's scale unit")
            continue
        before, after = text[max(0, m.start() - 80):m.start()], text[m.end():m.end() + 40]
        if _BEFORE_NUMBER_LIST.search(before) or _AFTER_NUMBER_LIST.search(after):
            event("abstained", "part of a list of numbers")
            continue
        s0, s1 = _sentence_around(text, m.start(), m.end())
        sentence = text[s0:s1]
        blocked = next((why for rx, why in _AMOUNT_BLOCKERS if rx.search(sentence)), None) or ("colon before the amount" if ":" in text[s0:m.start()] else None)
        if blocked:
            event("abstained", blocked)
            continue
        evidence = ("unit or currency word follows" if _AMOUNT_UNIT_AFTER.search(text[m.end():m.end() + 40]) else
                    "singular-amount cue before" if _AMOUNT_CUE_BEFORE.search(text[s0:m.start()]) else None)
        if evidence is None:
            event("abstained", "no evidence of a single amount")
            continue
        pieces.append(text[last:m.start()])
        pieces.append(f"{m.group('a')} {m.group('b')}")
        last = m.end()
        event("joined", evidence)
    pieces.append(text[last:])
    return "".join(pieces), events


def join_split_amounts(text):
    """amount-v1: see the grammar above. Raw text is never modified in place; a new string is returned."""
    return join_split_amounts_report(text)[0]
