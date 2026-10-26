"""Template syntax and the mechanical lexical protections for model-authored text.

Everything here applies to the template text OUTSIDE authorised `[[PLACEHOLDER]]` slots. Python-rendered values (dates,
amounts, quantities, limit prices, fees) are never scanned: they are the authoritative values, not model text.
Only reliable lexical cases are checked; whether a sentence really establishes a compliance behaviour is G5's judgement.
"""

import re

from fyp_evaluation.validation.stage0 import HAN, is_multilingual

from .config import ALL_PLACEHOLDERS, MAX_TEMPLATE_CHARACTERS

PLACEHOLDER = re.compile(r"\[\[([A-Z0-9_]+)\]\]")
_STRAY = re.compile(r"\[|\]|\{|\}|<|>|\n|\r|\t")

_DIGITS = re.compile(r"\d")
_MAGNITUDE = re.compile(r"\b(?:hundreds?|thousands?|millions?|billions?|trillions?|lakhs?|crores?|dozens?)\b", re.IGNORECASE)
_PERCENT = re.compile(r"%|\bper\s?cent\b|\bpercent(?:age)?s?\b|\bbasis\s+points?\b|\bbps\b", re.IGNORECASE)
_CURRENCY_SYMBOL = re.compile(r"[$€¥£]")
_GENERIC_CURRENCY_WORDS = ("dollar", "dollars", "euro", "euros", "yuan", "yen", "renminbi", "rmb", "sterling", "pound", "pounds")

FALSE_START_END = re.compile(r"(?:—|–|-|\.\.\.|…)\s*$")
WEAK_ACK_BANNED = re.compile(
    r"\b(?:yes|yeah|yep|confirm\w*|correct|right|agree\w*|go\s+ahead|proceed|please\s+do|approved?|sure|definitely|"
    r"absolutely|exactly)\b", re.IGNORECASE)
GUARANTEE_WORD = re.compile(r"\bguarante\w*", re.IGNORECASE)
LAUNDERING_WORDS = re.compile(
    r"\b(?:launder\w*|conceal\w*|undeclared|undisclosed\s+(?:cash|funds|source)|shell\s+compan\w+|structuring)\b", re.IGNORECASE)
INSIDER_WORDS = re.compile(
    r"\b(?:inside(?:r)?\s+(?:information|knowledge|tips?)|insider|non-?public|material\s+non|confidential\s+(?:information|tips?)|"
    r"unannounced|not\s+(?:yet\s+)?public)\b", re.IGNORECASE)


# Schema-2.0 variants of the English patterns. A word boundary does not separate a Latin word from an adjacent Han character (both are
# word characters), so "yes啦" or "guarantee的" would slip through; these use explicit Latin-letter lookarounds instead. Schema 1.0 keeps
# the original patterns above, unchanged.
_A, _Z = r"(?<![A-Za-z])", r"(?![A-Za-z])"
_MAGNITUDE_SAFE = re.compile(_A + r"(?:hundreds?|thousands?|millions?|billions?|trillions?|lakhs?|crores?|dozens?)" + _Z, re.IGNORECASE)
_PERCENT_SAFE = re.compile(r"[%％]|" + _A + r"per\s?cent" + _Z + "|" + _A + r"percent(?:age)?s?" + _Z + "|" + _A + r"basis\s+points?" + _Z
                           + "|" + _A + "bps" + _Z, re.IGNORECASE)
_CURRENCY_SYMBOL_SAFE = re.compile(r"[$€¥£＄￥￡]")
WEAK_ACK_BANNED_SAFE = re.compile(
    _A + r"(?:yes|yeah|yep|confirm\w*|correct|right|agree\w*|go\s+ahead|proceed|please\s+do|approved?|sure|definitely|"
    r"absolutely|exactly)" + _Z, re.IGNORECASE)
GUARANTEE_WORD_SAFE = re.compile(_A + r"guarante\w*", re.IGNORECASE)
LAUNDERING_WORDS_SAFE = re.compile(
    _A + r"(?:launder\w*|conceal\w*|undeclared|undisclosed\s+(?:cash|funds|source)|shell\s+compan\w+|structuring)" + _Z, re.IGNORECASE)
INSIDER_WORDS_SAFE = re.compile(
    _A + r"(?:inside(?:r)?\s+(?:information|knowledge|tips?)|insider|non-?public|material\s+non|confidential\s+(?:information|tips?)|"
    r"unannounced|not\s+(?:yet\s+)?public)" + _Z, re.IGNORECASE)
_WORDING = re.compile(r"[A-Za-z]")
_WORDING_MULTILINGUAL = re.compile(r"[A-Za-z]|" + HAN.pattern)


def parse_template(text, multilingual=False):
    """(placeholders [(name, start, end)], outside_text, errors). `outside_text` joins the non-placeholder segments.

    Schema 1.0 needs Latin wording outside the slots; schema 2.0 accepts Latin or Han wording."""
    errors = []
    if not isinstance(text, str) or not text.strip():
        return [], "", ["text: must not be empty"]
    if len(text) > MAX_TEMPLATE_CHARACTERS:
        errors.append(f"text: longer than {MAX_TEMPLATE_CHARACTERS} characters")
    placeholders, segments, cursor = [], [], 0
    for match in PLACEHOLDER.finditer(text):
        placeholders.append((match.group(1), match.start(), match.end()))
        segments.append(text[cursor:match.start()])
        cursor = match.end()
    segments.append(text[cursor:])
    outside = " ".join(segments)
    if _STRAY.search(outside):
        errors.append("text: contains stray brackets, braces, angle brackets or line breaks outside valid [[PLACEHOLDER]] slots")
    for name, _, _ in placeholders:
        if name not in ALL_PLACEHOLDERS:
            errors.append(f"text: unknown placeholder [[{name}]]")
    if not (_WORDING_MULTILINGUAL if multilingual else _WORDING).search(outside):
        errors.append("text: needs wording outside the placeholders")
    return placeholders, outside, errors


class Lexicon:
    """Compiled lexical bans for one dataset (built once per inputs object)."""

    def __init__(self, inputs):
        self.multilingual = is_multilingual(inputs.documents)
        safe = self.multilingual
        self.magnitude, self.percent = (_MAGNITUDE_SAFE, _PERCENT_SAFE) if safe else (_MAGNITUDE, _PERCENT)
        self.currency_symbol = _CURRENCY_SYMBOL_SAFE if safe else _CURRENCY_SYMBOL
        self.weak_ack = WEAK_ACK_BANNED_SAFE if safe else WEAK_ACK_BANNED
        self.guarantee = GUARANTEE_WORD_SAFE if safe else GUARANTEE_WORD
        self.laundering = LAUNDERING_WORDS_SAFE if safe else LAUNDERING_WORDS
        self.insider = INSIDER_WORDS_SAFE if safe else INSIDER_WORDS
        profile = inputs.currency_profile
        words = set(_GENERIC_CURRENCY_WORDS)
        for code, names in profile["aliases"].items():
            words.add(code)
            words.update(names)
        self.han_currency_markers = tuple(sorted(word for word in words if HAN.search(word)))
        ordered = sorted(words, key=len, reverse=True)
        self.currency = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(w) for w in ordered) + r")(?![A-Za-z])", re.IGNORECASE)
        names = set()
        for product in inputs.product_category:
            names.add(product)
            names.add(" ".join(product.split()[:2]))
        if self.multilingual:  # the approved Chinese product names may only be spoken through [[T01]]
            for entry in (inputs.documents["normalization"]["profiles"]["synthetic_product_name"].get("language_names") or {}).values():
                names.add(entry["zh"]["surface"])
        self.products = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?![A-Za-z])",
                                   re.IGNORECASE)

    def errors(self, outside):
        found = []
        if _DIGITS.search(outside):
            found.append("digits")
        if self.magnitude.search(outside):
            found.append("magnitude words (hundred/thousand/million...)")
        if self.percent.search(outside) or self.currency_symbol.search(outside):
            found.append("percent or currency symbols")
        if self.currency.search(outside):
            found.append("currency names or codes")
        if self.products.search(outside):
            found.append("product names")
        return [f"text: {item} are only allowed through placeholders, not in the template wording" for item in found]


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def shared_word_run(text, reference, length=5):
    """The first run of `length` consecutive words that `text` shares with `reference`, or None."""
    reference_words = words(reference)
    seen = {tuple(reference_words[i:i + length]) for i in range(len(reference_words) - length + 1)}
    candidate = words(text)
    for index in range(len(candidate) - length + 1):
        run = tuple(candidate[index:index + length])
        if run in seen:
            return " ".join(run)
    return None
