"""Template syntax and the mechanical lexical protections for model-authored text.

Everything here applies to the template text OUTSIDE authorised `[[PLACEHOLDER]]` slots. Python-rendered values (dates,
amounts, quantities, limit prices, fees) are never scanned: they are the authoritative values, not model text.
Only reliable lexical cases are checked; whether a sentence really establishes a compliance behaviour is G5's judgement.
"""

import re

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


def parse_template(text):
    """(placeholders [(name, start, end)], outside_text, errors). `outside_text` joins the non-placeholder segments."""
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
    if not re.search(r"[A-Za-z]", outside):
        errors.append("text: needs wording outside the placeholders")
    return placeholders, outside, errors


class Lexicon:
    """Compiled lexical bans for one dataset (built once per inputs object)."""

    def __init__(self, inputs):
        profile = inputs.currency_profile
        words = set(_GENERIC_CURRENCY_WORDS)
        for code, names in profile["aliases"].items():
            words.add(code)
            words.update(names)
        ordered = sorted(words, key=len, reverse=True)
        self.currency = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(w) for w in ordered) + r")(?![A-Za-z])", re.IGNORECASE)
        names = set()
        for product in inputs.product_category:
            names.add(product)
            names.add(" ".join(product.split()[:2]))
        self.products = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?![A-Za-z])",
                                   re.IGNORECASE)

    def errors(self, outside):
        found = []
        if _DIGITS.search(outside):
            found.append("digits")
        if _MAGNITUDE.search(outside):
            found.append("magnitude words (hundred/thousand/million...)")
        if _PERCENT.search(outside) or _CURRENCY_SYMBOL.search(outside):
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
