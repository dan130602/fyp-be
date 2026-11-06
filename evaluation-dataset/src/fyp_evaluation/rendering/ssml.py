"""Deterministic SSML for one dialogue turn.

Upstream (release `normalization.json` `spoken_forms` and the edge-case catalog) fixes HOW values are written and spoken: identifiers
character by character with digits as words ("S six five ... nine A"; "oh" for zero and "double"/"triple" only under their edge cases),
dates as "13 March 1999" or an ordinal, currencies by natural name, and financial shorthand ("500k", "10m") only under
EC_FINANCIAL_SHORTHAND, where the audio must keep the shorthand. The dialogue text is therefore already in spoken form and SSML only adds
`ssml-v3` pronunciation hints where a neural voice would misread a token. Every hint is a `<sub alias>` from the CLOSED table below, so
nothing depends on an engine switch (`say-as` is not used):

* the prefix and suffix LETTERS of a spoken identifier (a lone "A" would be read as the article): "A" -> "ay";
* spelled acronyms and currency codes (NRIC, FIN, ETF, US, SGD, USD, ...): letter names, "N R I C" -> "en ar eye see";
* financial shorthand keeps its suffix: "500k" -> "five hundred kay", "10m" -> "ten em". The suffix is never expanded to "thousand" or "million".

"oh", "double" and "triple" inside an identifier, dates, ordinals and everything else stay plain text for the voice. Hints are render inputs,
never part of the canonical text, and `assert_faithful` checks BOTH that the visible text is exactly the canonical turn text and that every
hint is the one the table prescribes; any other tag, attribute or alias is rejected.
"""

import re
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr

LETTER_NAMES = {"A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "eff", "G": "gee", "H": "aitch", "I": "eye", "J": "jay",
                "K": "kay", "L": "ell", "M": "em", "N": "en", "O": "oh", "P": "pee", "Q": "cue", "R": "ar", "S": "ess", "T": "tee",
                "U": "you", "V": "vee", "W": "double-you", "X": "ex", "Y": "why", "Z": "zed"}
SPELLED = ("NRIC", "FIN", "ETF", "US", "SGD", "USD", "EUR", "CNY", "JPY")  # guarded against normalization.json by a test
SHORTHAND_UNITS = {"k": "kay", "m": "em"}  # the suffix is spoken as its letter, never expanded

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_DIGIT = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_RUN = rf"(?:(?:double|triple) {_DIGIT}|{_DIGIT})"
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)([km])")
_HINT = re.compile(
    rf"(?<![\w'])(?P<prefix>[STFGM])(?P<middle>(?:[ ,]+{_RUN}){{2,}})[ ,]+(?P<suffix>[A-Z])(?![\w'])"
    r"|(?<![\w.])(?P<amount>\d+(?:\.\d+)?[km])(?!\w)"
    rf"|(?<![\w'])(?P<word>{'|'.join(SPELLED)})(?![\w'])")


def _integer_words(number):
    if number < 20:
        return _ONES[number]
    if number < 100:
        return _TENS[number // 10] + (f"-{_ONES[number % 10]}" if number % 10 else "")
    if number < 1000:
        return f"{_ONES[number // 100]} hundred" + (f" {_integer_words(number % 100)}" if number % 100 else "")
    raise ValueError(f"unsupported shorthand amount {number}: only 0-999 can precede a k or m suffix")


def _amount_words(digits):
    whole, _, fraction = digits.partition(".")
    return _integer_words(int(whole)) + ("".join(f" point {_ONES[int(d)]}" if i == 0 else f" {_ONES[int(d)]}" for i, d in enumerate(fraction)))


def permitted_alias(token):
    """The one alias the closed table prescribes for a hinted token, or None when the token is not in the table."""
    if len(token) == 1 and token in LETTER_NAMES:
        return LETTER_NAMES[token]
    if token in SPELLED:
        return " ".join(LETTER_NAMES[letter] for letter in token)
    if (match := _AMOUNT.fullmatch(token)):
        return f"{_amount_words(match[1])} {SHORTHAND_UNITS[match[2]]}"
    return None


def _sub(token):
    return f"<sub alias={quoteattr(permitted_alias(token))}>{escape(token)}</sub>"


def make_ssml(text):
    """The exact SSML sent to Polly for one turn."""
    parts, last = [], 0
    for match in _HINT.finditer(text):
        parts.append(escape(text[last:match.start()]))
        if match["prefix"]:
            parts.append(_sub(match["prefix"]) + escape(match["middle"]))
            parts.append(escape(text[match.end("middle"):match.start("suffix")]) + _sub(match["suffix"]))
        else:
            parts.append(_sub(match["amount"] or match["word"]))
        last = match.end()
    parts.append(escape(text[last:]))
    ssml = f"<speak>{''.join(parts)}</speak>"
    assert_faithful(text, ssml)
    return ssml


def assert_faithful(text, ssml):
    """Require BOTH that the visible text is exactly the canonical turn text AND that every pronunciation hint is the closed-table one.

    Only `<speak>` (no attributes) and `<sub alias>` (alias equal to the table's alias for its text, no children) are allowed.
    """
    try:
        root = ElementTree.fromstring(ssml)
    except ElementTree.ParseError as error:
        raise ValueError(f"SSML is not valid XML ({error}); the turn text may contain characters XML cannot carry") from error
    if root.tag != "speak" or root.attrib:
        raise ValueError("SSML must be a plain <speak> element")
    for element in root.iter():
        if element is root:
            continue
        if element.tag != "sub" or set(element.attrib) != {"alias"} or len(element):
            raise ValueError(f"SSML tag <{element.tag}> or its attributes are not a permitted pronunciation hint")
        expected = permitted_alias(element.text or "")
        if expected is None or element.attrib["alias"] != expected:
            raise ValueError(f"SSML alias {element.attrib['alias']!r} for {element.text!r} is not the one the pronunciation table prescribes")
    if "".join(root.itertext()) != text:
        raise ValueError("SSML does not carry exactly the canonical turn text")
