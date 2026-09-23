"""Small pilot-only gold-authoring verifier. Never parse or repair predictions."""

import re

from .dates import is_canonical_dob

_CURRENCIES = {"SGD", "USD", "EUR", "GBP", "AUD"}
_ORDINALS = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    "twentieth", "twenty-first", "twenty-second", "twenty-third", "twenty-fourth",
    "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth",
    "twenty-ninth", "thirtieth", "thirty-first",
)
_SMALL = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

# Days only run 1-31, so "twenty"/"thirty" are the only tens words that can
# prefix a single ordinal/cardinal day-word (e.g. "twenty fourth", "thirty
# first"). A bare lookbehind for a non-word character is not enough to reject
# these: the space inside "twenty fourth" already satisfies it, letting day 4's
# word "fourth" match as a false suffix of day 24's compound. Guard against
# both the space- and hyphen-joined spellings for each such prefix.
_DAY_COMPOUND_PREFIXES = tuple(prefix for prefix in (_TENS[2], _TENS[3]) if prefix)
_DAY_PREFIX_GUARD = "".join(
    f"(?<!{prefix} )(?<!{prefix}-)" for prefix in _DAY_COMPOUND_PREFIXES
)


def _two_digit_words(number):
    if number < 20:
        return _SMALL[number]
    tens, units = divmod(number, 10)
    return _TENS[tens] + (" " + _SMALL[units] if units else "")


def _word_form_pattern(text):
    """Allow spacing/hyphenation within a generated closed-table word form."""
    return re.escape(text).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+")


def verify_spoken_value(expected, planned_text):
    """Return None on a permitted form, otherwise an error string.

    Input is a validated extraction expected object. Text uses exact substrings;
    the five pilot currencies and identifier-shaped text permit separators only
    between canonical characters; identifier digits may use closed zero-nine
    words. DOB forms are generated, never parsed from
    speech. Other scalar kinds deliberately remain outside this pilot verifier.
    """
    if expected.get("kind") != "extraction":
        return "Spoken-value verification requires an extraction expected object"
    value = expected["expected_value"]
    if value is None:
        return None
    kind = expected["value_type"]
    if kind == "text":
        if value in _CURRENCIES or re.fullmatch(r"[STFGM][0-9]{7}[A-Z]", value):
            characters = [
                f"(?:{char}|(?i:{_SMALL[int(char)]}))" if char.isdigit() else re.escape(char)
                for char in value
            ]
            # Bound the whole match so the locked value cannot be accepted as a
            # substring of a longer alphanumeric run (e.g. locked S0000000X
            # inside planned text XS0000000X or S0000000XY).
            pattern = r"(?<![\w-])" + r"[\s,.\-]*".join(characters) + r"(?![\w-])"
            matches = re.search(pattern, planned_text) is not None
        else:
            matches = value in planned_text
    elif kind == "dob_dd_month_yyyy":
        if not is_canonical_dob(value):
            return "Expected DOB must already be canonical DD Month YYYY"
        day, month, year = value.split(" ")
        days = (day, str(int(day)), _ORDINALS[int(day) - 1], _two_digit_words(int(day)))
        years = [year]
        # Two-part year speech is needed only for the pilot's 19xx DOBs.
        # Other canonical years still match their unchanged four-digit form.
        if year.startswith("19") and int(year[2:]) >= 10:
            years.append(_two_digit_words(int(year[:2])) + " " + _two_digit_words(int(year[2:])))
        matches = any(
            re.search(r"(?<![\w-])" + _DAY_PREFIX_GUARD + _word_form_pattern(spoken_day)
                      + r"(?:\s+of)?\s+" + re.escape(month) + r"(?:,\s*|\s+)"
                      + _word_form_pattern(spoken_year)
                      + r"(?![\w-])", planned_text, flags=re.IGNORECASE) is not None
            for spoken_day in days for spoken_year in years
        )
    else:
        return f"Spoken-value verification is not implemented for pilot value_type {kind!r}"
    return None if matches else f"Planned text does not denote locked {kind} value {value!r}"
