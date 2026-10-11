"""Deterministic spoken-surface rendering of structured values, with an independent decoder for each form.

Python owns every rendered value: the model never writes a value, only a template with `[[PLACEHOLDER]]` slots. Each
`render_*` function maps (canonical value, form) to the exact surface that is spliced into the template; each
`decode_*` function maps a surface back to a canonical value so `compile` can prove the round trip.
"""

import re

DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_WORD_TO_DIGIT = {word: str(index) for index, word in enumerate(DIGIT_WORDS)} | {"oh": "0"}
_ORDINALS = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
    "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    "twentieth", "twenty-first", "twenty-second", "twenty-third", "twenty-fourth", "twenty-fifth", "twenty-sixth",
    "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first")
_SMALL = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
          "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")
_MAGNITUDES = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000, "b": 1_000_000_000, "billion": 1_000_000_000}


class RenderError(ValueError):
    """A value cannot be rendered in the requested form."""


# ---- NRIC / FIN ---------------------------------------------------------------------------------------
def _char_word(char):
    return DIGIT_WORDS[int(char)] if char.isdigit() else char


def render_nric(value, form, spoken_form=None):
    """K01. CHARACTER_BY_CHARACTER; OH_FOR_ZERO (every zero is 'oh'); DOUBLE_OR_TRIPLE_DIGIT (runs of the assigned length)."""
    if not re.fullmatch(r"[STFGM][0-9]{7}[A-Z]", str(value)):
        raise RenderError(f"{value!r} is not a canonical NRIC/FIN")
    if form == "CHARACTER_BY_CHARACTER":
        return " ".join(_char_word(c) for c in value)
    if form == "OH_FOR_ZERO":
        return " ".join("oh" if c == "0" else _char_word(c) for c in value)
    if form == "DOUBLE_OR_TRIPLE_DIGIT":
        length = {"DOUBLE": 2, "TRIPLE": 3}.get((spoken_form or {}).get("variant"))
        if length is None:
            raise RenderError("DOUBLE_OR_TRIPLE_DIGIT needs a DOUBLE or TRIPLE variant")
        words, index = [], 0
        while index < len(value):
            run = 1
            while value[index].isdigit() and index + run < len(value) and value[index + run] == value[index]:
                run += 1
            if value[index].isdigit() and run == length:
                words.append(("double" if length == 2 else "triple") + " " + _char_word(value[index]))
            else:  # a maximal run of any other length stays character by character, consumed whole
                words.extend(_char_word(value[index]) for _ in range(run))
            index += run
        return " ".join(words)
    raise RenderError(f"unsupported NRIC/FIN form {form}")


def decode_nric(surface):
    """Inverse of render_nric for every form (a repeated-digit word expands to its run)."""
    tokens, chars, index = surface.split(), [], 0
    while index < len(tokens):
        token = tokens[index].lower()
        if token in ("double", "triple"):
            digit = _WORD_TO_DIGIT.get(tokens[index + 1].lower()) if index + 1 < len(tokens) else None
            if digit is None:
                raise RenderError("'double'/'triple' must be followed by a digit word")
            chars.append(digit * (2 if token == "double" else 3))
            index += 2
            continue
        if token in _WORD_TO_DIGIT:
            chars.append(_WORD_TO_DIGIT[token])
        elif len(tokens[index]) == 1 and tokens[index].isalpha():
            chars.append(tokens[index].upper())
        else:
            raise RenderError(f"cannot decode {tokens[index]!r}")
        index += 1
    return "".join(chars)


# ---- date of birth ------------------------------------------------------------------------------------
def _two_digit_words(number):
    if number < 20:
        return _SMALL[number]
    tens, ones = divmod(number, 10)
    return _TENS[tens] + (f"-{_SMALL[ones]}" if ones else "")


def year_words(year):
    """19xx years as two words ('nineteen fifty-five'); 2000 stays four digits (the pilot spoken verifier's contract)."""
    if 1950 <= year <= 1999:
        return "nineteen " + _two_digit_words(year - 1900)
    if year == 2000:
        return "2000"
    raise RenderError(f"year {year} is outside the supported 1950-2000 range")


def _split_dob(value):
    match = re.fullmatch(r"(\d{2}) ([A-Za-z]+) (\d{4})", str(value))
    if not match or match.group(2) not in _MONTHS:
        raise RenderError(f"{value!r} is not a canonical DD Month YYYY date")
    return int(match.group(1)), match.group(2), int(match.group(3))


def render_dob(value, form):
    """K02. DAY_MONTH_YEAR '27 October 1991'; ORDINAL_DAY 'the twenty-seventh of October nineteen ninety-one'."""
    day, month, year = _split_dob(value)
    if form == "DAY_MONTH_YEAR":
        return f"{day} {month} {year}"
    if form == "ORDINAL_DAY":
        return f"the {_ORDINALS[day - 1]} of {month} {year_words(year)}"
    raise RenderError(f"unsupported date form {form}")


def decode_dob(surface):
    text = surface.strip().lower().replace("-", " ")
    for month in _MONTHS:
        if month.lower() not in text.split():
            continue
        before, _, after = text.partition(month.lower())
        day = None
        match = re.search(r"\b(\d{1,2})\b", before)
        if match:
            day = int(match.group(1))
        else:
            words = [w for w in before.split() if w not in ("the", "of")]
            for index, ordinal in enumerate(_ORDINALS, start=1):
                if words == ordinal.replace("-", " ").split():
                    day = index
        year_match = re.search(r"\b(\d{4})\b", after)
        if year_match:
            year = int(year_match.group(1))
        else:
            year = next((y for y in range(1950, 2001) if year_words(y).replace("-", " ") == after.strip()), None)
        if day is None or year is None:
            raise RenderError(f"cannot decode date {surface!r}")
        return f"{day:02d} {month} {year}"
    raise RenderError(f"no month in {surface!r}")


# ---- amounts, quantities, currencies, direction, order type -----------------------------------------------
def format_number(value):
    """Integers with comma grouping; other numbers with up to two decimals and no trailing zeros."""
    number = float(value)
    if number == int(number):
        return f"{int(number):,}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def decode_number(surface):
    """'1,500' -> 1500, '500k' -> 500000, '10m' -> 10000000, '1.2 million' -> 1200000, '0.3' -> 0.3."""
    text = surface.strip().lower().replace(",", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|thousand|m|million|b|billion)?", text)
    if not match:
        raise RenderError(f"cannot decode number {surface!r}")
    number = float(match.group(1)) * _MAGNITUDES.get(match.group(2), 1)
    return int(number) if number == int(number) else number


def render_amount(value, form, spoken_form=None):
    """T03 / T04. DIGITS_WITH_THOUSANDS_SEPARATORS; a REQUIRED FINANCIAL_SHORTHAND / MAGNITUDE_EXPRESSION uses its text."""
    if form == "DIGITS_WITH_THOUSANDS_SEPARATORS":
        return format_number(value)
    if form in ("FINANCIAL_SHORTHAND", "MAGNITUDE_EXPRESSION"):
        text = (spoken_form or {}).get("text")
        if not text:
            raise RenderError(f"{form} needs the skeleton's required text")
        return text
    raise RenderError(f"unsupported amount form {form}")


def currency_alias(code, aliases):
    return aliases[code][0]


def render_currency(code, form, aliases):
    """T02 / CURRENCY. NATURAL_NAME is the first listed alias ('Singapore dollars'); ISO_CODE is the code."""
    if form == "NATURAL_NAME":
        return currency_alias(code, aliases)
    if form == "ISO_CODE":
        return code
    raise RenderError(f"unsupported currency form {form}")


def decode_currency(surface, aliases):
    for code, names in aliases.items():
        if surface == code or surface in names:
            return code
    raise RenderError(f"cannot decode currency {surface!r}")


def render_direction(value, form):
    if form != "NATURAL_VERB":
        raise RenderError(f"unsupported direction form {form}")
    return {"BUY": "buy", "SELL": "sell"}[value]


def render_order_type(value, form):
    if form != "NATURAL_PHRASE":
        raise RenderError(f"unsupported order-type form {form}")
    return {"MARKET": "a market order", "LIMIT": "a limit order"}[value]


def render_product(value, form):
    if form != "FULL_PRODUCT_NAME":
        raise RenderError(f"unsupported product form {form}")
    return str(value)


# ---- scenario slots (fee, limit price) ----------------------------------------------------------------
def render_fee(fee, aliases):
    """G3 fee_disclosure: a RATE as '0.3%' or an AMOUNT as '35 Singapore dollars'."""
    if fee["basis"] == "RATE":
        return f"{format_number(fee['value'])}%"
    return f"{format_number(fee['value'])} {currency_alias(fee['currency'], aliases)}"


def render_limit_price(price, currency, aliases):
    return f"{format_number(price)} {currency_alias(currency, aliases)}"
