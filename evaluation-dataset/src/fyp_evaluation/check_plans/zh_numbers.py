"""Mandarin number, date and identifier rendering with strict, independent decoders (normalization `language_rules.zh`).

The rules are algorithmic: no table lists every number. Rendering is canonical (one surface per value); decoding accepts the
canonical surface plus the approved equivalents (两 or 二 before 百/千/万/亿, 〇 or 零 inside a year, spaces inside an
identifier) and rejects everything else, including shortened colloquial forms (一千二, 两万五), 幺, mixed Arabic digits and
ambiguous or non-calendar dates. A decoder never sees the expected value: it maps a surface to a canonical value or raises.

Provisional: the golden examples and rules are reviewed by tests that are written independently of this module, but they
still need native-speaker review before they count as linguistically accepted.
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from .render import RenderError, _MONTHS

DIGITS = "零一二三四五六七八九"
YEAR_DIGITS = "〇一二三四五六七八九"
_DIGIT_VALUE = {c: i for i, c in enumerate(DIGITS)} | {"〇": 0}
_SMALL_UNITS = ("", "十", "百", "千")
MAX_CARDINAL = 10 ** 12 - 1
_HAN_NUMERAL = "零〇一二两三四五六七八九十百千万亿"


# ---- cardinals ----------------------------------------------------------------------------------------
def render_cardinal(number):
    """Canonical Han cardinal of an integer 0 <= n < 10**12.

    万/亿 grouping; 零 marks an internal gap; a leading 十 drops its 一 (十二, 十二万); 两 replaces 二 only as the leading
    digit of the whole number directly before 百, 千, 万 or 亿 (两百, 两千二百, 两万五千, 两亿); 二 everywhere else.
    """
    if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number <= MAX_CARDINAL:
        raise RenderError(f"{number!r} is not an integer in 0..{MAX_CARDINAL}")
    if number == 0:
        return "零"
    sections = []  # (value 0..9999, unit) from the most significant section down
    rest = number
    for unit, size in (("亿", 10 ** 8), ("万", 10 ** 4), ("", 1)):
        sections.append((rest // size, unit))
        rest %= size
    out, zero_pending = [], False
    for index, (value, unit) in enumerate(sections):
        if value == 0:
            zero_pending = bool(out)
            continue
        if value >= 10 ** 4:  # only the top (亿) section can exceed four digits; keep the digit walk uniform
            raise RenderError(f"{number} is too large")
        digits = [(value // 10 ** p) % 10 for p in (3, 2, 1, 0)]
        for position, digit in zip((3, 2, 1, 0), digits):
            if digit == 0:
                if out:
                    zero_pending = True
                continue
            if zero_pending:
                out.append("零")
                zero_pending = False
            leading = not out
            if digit == 1 and position == 1 and leading:
                out.append("十")  # 十二, 十二万: the 一 of a leading 十 is dropped
                continue
            followed_by_unit = position > 1 or (position == 0 and unit != "" and not any(digits[:3]))
            if digit == 2 and leading and followed_by_unit:
                out.append("两" + _SMALL_UNITS[position])
            else:
                out.append(DIGITS[digit] + _SMALL_UNITS[position])
        out.append(unit)  # a zero gap at the end of this section still needs its 零 before the next digit
    return "".join(out)


def _evaluate(text):
    """Lenient evaluation of a Han cardinal; strictness comes from comparing against the canonical rendering."""
    result = section = pending = 0
    seen_digit = False
    for char in text:
        if char in _DIGIT_VALUE and char not in "零〇":
            if pending:
                raise RenderError(f"cannot decode cardinal {text!r}")
            pending, seen_digit = _DIGIT_VALUE[char], True
        elif char == "两":
            if pending:
                raise RenderError(f"cannot decode cardinal {text!r}")
            pending, seen_digit = 2, True
        elif char in "零〇":
            if pending:
                raise RenderError(f"cannot decode cardinal {text!r}")
        elif char in "十百千":
            section += (pending or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            pending = 0
        elif char in "万亿":
            total = section + pending
            if total == 0:
                raise RenderError(f"cannot decode cardinal {text!r}")
            if char == "万":
                result += total * 10 ** 4
            else:
                result = (result + total) * 10 ** 8
            section = pending = 0
        else:
            raise RenderError(f"cannot decode cardinal {text!r}")
    if not seen_digit and not any(c in text for c in "十百千万亿"):
        if text and all(c in "零〇" for c in text):
            return 0
        raise RenderError(f"cannot decode cardinal {text!r}")
    return result + section + pending


def parse_cardinal(text):
    """The integer a canonical Han cardinal denotes. Accepts 两 or 二 only directly before 百/千/万/亿; rejects shortened or
    non-canonical forms (一千二, 两万五, 一十二, 一百十二, 二百零, trailing 零)."""
    if not isinstance(text, str) or not text or not re.fullmatch(f"[{_HAN_NUMERAL}]+", text):
        raise RenderError(f"cannot decode cardinal {text!r}")
    if re.search(r"两(?![百千万亿])", text):
        raise RenderError(f"两 is only used directly before 百, 千, 万 or 亿 in {text!r}")
    number = _evaluate(text)
    if number > MAX_CARDINAL or render_cardinal(number).replace("两", "二") != text.replace("两", "二").replace("〇", "零"):
        raise RenderError(f"{text!r} is not a canonical Mandarin cardinal")
    return number


# ---- decimals and percentages ---------------------------------------------------------------------------
def render_decimal(value, max_places=4):
    """A non-negative Decimal as integer part + 点 + digit-by-digit fraction (三十七点五, 零点七五)."""
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise RenderError(f"{value!r} is not a number") from error
    if not number.is_finite() or number < 0:
        raise RenderError(f"{value!r} is not a non-negative number")
    text = format(number.normalize(), "f")
    whole, _, fraction = text.partition(".")
    if len(fraction) > max_places:
        raise RenderError(f"{value!r} needs {len(fraction)} decimal places; at most {max_places} can be rendered exactly")
    out = render_cardinal(int(whole))
    return out + ("点" + "".join(DIGITS[int(c)] for c in fraction) if fraction else "")


def parse_decimal(text):
    whole, point, fraction = text.partition("点")
    if not whole or (point and not fraction) or "点" in fraction:
        raise RenderError(f"cannot decode number {text!r}")
    if point and (not all(c in DIGITS for c in fraction) or fraction.endswith("零")):
        raise RenderError(f"cannot decode the fraction of {text!r}")
    number = Decimal(parse_cardinal(whole))
    if point:
        number += Decimal("0." + "".join(str(DIGITS.index(c)) for c in fraction))
    return number


def render_percent(value, max_places=4):
    return "百分之" + render_decimal(value, max_places)


def parse_percent(text):
    if not text.startswith("百分之"):
        raise RenderError(f"cannot decode percentage {text!r}")
    return parse_decimal(text[len("百分之"):])


# ---- years and dates ------------------------------------------------------------------------------------
def render_year(year):
    if not isinstance(year, int) or not 1000 <= year <= 9999:
        raise RenderError(f"year {year!r} is outside 1000-9999")
    return "".join(YEAR_DIGITS[int(c)] for c in str(year))


def render_date(canonical):
    """'05 January 1967' -> 一九六七年一月五日."""
    match = re.fullmatch(r"(\d{2}) ([A-Za-z]+) (\d{4})", str(canonical))
    if not match or match.group(2) not in _MONTHS:
        raise RenderError(f"{canonical!r} is not a canonical DD Month YYYY date")
    day, month, year = int(match.group(1)), _MONTHS.index(match.group(2)) + 1, int(match.group(3))
    try:
        date(year, month, day)
    except ValueError as error:
        raise RenderError(f"{canonical!r} is not a calendar date") from error
    return f"{render_year(year)}年{render_cardinal(month)}月{render_cardinal(day)}日"


def parse_date(text):
    """A Han date back to the canonical 'DD Month YYYY'. Years accept 〇 or 零; an impossible or non-canonical date is rejected."""
    match = re.fullmatch(r"([〇零一二三四五六七八九]{4})年([^年月日]+)月([^年月日]+)日", text.strip())
    if not match:
        raise RenderError(f"cannot decode date {text!r}")
    year = int("".join(str(_DIGIT_VALUE[c]) for c in match.group(1)))
    month, day = parse_cardinal(match.group(2)), parse_cardinal(match.group(3))
    try:
        date(year, month, day)
    except ValueError as error:
        raise RenderError(f"{text!r} is not a calendar date") from error
    return f"{day:02d} {_MONTHS[month - 1]} {year}"


# ---- NRIC / FIN -----------------------------------------------------------------------------------------
def render_identifier(value):
    """'S8779219E' -> S八七七九二一九E: the letters stay uppercase Latin, each digit is a Han digit (二 always)."""
    if not re.fullmatch(r"[STFGM][0-9]{7}[A-Z]", str(value)):
        raise RenderError(f"{value!r} is not a canonical NRIC/FIN")
    return value[0] + "".join(DIGITS[int(c)] for c in value[1:8]) + value[8]


def parse_identifier(text):
    """Strip interior spaces (approved), then require prefix letter + seven Han digits + suffix letter."""
    compact = re.sub(r"[ 　]+", "", text)
    match = re.fullmatch(r"([A-Z])([零一二三四五六七八九]{7})([A-Z])", compact)
    if not match:
        raise RenderError(f"cannot decode identifier {text!r}")
    value = match.group(1) + "".join(str(DIGITS.index(c)) for c in match.group(2)) + match.group(3)
    if not re.fullmatch(r"[STFGM][0-9]{7}[A-NP-Z]", value):
        raise RenderError(f"{value!r} is not a canonical NRIC/FIN")
    return value
