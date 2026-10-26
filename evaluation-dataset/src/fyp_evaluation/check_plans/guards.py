"""Schema-2.0 mechanical guards on model-authored template text (outside the authorised `[[PLACEHOLDER]]` slots).

These checks are mechanical and deliberately narrow. They decide only whether the wording can be trusted not to carry a
value, a script or a language the call's frozen requirements forbid. They never judge meaning: whether a sentence
establishes a compliance behaviour, contradicts a value or reads naturally is G5/G7's job, and the Mandarin compliance
wording (guarantees, laundering, insider information) is deliberately NOT checked here at all.

The value-expression guard is not a Mandarin number parser and does not prove that no unauthorised value is present. It
rejects clearly identifiable numeric expressions that would bypass the protected slots (规则 N1-N5 below) and leaves
ordinary wording alone (等一下, 不一定, 第二只基金, 千万别).

All word lists are provisional until reviewed by a native speaker.
"""

import re

from fyp_evaluation.validation.stage0 import HAN, LATIN

_NUMERALS = "零〇一二两三四五六七八九十百千万亿"
_UNITS = "十百千万亿"
_RUN = re.compile(f"[{_NUMERALS}]+(?:点[{_NUMERALS}]+)*")
_VALUE_MARKERS = ("元", "块", "刀", "股", "单位", "岁", "％", "%")  # directly after a numeral run: a money or share expression
_DATE_MARKERS = ("年", "月", "日", "号")
_EVERYDAY_AFTER_ONE = ("年", "日")  # 一年 / 一日 are ordinary wording, not dates
_AFTER_QIANWAN = "别不要务记得"  # 千万别 / 千万不要 / 千万记得: the idiom "whatever you do", not ten million
_IDIOMS = {"万一", "一一"}
_AFTER_PLACEHOLDER_UNIT = re.compile(r"\]\](?:元|块|刀)")

# Bare affirmatives: an utterance that is exactly one of these (after punctuation and neutral fillers are removed) reads as
# an unqualified yes. 吧 / 吗 / 呢 and Singlish "ah" mark uncertainty or a question and are never removed; a question mark
# keeps the utterance out of this check.
_BARE_AFFIRMATIVES = {"可以", "好", "好的", "行", "没问题", "对", "是", "can", "ok", "okay", "sure", "yes", "yeah"}
_NEUTRAL_HAN = "啦啊呀哦嗯嘛"
_NEUTRAL_WORDS = {"lah", "lor", "leh", "mm", "mmm"}
_PUNCTUATION = re.compile(r"[。！!，,.、…~～；;：:\"'“”‘’()（）\-—–]+")


def bare_affirmative(text):
    """Whether the whole utterance is a single unqualified affirmative (a deliberately conservative constraint on the plan's shape)."""
    lowered = text.strip().lower()
    if "?" in lowered or "？" in lowered:
        return False
    tokens = [t for t in _PUNCTUATION.sub(" ", lowered).split() if t not in _NEUTRAL_WORDS]
    cleaned = " ".join("".join(c for c in token if c not in _NEUTRAL_HAN) for token in tokens).strip()
    return cleaned in _BARE_AFFIRMATIVES


def value_expression_errors(outside, extra_markers=()):
    """Numeric expressions written outside the slots.

    A "numeral run" is a maximal sequence of 零〇一二两三四五六七八九十百千万亿 (with 点 only between numerals). Rejected:
    N1 a run containing 点 (三点五); N2 a run of three or more numerals (二〇〇一, 五十万, 一百二); N3 a two-numeral run with a
    unit 十百千万亿 (十二, 十万, 两千, 千万 as an amount); N4 a run directly followed by a value marker (买五股, 五月, 三十元,
    a currency name); N5 百分之. Not rejected: a run after 第 (第二只基金), 千万 before 别/不/要/务/记/得, the idioms 万一 and
    一一, single numerals (一下, 一定, 不一定, 两只基金), unitless pairs (两三天) and 十分.
    """
    markers = tuple(sorted({*_VALUE_MARKERS, *extra_markers}, key=len, reverse=True))
    found = []
    for match in _RUN.finditer(outside):
        run, start, end = match.group(), match.start(), match.end()
        if (start and outside[start - 1] == "第") or run in _IDIOMS:
            continue
        following = outside[end:]
        if run == "千万" and following[:1] and following[0] in _AFTER_QIANWAN:
            continue
        numerals = run.replace("点", "")
        dated = any(following.startswith(m) for m in _DATE_MARKERS) and not (run == "一" and following.startswith(_EVERYDAY_AFTER_ONE))
        if "点" in run:
            found.append(f"{run!r} (a decimal)")
        elif len(numerals) >= 3:
            found.append(f"{run!r} (a multi-digit number)")
        elif len(numerals) == 2 and any(c in _UNITS for c in numerals):
            found.append(f"{run!r} (a number with a unit)")
        elif any(following.startswith(m) for m in markers) or dated:
            found.append(f"{run + following[:1]!r} (a number with a value marker)")
    if "百分之" in outside:
        found.append("'百分之' (a percentage)")
    return [f"text: Chinese numeric expressions are only allowed through placeholders, not in the template wording: {', '.join(found)}"] if found else []


def template_errors(ctx, lexicon, event, entry, outside):
    """Schema-2.0 mechanical errors for one event's template (messages without the event prefix)."""
    errors = []
    rules = ctx.language_requirements["script_rules"]
    if rules["han_characters"] == "FORBIDDEN" and HAN.search(outside):
        errors.append("text: Chinese characters are not allowed in this call's wording (frozen script rule)")
    if rules["latin_letters"] == "K01_IDENTIFIER_SPAN_ONLY" and LATIN.search(outside):
        errors.append("text: Latin letters are only allowed inside the identifier slot in this call's wording (frozen script rule)")
    pin = event.get("required_language")
    if pin == "zh" and not HAN.search(outside):
        errors.append("text: this event is pinned to Mandarin, so its wording needs Chinese characters")
    if pin == "en" and not LATIN.search(outside):
        errors.append("text: this event is pinned to English, so its wording needs English words")
    if HAN.search(outside):
        errors += value_expression_errors(outside, lexicon.han_currency_markers)
    if _AFTER_PLACEHOLDER_UNIT.search(entry["text"]):
        errors.append("text: 元/块/刀 directly after a placeholder is not a permitted currency expression")
    if event["kind"] == "WEAK_ACKNOWLEDGEMENT" and bare_affirmative(entry["text"]):
        errors.append("text: a planned weak acknowledgement must not be a bare unqualified affirmative (for example 可以 or Can lah)")
    return errors


# Public names for the G6 adjacency check (the same numeral and value-marker sets).
NUMERALS = _NUMERALS
VALUE_MARKERS = _VALUE_MARKERS
