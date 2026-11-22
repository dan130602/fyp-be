"""Overlapping-window plan and transcript reconciliation for the MERaLiON overlap pilot ("overlap-15-v1"). Pure text and sample arithmetic: no model, no audio.

Plan      windows of at most 30 s (480000 samples at 16 kHz) that advance by 15 s (240000), so neighbours share 15 s; the last window ends at the end of the
          recording. Every sample is decoded in two windows at positions 15 s apart (one near an edge when it is in the other's middle), except the first 15 s
          and the last stretch of the recording, which only one window covers. (An advance of 20 s would leave the middle 10 s of every stride in a single window.)
Text      every character of a raw window text belongs to exactly one token: `lead` (text before the first token, on token 0 only), `pre` (symbols glued to the front
          of a token: $ ¥ - − + ( " ...), `surface`, `post` (everything up to the next token's `pre`). Rendering the tokens of one window gives the raw text
          byte for byte, and a token that is kept in a merge keeps all four parts. Only text in a genuinely reconciled stretch (a disputed or duplicated stretch)
          can differ from the raw window text, and that is recorded.
Merge     each window is decoded on its own (raw text kept untouched). Windows are folded in order: the text built so far (its last window's tokens) is aligned
          with the next window's text by an overlap alignment, a global alignment in which the end of the earlier text is paired with the start of the later text
          and the rest of each is free. How much text overlaps is found by the alignment, not assumed from the 15 s: token density varies too much inside a window
          for a fixed share to be safe. An optional soft prior (`expected_overlap_share`) discourages an alignment that starts far from where the overlap should begin.
            * tokens both windows produced are kept once (a genuine repeat inside the overlap is kept as often as the windows agree on it);
            * tokens only one window produced are KEPT (a union: the aim is to recover what a window left out). THE UNION ALSO KEEPS A HALLUCINATION OR A LOOP
              FROM EITHER WINDOW. This is a documented policy, not a safeguard: long one-sided runs inside the aligned stretch and repeated loops are flagged in the
              seam record ("long_one_sided_run", "loop_in_one_sided_run") so they can be reviewed; nothing is removed on that basis;
            * where both produced different text for the same stretch, one version is kept, never both, chosen by `edge_position_heuristic`: the version whose
              TOKEN POSITION is further from its own window's edge. That is a position in the text, not a distance in audio time and not a confidence; no word
              times or scores exist. The disagreement is logged with both versions and flagged when a number or a negation is involved. A sign glued to a number
              (-5, −5) is part of the token's identity, so "-5%" against "5%" is a number conflict, not a match.
Unresolved if the two edges share no run of at least 4 tokens, nothing can be anchored. That is NOT a reconciliation and is reported as status "unresolved" (a first-
          class count, `summarise_seams`). Fallback, deterministic: keep both edge texts in full, in window order. Why: the usual cause is that one window produced
          nothing for the overlap (or the overlap holds no speech), and dropping text would re-create the omission the overlap exists to repair. Cost: if both
          windows did decode the overlap in different words, that passage appears twice. This cannot be told apart from text alone, so the record carries
          `probable_duplicate_passage` (share of the later edge's content words that also occur in the earlier edge) and both raw edge texts; a run with unresolved
          seams must be judged by its insertion count and by those flags, never by the merge alone.
No word times exist, so nothing here claims to know where in seconds a word was said. Every decision is written to the seam record.
"""
import difflib
import math
import re
import unicodedata
from decimal import Decimal

RATE = 16000
WINDOW = 30 * RATE
STRIDE = 15 * RATE
MERGE_VERSION = "2.3"      # recorded with every merged prediction; a different version is a different merger. Policy "m0" reproduces v2.1 exactly.
MINIMUM_ANCHOR = 4         # tokens in a run both windows share before a seam counts as anchored
SMALL_OVERLAP = 0.2        # an expected overlap share below this uses the small-overlap path (fill-to-30 windows share about a second): shorter anchors, a tighter position prior
DUPLICATE_MIN = 4          # an adjacent repeated run of this many tokens at a seam is flagged as a possible duplicate
LONG_ONE_SIDED = 8         # a run of this many tokens that only one window produced, inside the aligned stretch, is flagged
MATCH, MISMATCH, GAP = 2, -1, -2      # alignment scores
PRIOR = 0.25               # score lost per token the alignment start lies beyond the tolerance band around where the overlap should begin
AMBIGUITY_MARGIN = 3       # another alignment ending this close in score (and 4+ tokens elsewhere) makes the match ambiguous
PROBABLE_DUPLICATE = 0.5   # share of the later edge's content words that also occur in the earlier edge

_TOKEN = re.compile(r"[㐀-鿿]|\d+(?:[.,:]\d+)*|[^\W\d_㐀-鿿]+(?:['’][^\W\d_㐀-鿿]+)*", re.UNICODE)     # a Latin word ends where Han begins (g九五六 = g + 九 五 六)
_PREFIX = set("$¥€£₩₹￥＄-−－–—+＋±~≈#@([{\"'“‘«¿¡（【《「『")        # symbols that, glued to the front of a token, travel with it
_MINUS = set("-−－")
NEGATIONS = {"not", "no", "never", "cannot", "without", "neither", "nor", "none", "nothing", "nobody", "dont", "doesnt", "didnt", "wont", "wouldnt", "cant",
             "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt", "shouldnt", "couldnt", "不", "没", "沒", "未", "非", "别", "別", "无", "無", "勿"}
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
                "point", "percent", "零", "一", "二", "两", "兩", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "萬", "亿", "億"}


def plan_windows(total_samples, window=WINDOW, stride=STRIDE):
    """[(start, end)] covering the recording; no window longer than `window`; consecutive windows overlap by window - stride samples."""
    if total_samples <= 0:
        return []
    if window <= stride:
        raise ValueError("the stride must be shorter than the window or there is no overlap")
    out, start = [], 0
    while True:
        end = min(start + window, total_samples)
        out.append((start, end))
        if end == total_samples:
            return out
        start += stride


def plan_fill30(energy_plan, total_samples, window=WINDOW):
    """Option A: every energy window shorter than `window` is extended with REAL audio up to `window` (30 s), so the number of windows is unchanged. Returns
    [(start, end, owned_start, owned_end)] where (owned_start, owned_end) is the original energy interval, always inside (start, end). Rule, no reference text: a window
    already at the limit is unchanged; the last window is extended backwards (nothing follows it); every other window is extended forwards, and if the recording ends before
    that fills the window the remainder is added backwards. The extension repeats the head of the next window, so seams overlap by the extension, not by half a window."""
    out = []
    for k, (a, b) in enumerate(energy_plan):
        if b - a >= window:
            out.append((a, b, a, b))
        elif k == len(energy_plan) - 1:
            out.append((max(0, b - window), b, a, b))
        else:
            end = min(total_samples, a + window)
            out.append((max(0, end - window), end, a, b))
    return out


def seam_shares(windows):
    """Per-seam expected overlap share for merge_windows: for consecutive windows [(start, end), ...], the share of the EARLIER window's audio that the next window repeats."""
    shares = []
    for (a0, a1), (b0, b1) in zip(windows, windows[1:]):
        shares.append(max(0.0, min(1.0, (a1 - b0) / (a1 - a0))) if a1 > a0 else 0.0)
    return shares


# ---- merger policies (v2.2) -------------------------------------------------------------------------------------------------------------------
class Policy:
    """A named, versioned set of merge rules. m0 is v2.1 unchanged; every other policy changes only what its fields say (see PILOT_fg_v1.md)."""
    FIELDS = dict(currency=False,            # M1: number units compare their currency as well as their value
                  flags=False,               # M1: script-profile and divergence flags, retry triggers (never change the text)
                  position_prior=True,       # A1 ablation turns the soft position prior off
                  longer_side=True,          # A2 ablation turns the "much fuller version wins" rule off
                  exclude_isolated=False,    # M2-AX: drop a window whose two seams are both unresolved
                  anchor_tiebreak=False,     # M2-AC: an ambiguous alignment is re-run with a strong position prior and the choice is logged
                  coherent_blocks=False,     # M2-AC: conflict blocks touching a number, negation or consent term are taken whole from one window
                  containment=False,         # M2-AC: the fuller version is kept only if it contains the other one
                  relaxed_pass=False,        # M2-AC: an unresolved seam gets a second pass with anchors of 2 units inside the estimated overlap text region
                  dedup=False,               # M2-AC: a one-sided run that repeats what the other window supplied at the same seam is dropped
                  low_margin_choices=False,  # v2.3 experiment: a retry may decide a disputed region whose edge-position decision is a coin toss (|margin| < LOW_MARGIN), not only value/negation conflicts
                  apostrophes=False,         # v2.3 experiment (m2ax_ap): inside an equal run, when two windows spell a word the same except for an apostrophe, the surface WITH the apostrophe is kept (don't, it's, fund's)
                  sequential_join=False)     # v2.3 experiment (m2ax_seq): a disputed region at the end of the earlier window whose two versions are DIFFERENT text sharing a boundary phrase is joined, not chosen

    def __init__(self, name, **kw):
        self.name = name
        for key, default in self.FIELDS.items():
            setattr(self, key, kw.pop(key, default))
        if kw:
            raise TypeError(f"unknown policy fields {sorted(kw)}")


_M1 = dict(currency=True, flags=True)
_AC = dict(_M1, anchor_tiebreak=True, coherent_blocks=True, containment=True, relaxed_pass=True)
POLICIES = {"m0": Policy("m0"), "m1": Policy("m1", **_M1), "m2ax": Policy("m2ax", **_M1, exclude_isolated=True), "m2ac": Policy("m2ac", **_AC, dedup=True),
            "m2ac_nodup": Policy("m2ac_nodup", **_AC), "a1": Policy("a1", **_M1, position_prior=False), "a2": Policy("a2", **_M1, longer_side=False),
            "m2ax_seq": Policy("m2ax_seq", **_M1, exclude_isolated=True, sequential_join=True),
            "m2ax_ap": Policy("m2ax_ap", **_M1, exclude_isolated=True, apostrophes=True),
            "m2ax_seq_ap": Policy("m2ax_seq_ap", **_M1, exclude_isolated=True, sequential_join=True, apostrophes=True),
            "m2ax_seq_ap_lm": Policy("m2ax_seq_ap_lm", **_M1, exclude_isolated=True, sequential_join=True, apostrophes=True, low_margin_choices=True),
            "m2ac_x": Policy("m2ac_x", **_AC, dedup=True, exclude_isolated=True), "m2ac_x_nodup": Policy("m2ac_x_nodup", **_AC, exclude_isolated=True)}
LOW_MARGIN = 0.3                 # |left distance - right distance| below this: the edge-position heuristic is not informative (measured on the pilot + design decisions; frozen before the retries were decoded)
SEQUENTIAL_MIN_UNITS = 3        # v2.3 sequential join: shared boundary phrase of at least this many units (one Han character is one unit)
SEQUENTIAL_MAX_SIMILARITY = 0.5  # ... and the two versions must be clearly different text overall (similar versions are two renderings of the same speech)
_ACTIVE = POLICIES["m0"]
_FLIPS = frozenset()      # evaluation only (choice_oracle.py): occurrence ids (left window label, right window label, region ordinal in the seam) whose kept version is reversed
_CHOICES = {}              # M5 tie-breaks: (left window label, right window label, left norm text, right norm text) -> "left" | "right"; only for a value or negation conflict
_SEAM_KEY = None

# currency identity (M1). Canonical codes; "$" and "¥" are generic (unspecified dollar / yuan-or-yen) and compatible with their members, never with another family.
CURRENCY_SYMBOL = {"$": "$", "＄": "$", "¥": "¥", "￥": "¥", "€": "EUR", "£": "GBP"}
CURRENCY_CODE = {"usd": "USD", "sgd": "SGD", "rmb": "CNY", "cny": "CNY", "jpy": "JPY", "eur": "EUR", "gbp": "GBP"}
CURRENCY_WORD = {"dollar": "$", "dollars": "$", "yuan": "CNY", "renminbi": "CNY", "yen": "JPY", "euro": "EUR", "euros": "EUR", "pound": "GBP", "pounds": "GBP"}
CURRENCY_HAN = {"美元": "USD", "新元": "SGD", "人民币": "CNY", "日元": "JPY", "欧元": "EUR", "英镑": "GBP", "元": "CNY"}
_GENERIC = {"$": {"$", "USD", "SGD"}, "¥": {"¥", "CNY", "JPY"}}


def currencies_compatible(a, b):
    """True if two currencies (or None = unspecified) can be the same: equal, one unspecified, or a generic symbol against one of its members."""
    if a is None or b is None or a == b:
        return True
    return b in _GENERIC.get(a, ()) or a in _GENERIC.get(b, ())


def _currency_of(toks, i, j):
    """The currency of the number tokens toks[i:j] from evidence in the text only: a glued symbol (S$ and US$ through the letters before it), a code or currency word right before,
    or a currency word right after. None when the text gives none: an unspecified amount is never given a currency."""
    first = toks[i]
    for ch in first.pre:
        if ch in CURRENCY_SYMBOL:
            symbol = CURRENCY_SYMBOL[ch]
            if symbol == "$" and i > 0 and toks[i - 1].norm in ("s", "us") and toks[i - 1].post == "" and _adjacent(toks[i - 1], first):
                return "SGD" if toks[i - 1].norm == "s" else "USD"
            return symbol
    if i > 0 and _adjacent(toks[i - 1], first):
        before = toks[i - 1].norm
        if before in CURRENCY_CODE:
            return CURRENCY_CODE[before]
        if i >= 3 and "".join(t.norm for t in toks[i - 3:i]) == "人民币" and all(_adjacent(a, b) for a, b in zip(toks[i - 3:i], toks[i - 2:i + 1])):
            return "CNY"
    last = toks[j - 1]
    if j < len(toks) and _adjacent(last, toks[j]):
        after = toks[j].norm
        if after in CURRENCY_WORD:
            return CURRENCY_WORD[after]
        if after in CURRENCY_CODE:
            return CURRENCY_CODE[after]
        for n in (3, 2, 1):
            if j + n <= len(toks) and all(_adjacent(a, b) for a, b in zip(toks[j - 1:j + n - 1], toks[j:j + n])):
                word = "".join(t.norm for t in toks[j:j + n])
                if word in CURRENCY_HAN:
                    return CURRENCY_HAN[word]
    return None


class Token:
    """One token of a raw window text with every character that belongs to it. `norm` is what is compared; the other parts are what is rendered."""
    __slots__ = ("norm", "surface", "pre", "post", "lead", "origin", "index")

    def __init__(self, norm, surface, pre, post, lead, origin, index):
        self.norm, self.surface, self.pre, self.post, self.lead, self.origin, self.index = norm, surface, pre, post, lead, origin, index

    @property
    def text(self):
        return self.pre + self.surface + self.post


def tokens(text, origin=0):
    """Tokens of `text`; render(tokens(text)) == text exactly. Text with no token at all gives []."""
    found = list(_TOKEN.finditer(text))
    if not found:
        return []
    pres, signs, prev_end = [], [], 0
    for m in found:
        gap = text[prev_end:m.start()]
        k = len(gap)
        while k > 0 and gap[k - 1] in _PREFIX:
            k -= 1
        pres.append(gap[k:])
        # a minus glued to a number is a sign only if it follows whitespace or the start of the text (342-968 is a range, not 968 negated)
        signs.append(bool(gap[k:]) and gap[k:][0] in _MINUS and ((k > 0 and gap[k - 1].isspace()) or (k == 0 and prev_end == 0)))
        prev_end = m.end()
    out = []
    for i, m in enumerate(found):
        stop = found[i + 1].start() - len(pres[i + 1]) if i + 1 < len(found) else len(text)
        norm = unicodedata.normalize("NFKC", m.group()).lower().replace("'", "").replace("’", "")        # don't = dont: a spelling difference is not a conflict
        if signs[i] and any(c.isdigit() for c in norm):
            norm = "-" + norm
        lead = text[:m.start() - len(pres[0])] if i == 0 else ""
        out.append(Token(norm, m.group(), pres[i], text[m.end():stop], lead, origin, i))
    return out


def is_number(token):
    return any(c.isdigit() for c in token.norm) or token.norm in NUMBER_WORDS


def is_negation(token):
    return token.norm in NEGATIONS


def render(selected):
    """Text of a list of tokens, every character of every token kept. A space is put where the next token is not the original successor (a seam jump) and
    neither side already has whitespace there, so ASCII punctuation never ends up glued to a word from another window."""
    out, previous, last = [], None, ""
    for t in selected:
        piece = (t.lead if t.index == 0 else "") + t.pre + t.surface + t.post
        if previous is not None and not (t.origin == previous.origin and t.index == previous.index + 1) and last and not last.isspace() and piece[:1] and not piece[:1].isspace():
            out.append(" ")
        out.append(piece)
        if piece:
            last = piece[-1]
        previous = t
    return "".join(out)


def _copy(t, **changes):
    fields = {name: getattr(t, name) for name in Token.__slots__}
    fields.update(changes)
    return Token(**fields)


# ---- alignment units: numbers compared by value, never rewritten ---------------------------------------------------------------------------

EN_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
EN_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_HUNDRED = {"hundred": 100}
EN_BIG = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
HAN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
HAN_SMALL = {"十": 10, "百": 100, "千": 1000}
HAN_BIG = {"万": 10 ** 4, "萬": 10 ** 4, "亿": 10 ** 8, "億": 10 ** 8}
HAN_NUMERAL_CHARS = set(HAN_DIGITS) | set(HAN_SMALL) | set(HAN_BIG) | {"点", "點"}
_DIGITS_TOKEN = re.compile(r"-?\d+(?:[.,:]\d+)*$")


def _category(norm):
    if norm in EN_ONES:
        return "ones"
    if norm in EN_TEENS:
        return "teen"
    if norm in EN_TENS:
        return "tens"
    if norm in EN_HUNDRED:
        return "hundred"
    if norm in EN_BIG:
        return "big"
    if norm == "point":
        return "point"
    if norm == "percent":
        return "percent"
    return None


# which English number word may follow which inside one number ("fifty fifty" is two numbers, "twenty five" is one, "nine five six" reads digit by digit)
_FOLLOWS = {None: {"ones", "teen", "tens", "hundred", "big"},
            "ones": {"ones", "hundred", "big", "point", "percent"}, "teen": {"hundred", "big", "point", "percent"}, "tens": {"ones", "big", "point", "percent"},
            "hundred": {"ones", "teen", "tens", "big", "point", "percent"}, "big": {"ones", "teen", "tens", "hundred", "big", "point", "percent"},
            "point": {"ones"}, "percent": set()}


def _kind(token):
    norm = token.norm
    if _DIGITS_TOKEN.match(norm):
        return "digits"
    if _category(norm):
        return "en"
    if norm and all(c in HAN_NUMERAL_CHARS for c in norm):
        return "han"
    return None


def _adjacent(a, b):
    return a.origin == b.origin and b.index == a.index + 1


def _collect_run(toks, i):
    """End (exclusive) of the number run that starts at toks[i], or i when none does. Only tokens that are neighbours in one window join a run."""
    n, t = len(toks), toks[i]
    kind = _kind(t)
    if kind == "digits":
        j = i + 1

        def more():
            return j < n and _adjacent(toks[j - 1], toks[j])

        if more() and _kind(toks[j]) == "han" and (toks[j].norm[0] in HAN_SMALL or toks[j].norm[0] in HAN_BIG):
            while more() and _kind(toks[j]) == "han":
                j += 1
        elif more() and (toks[j].norm in EN_HUNDRED or toks[j].norm in EN_BIG):
            while more() and (toks[j].norm in EN_BIG or toks[j].norm in EN_HUNDRED):
                j += 1
        if more() and toks[j].norm == "percent":
            j += 1
        return j
    if kind == "han":
        j = i + 1
        while j < n and _adjacent(toks[j - 1], toks[j]) and _kind(toks[j]) == "han":
            j += 1
        return j
    if kind == "en":
        state, j = None, i
        while j < n and (j == i or _adjacent(toks[j - 1], toks[j])):
            norm = toks[j].norm
            if norm == "and" and j + 1 < n and state in ("hundred", "big") and _adjacent(toks[j], toks[j + 1]) and _category(toks[j + 1].norm) in ("ones", "teen", "tens"):
                j += 1
                continue
            category = _category(norm)
            if category is None or category not in _FOLLOWS[state]:
                break
            state, j = category, j + 1
        if j == i or all(_category(t_.norm) in ("point", "percent") for t_ in toks[i:j]):
            return i
        return j
    return i


def _plain(value):
    """A Decimal as a plain digit string without exponent or a useless decimal part."""
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _han_value(chars, start=None):
    """Positional value of a Han numeral string (十百千 within a section, 万亿 closing sections); `start` is a number already read (the 25 of 25万)."""
    total, section, num = Decimal(0), Decimal(0), start
    for c in chars:
        if c in HAN_DIGITS:
            num = Decimal(HAN_DIGITS[c])
        elif c in HAN_SMALL:
            section += (num if num is not None else Decimal(1)) * HAN_SMALL[c]
            num = None
        elif c in HAN_BIG:
            section += num if num is not None else Decimal(0)
            total += (section if section else Decimal(1)) * HAN_BIG[c]
            section, num = Decimal(0), None
        else:
            return None
    return total + section + (num if num is not None else Decimal(0))


def _han_keys(chars, start=None):
    """Candidate canonical values of a Han numeral string: the digit-by-digit reading (九五六零 = 9560) and the positional one (一百二十万 = 1200000)."""
    if "点" in chars or "點" in chars:
        cut = min(i for i, c in enumerate(chars) if c in "点點")
        whole, fraction = chars[:cut], chars[cut + 1:]
        if not all(c in HAN_DIGITS for c in fraction):
            return set()
        base = _han_keys(whole, start) if whole or start is not None else {"0"}
        return {b + "." + "".join(str(HAN_DIGITS[c]) for c in fraction) for b in base} if fraction else set()
    keys = set()
    if start is None and all(c in HAN_DIGITS for c in chars):
        keys.add("".join(str(HAN_DIGITS[c]) for c in chars))            # digit by digit; a single digit is the same either way
    if start is not None or any(c in HAN_SMALL or c in HAN_BIG for c in chars):
        value = _han_value(chars, start)
        if value is not None:
            keys.add(_plain(value))
    return keys


def _en_keys(words):
    """Canonical value of a grammatical run of English number words (digit words alone read digit by digit)."""
    percent = bool(words) and words[-1] == "percent"
    if percent:
        words = words[:-1]
    words = [w for w in words if w != "and"]
    fraction = ""
    if "point" in words:
        cut = words.index("point")
        fraction = "." + "".join(str(EN_ONES[w]) for w in words[cut + 1:] if w in EN_ONES)
        words = words[:cut]
    if words and len(words) >= 2 and all(w in EN_ONES for w in words):
        whole = "".join(str(EN_ONES[w]) for w in words)
    else:
        total, current = Decimal(0), Decimal(0)
        for w in words:
            if w in EN_ONES:
                current += EN_ONES[w]
            elif w in EN_TEENS:
                current += EN_TEENS[w]
            elif w in EN_TENS:
                current += EN_TENS[w]
            elif w in EN_HUNDRED:
                current = (current if current else Decimal(1)) * 100
            else:
                total += (current if current else Decimal(1)) * EN_BIG[w]
                current = Decimal(0)
        whole = _plain(total + current)
    return {whole + fraction + ("%" if percent else "")}


def _run_keys(run, percent_prefix=False):
    """Candidate canonical keys of a run of number tokens (a set of strings starting with '#'), or a key that only equals the identical text."""
    first = run[0]
    kind = _kind(first)
    pct = percent_prefix or run[-1].norm == "percent" or run[-1].post[:1] in ("%", "％")
    keys = set()
    if kind == "digits":
        base = first.norm.replace(",", "")
        rest = [t for t in run[1:] if t.norm != "percent"]
        if not rest:
            keys.add(base)
        else:
            try:
                value = Decimal(base)
            except Exception:                                                  # a time such as 10:30 followed by a unit: not a quantity
                value = None
            if value is not None and rest and _kind(rest[0]) == "han":
                got = _han_value("".join(t.norm for t in rest), value)
                if got is not None:
                    keys.add(_plain(got))
            elif value is not None:
                for t in rest:
                    value *= (EN_HUNDRED.get(t.norm) or EN_BIG.get(t.norm) or 1)
                keys.add(_plain(value))
    elif kind == "han":
        keys = _han_keys("".join(t.norm for t in run))
    elif kind == "en":
        keys = _en_keys([t.norm for t in run])
    keys = {"#" + k.rstrip("%") + ("%" if (pct or k.endswith("%")) else "") for k in keys}
    return frozenset(keys or {"#?" + " ".join(t.norm for t in run)})


class Unit:
    """What the aligner compares: one ordinary token, or a run of number tokens (digits with separators, English number words, Han numerals) whose candidate keys are
    canonical values. Two units match if their value keys intersect (25 = twenty five = 二十五 = 二五; 1,200,000 = 一百二十万 = one million two hundred thousand; 九五六零八一九 =
    9560819). A unit only POINTS at its original tokens: whichever side is kept, its characters are rendered exactly as the window wrote them."""
    __slots__ = ("tokens", "key", "keys", "text", "number", "currency")

    def __init__(self, tokens, keys=None):
        self.tokens = tokens
        self.currency = None
        self.keys = keys
        self.number = keys is not None
        self.key = None if keys is not None else tokens[0].norm
        self.text = " ".join(t.norm for t in tokens)


def same_unit(a, b):
    if a.number != b.number:
        return False
    if a.number:
        return bool(a.keys & b.keys) and (not _ACTIVE.currency or currencies_compatible(a.currency, b.currency))
    return a.key == b.key


def units_of(toks):
    out, i, n = [], 0, len(toks)
    while i < n:
        t = toks[i]
        if t.norm == "百" and i + 3 <= n - 1 and toks[i + 1].norm == "分" and toks[i + 2].norm == "之" and _adjacent(t, toks[i + 1]) and _adjacent(toks[i + 1], toks[i + 2]) \
                and _adjacent(toks[i + 2], toks[i + 3]) and _kind(toks[i + 3]) in ("han", "digits"):
            j = _collect_run(toks, i + 3)                                # 百分之五十 = 50 %
            out.append(Unit(toks[i:j], _run_keys(toks[i + 3:j], percent_prefix=True)))
            i = j
            continue
        j = _collect_run(toks, i)
        if j > i:
            unit = Unit(toks[i:j], _run_keys(toks[i:j]))
            if _ACTIVE.currency:
                unit.currency = _currency_of(toks, i, j)
            out.append(unit)
            i = j
        else:
            out.append(Unit([t]))
            i += 1
    return out


def _has_loop(words):
    """An n-gram (1 to 6 tokens) that repeats back to back at least three times."""
    for n in range(1, 7):
        for i in range(len(words) - 3 * n + 1):
            if words[i:i + n] == words[i + n:i + 2 * n] == words[i + 2 * n:i + 3 * n]:
                return True
    return False


def align_units(a_units, b_units, start_cost=None):
    """Overlap alignment of the end of the earlier text (`a_units`) with the start of the later text (`b_units`) over alignment units (see Unit): the alignment ends at
    the last unit of `a_units`, begins at b_units[0], and is free to begin anywhere in `a_units` and to end anywhere in `b_units`. Returns (first a unit used, ops, last b
    unit used + 1, ambiguity), ops in order as ("equal"|"sub"|"left"|"right", a_index_or_None, b_index_or_None). Global over the whole overlap, so a repeated phrase is
    not matched in the wrong place as easily as with a greedy longest-block match. `start_cost[i]` is a soft penalty for starting at unit i (the position prior).
    `ambiguity` is the b index of another alignment whose score is within AMBIGUITY_MARGIN of the best and that ends 4 or more units elsewhere, else None."""
    n, m = len(a_units), len(b_units)
    start_cost = start_cost or [0.0] * (n + 1)
    by_text, by_value = {}, {}
    for j, u in enumerate(b_units):
        if u.number:
            for key in u.keys:
                by_value.setdefault(key, []).append(j)
        else:
            by_text.setdefault(u.key, []).append(j)
    equal = []
    for u in a_units:
        row = [False] * m
        for j in (by_text.get(u.key, ()) if not u.number else [j for key in u.keys for j in by_value.get(key, ())]):
            if u.number and _ACTIVE.currency and not currencies_compatible(u.currency, b_units[j].currency):
                continue                                                   # the same digits in different currencies are different amounts
            row[j] = True
        equal.append(row)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        score[0][j] = j * GAP - start_cost[0]
    score[0][0] = -start_cost[0]
    for i in range(1, n + 1):
        row, above, eq = score[i], score[i - 1], equal[i - 1]
        row[0] = -start_cost[i]                              # skipping the first i units of the earlier text costs nothing but the prior
        for j in range(1, m + 1):
            best = above[j - 1] + (MATCH if eq[j - 1] else MISMATCH)
            up, left = above[j] + GAP, row[j - 1] + GAP
            row[j] = best if best >= up and best >= left else (up if up >= left else left)
    end = max(range(m + 1), key=lambda j: (score[n][j], -j))       # a prefix of b of any length; ties: the shorter
    ambiguity = None
    for j in range(m + 1):
        if abs(j - end) >= 4 and score[n][j] >= score[n][end] - AMBIGUITY_MARGIN:
            ambiguity = j
            break
    i, j, ops = n, end, []
    while j > 0:
        if i > 0 and score[i][j] == score[i - 1][j - 1] + (MATCH if equal[i - 1][j - 1] else MISMATCH):
            ops.append(("equal" if equal[i - 1][j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and score[i][j] == score[i - 1][j] + GAP:
            ops.append(("left", i - 1, None))
            i -= 1
        else:
            ops.append(("right", None, j - 1))
            j -= 1
    ops.reverse()
    return i, ops, end, ambiguity


def align_overlap(tail, head, expected_share=None):
    """Token-level entry point: units are built from the two token lists and aligned (see align_units). Indices in the result are unit indices."""
    tu, hu = units_of(tail), units_of(head)
    return align_units(tu, hu)


def _positions(units):
    pos, total = [], 0
    for u in units:
        pos.append(total)
        total += len(u.tokens)
    pos.append(total)
    return pos


def _has_apostrophe(surface):
    return "'" in surface or "’" in surface


def _flat(units, indices):
    return [t for i in indices for t in units[i].tokens]


CONSENT_WORDS = {"yes", "yeah", "agree", "agreed", "understand", "understood", "accept", "accepted", "consent", "confirm", "confirmed"}


def _is_subsequence(short, long_):
    it = iter(long_)
    return all(x in it for x in short)


def _contains_run(run, seq):
    n = len(run)
    return n > 0 and any(seq[i:i + n] == run for i in range(len(seq) - n + 1))


def _unit_critical(unit):
    return unit.number or any(is_negation(t) or t.norm in CONSENT_WORDS for t in unit.tokens)


def _join_blocks(ops, tu, hu):
    """M2-AC coherent conflict blocks: two non-equal stretches separated by at most 2 equal units become one block when either touches a number, a negation or a consent term, so
    an amount or a 'not' is never assembled from pieces of two versions. The equal units between them become ('joined', i, j) and are taken with the block."""
    n, out, k = len(ops), list(ops), 0
    runs = []
    while k < n:
        if out[k][0] != "equal":
            a = k
            while k < n and out[k][0] != "equal":
                k += 1
            runs.append((a, k))
        else:
            k += 1
    for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
        if 1 <= a2 - b1 <= 2:
            touched = any((o[1] is not None and _unit_critical(tu[o[1]])) or (o[2] is not None and _unit_critical(hu[o[2]])) for o in out[a1:b1] + out[a2:b2])
            if touched:
                for q in range(b1, a2):
                    out[q] = ("joined", out[q][1], out[q][2])
    return out


def _sequential_join(left_tokens, right_tokens, left_units, tail_units):
    """Number of shared boundary units if this disputed region is a sequential continuation (see the policy field), else None. Reference-free: only the two versions and their position."""
    if not left_units or max(left_units) != tail_units - 1:               # the region must reach the end of the earlier window's text
        return None
    left, right = [t.norm for t in left_tokens], [t.norm for t in right_tokens]
    best = 0
    for n in range(min(len(left), len(right)) - 1, SEQUENTIAL_MIN_UNITS - 1, -1):    # both versions must have material beyond the shared phrase
        if left[-n:] == right[:n]:
            best = n
            break
    if not best or len(set(left[-best:])) < 2:
        return None
    near = left_tokens[-best - 2:] + right_tokens[:best + 2]                 # a value, negation or consent word at or next to the join is never merged silently
    if any(is_number(t) or is_negation(t) for t in near):
        return None
    if difflib.SequenceMatcher(None, left, right, autojunk=False).ratio() >= SEQUENTIAL_MAX_SIMILARITY:
        return None
    return best


def reconcile(tail, head, n_left, n_right, expected_share=None, _minimum_anchor=None, _allow_relaxed=True):
    """Merge the end of the earlier text with the later text. Returns (tokens replacing `tail`, record). `n_left` and `n_right` are the token counts of the two
    windows, used by `edge_position_heuristic` (which version of a disputed stretch sits further from its own window's edge, by token position).
    Inside the aligned stretch both windows heard the same audio, so any stretch where BOTH produced text but not equal text is one disputed region: exactly one
    version is kept, never both. Text only one window produced stays (the union)."""
    record = {"status": "anchored", "left_edge_text": render(tail), "right_edge_text": render(head), "tail_tokens": len(tail), "head_tokens": len(head),
              "anchors": [], "left_only": [], "right_only": [], "replaced": [], "format_variants": [], "disputed_regions": 0, "duplicate_resolved": 0,
              "unanchored": False, "duplicates": [], "numeric_conflicts": 0, "negation_conflicts": 0,
              "currency_unverified": [], "one_sided_number_runs": 0, "one_sided_negation_runs": 0, "long_one_sided_runs": [], "ambiguous_alignment": False, "probable_duplicate_passage": None,
              "overlap_start_share": None, "overlap_position_unexpected": False, "edge_basis": "edge_position_heuristic", "risk_flags": []}
    if not tail or not head:                                  # one side produced nothing here: keep whatever the other side has
        record["status"] = "one_side_empty"
        return list(tail) + list(head), record
    tu, hu = units_of(tail), units_of(head)
    t_pos, h_pos = _positions(tu), _positions(hu)
    small = expected_share is not None and expected_share < SMALL_OVERLAP

    def prior(multiplier=1.0):
        if expected_share is None or len(tail) < 20:
            return None
        band = min(0.25, 2.5 * expected_share) if small else 0.25                  # small overlap: the start of the aligned stretch must lie close to the end of the earlier text
        expected, tolerance = round((1 - expected_share) * len(tail)), max(3, round(band * len(tail)))
        return [multiplier * PRIOR * max(0, abs(t_pos[i] - expected) - tolerance) for i in range(len(tu) + 1)]

    start_cost = prior() if _ACTIVE.position_prior else None
    start, ops, end, ambiguity = align_units(tu, hu, start_cost)
    if _ACTIVE.anchor_tiebreak and ambiguity is not None and prior(4.0) is not None:       # M2-AC: an ambiguous anchor is decided by the expected start, not left to a soft cost
        start, ops, end, ambiguity = align_units(tu, hu, prior(4.0))
        record["tie_break"] = "expected_start_strong_prior"
    run = best = 0
    for op, _, _ in ops:
        run = run + 1 if op == "equal" else 0
        best = max(best, run)
    minimum_anchor = _minimum_anchor or (max(2, min(MINIMUM_ANCHOR, round(0.5 * expected_share * len(tail)))) if small else MINIMUM_ANCHOR)
    record["minimum_anchor"] = minimum_anchor
    if best < minimum_anchor and _ACTIVE.relaxed_pass and _allow_relaxed and expected_share is not None and len(tu) >= 8 and len(hu) >= 8:
        # M2-AC second pass: anchors of 2 units, searched only inside the ESTIMATED overlap text region (an approximation: 1.5 x the audio share of each text, never an audio interval)
        cut_t = max(0, len(tu) - math.ceil(min(1.0, 1.5 * expected_share) * len(tu)))
        cut_h = math.ceil(min(1.0, 1.5 * expected_share) * len(hu))
        sub_tail, sub_head = _flat(tu, range(cut_t, len(tu))), _flat(hu, range(0, cut_h))
        merged_sub, sub = reconcile(sub_tail, sub_head, n_left, n_right, None, _minimum_anchor=2, _allow_relaxed=False)
        if sub["status"] == "anchored":
            sub["left_edge_text"], sub["right_edge_text"] = record["left_edge_text"], record["right_edge_text"]
            sub["tail_tokens"], sub["head_tokens"] = len(tail), len(head)
            sub["relaxed_second_pass"] = {"anchor_units_min": 2, "tail_region_units": len(tu) - cut_t, "head_region_units": cut_h, "estimate": "1.5 x audio share of each text"}
            sub["risk_flags"] = sorted(set(sub["risk_flags"] + ["relaxed_second_pass"]))
            return list(tail[:t_pos[cut_t]]) + merged_sub + list(head[h_pos[cut_h]:]), sub
    if best < minimum_anchor:
        record["status"], record["unanchored"] = "unresolved", True
        content = lambda tokens_: {t.norm for t in tokens_ if len(t.norm) >= 4 or any("㐀" <= c <= "鿿" for c in t.norm)}
        later, earlier = content(head[:40]), content(tail)
        record["probable_duplicate_passage"] = bool(later) and len(later & earlier) / len(later) >= PROBABLE_DUPLICATE
        record["risk_flags"] += ["unresolved"] + (["unresolved_small_overlap"] if small else []) + (["probable_duplicate_passage"] if record["probable_duplicate_passage"] else [])
        return list(tail) + list(head), record
    n_left, n_right = max(1, n_left), max(1, n_right)
    share = t_pos[start] / len(tail) if len(tail) else 0
    record["overlap_start_share"] = round(share, 2)
    if expected_share is not None:
        record["overlap_position_unexpected"] = len(tail) >= n_left * 0.9 and abs(share - (1 - expected_share)) > 0.3 and len(head) >= 20
    else:
        record["overlap_position_unexpected"] = len(tail) >= n_left * 0.9 and not 0.25 <= share <= 0.8 and len(head) >= 20
    if ambiguity is not None:
        record["ambiguous_alignment"] = True
        record["ambiguous_alternative_head_tokens"] = h_pos[ambiguity]
        record["risk_flags"].append("ambiguous_alignment")
    if _ACTIVE.coherent_blocks:
        ops = _join_blocks(ops, tu, hu)
    selected = list(tail[:t_pos[start]])                           # before the aligned stretch: only the earlier window has it
    if start:
        record["left_only"].append(" ".join(t.norm for t in selected))

    def left_distance(indices):       # edge_position_heuristic: share of each window's tokens that lies beyond the stretch, towards its own edge
        return (len(tail) - (t_pos[min(indices)] + t_pos[max(indices) + 1] - 1) / 2) / n_left

    def right_distance(indices):
        return ((h_pos[min(indices)] + h_pos[max(indices) + 1] - 1) / 2) / n_right

    k = 0
    while k < len(ops):
        kind = ops[k][0]
        run_ops = []
        if kind == "equal":
            while k < len(ops) and ops[k][0] == "equal":
                run_ops.append(ops[k])
                k += 1
            lefts, rights = [o[1] for o in run_ops], [o[2] for o in run_ops]
            use_left = left_distance(lefts) >= right_distance(rights)
            if _ACTIVE.apostrophes:
                for ti, hj in zip(lefts, rights):
                    pick, other = (tu[ti].tokens, hu[hj].tokens) if use_left else (hu[hj].tokens, tu[ti].tokens)
                    if len(pick) == 1 and len(other) == 1 and not _has_apostrophe(pick[0].surface) and _has_apostrophe(other[0].surface):
                        pick = other                  # same word, same norm: the window that wrote the apostrophe wrote it in standard orthography
                    selected += pick
            else:
                selected += _flat(tu, lefts) if use_left else _flat(hu, rights)
            for ti, hj in zip(lefts, rights):
                if _ACTIVE.currency and tu[ti].number and tu[ti].currency != hu[hj].currency:     # compatible but not identical currency evidence: never verified
                    record["currency_unverified"].append({"left": render(tu[ti].tokens), "right": render(hu[hj].tokens), "left_currency": tu[ti].currency, "right_currency": hu[hj].currency})
                if tu[ti].number and tu[ti].text != hu[hj].text:           # the same value written differently: one surface kept, logged, not a conflict
                    record["format_variants"].append({"left": tu[ti].text, "right": hu[hj].text, "kept": "left" if use_left else "right",
                                                      "left_text": render(tu[ti].tokens), "right_text": render(hu[hj].tokens)})
            if len(run_ops) >= MINIMUM_ANCHOR:
                record["anchors"].append({"size": len(run_ops), "text": " ".join(tu[i].text for i in lefts)})
            continue
        while k < len(ops) and ops[k][0] != "equal":
            run_ops.append(ops[k])
            k += 1
        lefts = [o[1] for o in run_ops if o[1] is not None]
        rights = [o[2] for o in run_ops if o[2] is not None]
        if lefts and rights:
            # both windows wrote something for the same audio and it is not the same: one disputed region, one version kept (never both)
            left_tokens, right_tokens = _flat(tu, lefts), _flat(hu, rights)
            joined = _sequential_join(left_tokens, right_tokens, lefts, len(tu)) if _ACTIVE.sequential_join else None
            if joined is not None:           # two DIFFERENT versions that continue one another across a shared boundary phrase: both are kept, the phrase once (never an either-or choice)
                n_shared = joined
                record.setdefault("sequential_joins", []).append({"shared_units": n_shared, "shared_text": " ".join(t.norm for t in left_tokens[-n_shared:]), "left_text": render(left_tokens), "right_text": render(right_tokens),
                                                                  "rule": f"v2.3 sequential join: region at the end of the earlier window, versions share >= {SEQUENTIAL_MIN_UNITS} boundary units and are < {SEQUENTIAL_MAX_SIMILARITY} similar; experimental heuristic"})
                record["risk_flags"].append("sequential_join")
                record["disputed_regions"] += 1
                selected += left_tokens + right_tokens[n_shared:]
                continue
            if _ACTIVE.containment:
                short, long_ = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
                substantial = len(long_) > len(short) and _is_subsequence([t.norm for t in short], [t.norm for t in long_])
                label = "containment"
            else:
                substantial = _ACTIVE.longer_side and max(len(left_tokens), len(right_tokens)) >= 2 * min(len(left_tokens), len(right_tokens)) and abs(len(left_tokens) - len(right_tokens)) >= 6
                label = "longer_side"
            if substantial:                                              # one side wrote much more: keep the fuller one rather than lose a recovered stretch
                keep_left, basis = len(left_tokens) > len(right_tokens), label
            else:
                keep_left, basis = left_distance(lefts) >= right_distance(rights), "edge_position_heuristic"
            numeric = any(is_number(t) for t in left_tokens + right_tokens)
            negation = any(is_negation(t) for t in left_tokens + right_tokens)
            margin = left_distance(lefts) - right_distance(rights)
            if _CHOICES and (numeric or negation or (_ACTIVE.low_margin_choices and abs(margin) < LOW_MARGIN)):
                forced = _CHOICES.get((_SEAM_KEY[0], _SEAM_KEY[1], "region", len(record["replaced"])))          # occurrence-specific (seam, ordinal), used by the low-margin retries
                if forced is None and (numeric or negation):
                    forced = _CHOICES.get((_SEAM_KEY[0], _SEAM_KEY[1], " ".join(t.norm for t in left_tokens), " ".join(t.norm for t in right_tokens)))
                if forced is not None:                                   # a retry supported exactly one of the two EXISTING versions; a third value is never adopted
                    keep_left, basis = forced == "left", "retry_tiebreak"
            if _FLIPS and (_SEAM_KEY[0], _SEAM_KEY[1], len(record["replaced"])) in _FLIPS:      # a single-decision flip experiment: this ONE disputed region keeps the other version
                keep_left, basis = not keep_left, basis + "+flipped"
            pure = all(o[0] == "sub" for o in run_ops)
            record["numeric_conflicts"] += numeric
            record["negation_conflicts"] += negation
            record["disputed_regions"] += 1
            record["duplicate_resolved"] += 0 if pure else 1
            record["replaced"].append({"left": " ".join(t.norm for t in left_tokens), "right": " ".join(t.norm for t in right_tokens), "kept": "left" if keep_left else "right",
                                       "basis": basis, "edge_margin": round(margin, 3), "kind": "substitution" if pure else "disputed_region", "number_involved": numeric, "negation_involved": negation,
                                       "left_text": render(left_tokens), "right_text": render(right_tokens),
                                       "value_conflict": ({"candidates": [render(left_tokens), render(right_tokens)], "chosen": "left" if keep_left else "right", "resolved": False,
                                                           "provenance": f"{basis}: not evidence that the chosen value is correct; both candidates are kept in this record"} if numeric or negation else None)})
            selected += left_tokens if keep_left else right_tokens
        else:
            side_units, indices, side = (tu, lefts, "left") if lefts else (hu, rights, "right")
            chosen = _flat(side_units, indices)
            if _ACTIVE.dedup and len(chosen) >= 6:                      # M2-AC: a run that repeats what the OTHER window supplied inside this seam's aligned stretch is not added a second time
                other = [t.norm for t in (head[:h_pos[end]] if side == "left" else tail[t_pos[start]:])]
                if _contains_run([t.norm for t in chosen], other):
                    record.setdefault("duplicate_one_sided_runs", []).append({"side": side, "tokens": len(chosen), "text": " ".join(t.norm for t in chosen)})
                    record["risk_flags"].append("duplicate_one_sided_run")
                    continue
            selected += chosen
            words = [t.norm for t in chosen]
            record["left_only" if side == "left" else "right_only"].append(" ".join(words))
            looped = _has_loop(words)
            if len(words) >= LONG_ONE_SIDED or looped:        # inside the aligned stretch: a recovered omission and a hallucination look the same here
                record["long_one_sided_runs"].append({"side": side, "tokens": len(words), "loop": looped, "text": " ".join(words)})
                record["risk_flags"] += (["long_one_sided_run"] if len(words) >= LONG_ONE_SIDED else []) + (["loop_in_one_sided_run"] if looped else [])
    selected += list(head[h_pos[end]:])                            # after the aligned stretch: only the later window has it
    if h_pos[end] < len(head):
        record["right_only"].append(" ".join(t.norm for t in head[h_pos[end]:]))
    for side, words in (("left", [t.norm for t in tail[:t_pos[start]]]), ("right", [t.norm for t in head[h_pos[end]:]])):     # outside the aligned stretch only a loop is flagged
        if _has_loop(words):
            record["long_one_sided_runs"].append({"side": side, "tokens": len(words), "loop": True, "text": " ".join(words), "outside_aligned_stretch": True})
            record["risk_flags"].append("loop_in_one_sided_run")
    for words in record["left_only"] + record["right_only"]:         # text only one window produced: kept, but a lone number or negation is worth a look
        marks = [Token(w, w, "", "", "", 0, 0) for w in words.split()]
        record["one_sided_number_runs"] += any(is_number(t) for t in marks)
        record["one_sided_negation_runs"] += any(is_negation(t) for t in marks)
    if record["numeric_conflicts"]:
        record["risk_flags"].append("number_conflict")
    if record["negation_conflicts"]:
        record["risk_flags"].append("negation_conflict")
    flat = [t.norm for t in selected]
    for n in range(DUPLICATE_MIN, 13):
        for q in range(len(flat) - 2 * n + 1):
            if flat[q:q + n] == flat[q + n:q + 2 * n]:
                record["duplicates"].append(" ".join(flat[q:q + n]))
    record["duplicates"] = sorted(set(record["duplicates"]))
    if record["duplicates"]:
        record["risk_flags"].append("possible_duplicate_run")
    record["risk_flags"] = sorted(set(record["risk_flags"]))
    return selected, record


HAN_RANGE = ("\u3400", "\u9fff")
DIVERGENCE = 0.6           # cross-language divergence flag: a window's Han share differs from EACH neighbour's by more than this (a retry trigger, never a deletion)


def han_share(toks):
    """Share of Han characters among the tokens of one window's WHOLE text (a script profile; there are no word times, so it is never computed over an audio interval)."""
    if not toks:
        return 0.0
    return sum(1 for t in toks if HAN_RANGE[0] <= t.norm[:1] <= HAN_RANGE[1]) / len(toks)


def _merge_core(own, texts, shares, skip=frozenset(), bounds=None, labels=None):
    """Merge the windows in order, leaving out the windows in `skip`. Two neighbours around a skipped window are joined by concatenation (an abutting join): no alignment is attempted
    because their audio does not overlap, so aligning them as overlapping transcripts would invent a correspondence. Returns (accumulated tokens, carried text, seam records)."""
    active = [k for k in range(len(own)) if k not in skip]
    accumulated, seams, carried = [], [], ""
    first = active[0]
    if own[first]:
        accumulated = list(own[first])
    else:
        carried = texts[first]
    previous = own[first]
    last = first
    for k in active[1:]:
        joined = k - last > 1 and (bounds is None or bounds[last][1] <= bounds[k][0])      # windows in between were excluded AND the two neighbours really share no audio
        if k - last > 1 and not joined and bounds is not None:
            shares = list(shares) if isinstance(shares, (list, tuple)) else [shares] * (len(own) - 1)
            shares[k - 1] = max(0.0, min(1.0, (bounds[last][1] - bounds[k][0]) / max(1, bounds[last][1] - bounds[last][0])))
        if joined:                                             # the windows in between were excluded: abutting join, no alignment
            skipped = list(range(last + 1, k))
            record = {"seam": last, "windows": [last, k], "status": "abutting_join", "seam_kind": "abutting_join", "excluded_windows": skipped,
                      "excluded_text": [texts[q] for q in skipped], "risk_flags": [], "unanchored": False,
                      "note": "no alignment attempted: the neighbours do not share audio; words cut at this join can be lost"}
            if own[k]:
                accumulated = accumulated + list(own[k])
                previous = own[k]
            seams.append(record)
            last = k
            continue
        if not own[k]:                                        # a window with no token: nothing to reconcile, its characters are kept in place
            if accumulated:
                accumulated[-1] = _copy(accumulated[-1], post=accumulated[-1].post + texts[k])
            else:
                carried += texts[k]
            seams.append({"seam": k - 1, "windows": [k - 1, k], "status": "one_side_empty", "tokenless_window": k, "risk_flags": [], "unanchored": False})
            last = k
            continue
        n_tail = min(len(accumulated), len(previous))
        share = shares[k - 1] if isinstance(shares, (list, tuple)) else shares
        global _SEAM_KEY
        _SEAM_KEY = (labels[last], labels[k]) if labels else (last, k)
        merged, record = reconcile(accumulated[len(accumulated) - n_tail:], own[k], len(previous), len(own[k]), share)
        record["seam"] = k - 1
        record["windows"] = [last, k]
        seams.append(record)
        accumulated = accumulated[:len(accumulated) - n_tail] + merged
        previous = own[k]
        last = k
    return accumulated, carried, seams


def merge_windows(texts, *, expected_overlap_share=None, policy="m0", bounds=None, provenance=False, skip=None, labels=None, choices=None, flips=None):
    """One transcript from the raw text of each window in order, plus a record per seam. Raw texts are never altered; only the result is built. One window is
    returned unchanged. `expected_overlap_share` is the share of the EARLIER window's audio that the next window repeats (0.5 for the planned 15 s overlap of 30 s windows):
    one number for every seam, or a list with one entry per seam (windows whose overlaps differ, such as fill-to-30 windows) or None for entries without a prior. It enables the
    soft position prior; below SMALL_OVERLAP the small-overlap path (shorter anchors, tighter prior) is used.
    `policy` names a rule set (POLICIES; "m0" is v2.1 unchanged). `bounds` is [(start_sample, end_sample)] per window: the M2-AX exclusion needs it to check that the neighbours of an
    excluded window really abut. With provenance=True a third value is returned: per window, the token spans retained, discarded and the seam outcome (source window and token indexes)."""
    global _ACTIVE, _CHOICES, _FLIPS
    pol = POLICIES[policy] if isinstance(policy, str) else policy
    previous_policy, _ACTIVE = _ACTIVE, pol
    previous_choices, _CHOICES = _CHOICES, (choices or {})
    previous_flips, _FLIPS = _FLIPS, frozenset(flips or ())
    try:
        if not texts:
            return ("", [], {}) if provenance else ("", [])
        if len(texts) == 1:
            return (texts[0], [], {"policy": pol.name, "windows": [{"window": 0, "retained": "all", "discarded": []}]}) if provenance else (texts[0], [])
        own = [tokens(t, k) for k, t in enumerate(texts)]
        explicit_skip = skip
        skip, dry_seams = set(), None
        if explicit_skip is not None:                                   # the caller fixes which windows are excluded (a retry run keeps the base merge's exclusions)
            skip = set(explicit_skip)
            if skip:
                _, _, dry_seams = _merge_core(own, texts, expected_overlap_share, labels=labels)
        elif pol.exclude_isolated and len(texts) >= 3:
            if bounds is None:
                raise ValueError("the M2-AX exclusion needs the window bounds to check that the neighbours abut")
            _, _, dry_seams = _merge_core(own, texts, expected_overlap_share)
            status = {s_["seam"]: s_["status"] for s_ in dry_seams if "seam" in s_}
            for k in range(1, len(texts) - 1):
                abut = bounds[k - 1][1] >= bounds[k + 1][0]            # the earlier neighbour ends where (or after) the later one starts: no audio is left uncovered
                if status.get(k - 1) == "unresolved" and status.get(k) == "unresolved" and (k - 1) not in skip and abut and own[k - 1] and own[k + 1]:
                    skip.add(k)
        accumulated, carried, seams = _merge_core(own, texts, expected_overlap_share, frozenset(skip), bounds=bounds, labels=labels)
        for k in skip:                                             # the original seam records of an excluded window stay, marked excluded: they are still possible retry triggers
            for original in dry_seams:
                if original.get("seam") in (k - 1, k):
                    kept = dict(original)
                    kept["excluded"] = True
                    kept["excluded_window"] = k
                    seams.append(kept)
        if carried and accumulated:
            accumulated[0] = _copy(accumulated[0], lead=carried + accumulated[0].lead)
        text = render(accumulated) if accumulated else carried
        extra = None
        if pol.flags:
            shares_han = [han_share(o) for o in own]
            for s_ in seams:
                if s_.get("excluded") or "windows" not in s_:
                    continue
                s_["han_share"] = [round(shares_han[w], 3) for w in s_["windows"]]
            for k in range(len(own)):
                neighbours = [q for q in (k - 1, k + 1) if 0 <= q < len(own)]
                touching = [s_ for s_ in seams if k in s_.get("windows", ()) and s_["status"] == "unresolved" and not s_.get("excluded")]
                if touching and neighbours and all(abs(shares_han[k] - shares_han[q]) > DIVERGENCE for q in neighbours):
                    for s_ in touching:
                        s_["risk_flags"] = sorted(set(s_["risk_flags"] + ["cross_language_divergence"]))
                        s_["divergent_window"] = k
        if provenance:
            kept_idx = {}
            for t in accumulated:
                kept_idx.setdefault(t.origin, set()).add(t.index)
            wins = []
            for k, o in enumerate(own):
                got = kept_idx.get(k, set())
                dropped = [i for i in range(len(o)) if i not in got]
                spans, start = [], None
                for i in range(len(o) + 1):
                    if i < len(o) and i in dropped:
                        start = i if start is None else start
                    elif start is not None:
                        spans.append([start, i - 1])
                        start = None
                wins.append({"window": k, "tokens": len(o), "retained_tokens": len(got), "discarded_token_spans": spans, "excluded": k in skip})
            extra = {"policy": pol.name, "version": MERGE_VERSION, "windows": wins}
        return (text, seams, extra) if provenance else (text, seams)
    finally:
        _ACTIVE, _CHOICES, _FLIPS = previous_policy, previous_choices, previous_flips


def summarise_seams(seams):
    """Counts a run must report: seams by status, unresolved seams (and how many look like duplicated passages), ambiguous alignments, conflicts, flags."""
    out = {"seams": len([x for x in seams if not x.get("excluded")]), "excluded_seam_records": sum(1 for x in seams if x.get("excluded")), "abutting_join": 0, "anchored": 0, "unresolved": 0, "one_side_empty": 0, "probable_duplicate_passages": 0, "ambiguous_alignments": 0,
           "number_conflicts": 0, "negation_conflicts": 0, "long_one_sided_runs": 0, "loops_in_one_sided_runs": 0, "possible_duplicate_runs": 0,
           "format_variants": 0, "disputed_regions": 0, "duplicate_resolved": 0, "value_conflicts_unresolved": 0, "unresolved_small_overlap": 0}
    for s in seams:
        if s.get("excluded"):                                  # the original record of an excluded window's seam: kept for the retry triggers, not counted as a seam of the merge
            continue
        out[s["status"]] += 1
        out["probable_duplicate_passages"] += bool(s.get("probable_duplicate_passage"))
        out["ambiguous_alignments"] += bool(s.get("ambiguous_alignment"))
        out["number_conflicts"] += s.get("numeric_conflicts", 0)
        out["negation_conflicts"] += s.get("negation_conflicts", 0)
        out["long_one_sided_runs"] += len(s.get("long_one_sided_runs", []))
        out["loops_in_one_sided_runs"] += sum(1 for r in s.get("long_one_sided_runs", []) if r["loop"])
        out["possible_duplicate_runs"] += bool(s.get("duplicates"))
        out["format_variants"] += len(s.get("format_variants", []))
        out["disputed_regions"] += s.get("disputed_regions", 0)
        out["duplicate_resolved"] += s.get("duplicate_resolved", 0)
        out["value_conflicts_unresolved"] += sum(1 for r in s.get("replaced", []) if r.get("value_conflict"))
        out["unresolved_small_overlap"] += "unresolved_small_overlap" in s.get("risk_flags", [])
    return out
